"""Worker functions for Xet subprocess isolation.

These functions run inside child processes spawned by ``XetSubprocessRunner``.
They import ``hf_xet`` locally (never at module level) so the Rust .pyd
extension is loaded only in the child process — safe to terminate without
affecting the main process.

Each worker:
1. Receives a plain ``dict`` of parameters (picklable across process boundary)
2. Receives a ``multiprocessing.Queue`` for sending ``SubprocessMessage`` back
3. Receives a ``multiprocessing.Event`` for cancellation signaling
4. Puts progress event messages during the transfer
5. Puts a terminal message (result/error/cancelled) when done

.. warning::

    These functions MUST be defined at module top-level (not nested)
    so they are picklable by ``multiprocessing`` with the ``spawn`` context.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import signal
import time
from typing import Any, Dict, List

from .subprocess_messages import SubprocessMessage


# ── Initialization & Safe IO ─────────────────────────────────────

def _init_worker() -> None:
    """Initialize worker process state.
    
    Ignores SIGINT so that Ctrl+C is exclusively handled by the main
    process (which will cleanly terminate this worker).
    """
    try:
        if mp.current_process().name != "MainProcess":
            signal.signal(signal.SIGINT, signal.SIG_IGN)
    except Exception:
        pass


def _safe_put(mp_queue: mp.Queue, message: SubprocessMessage) -> None:
    """Put a message to the multiprocessing queue, suppressing all errors.

    Catches ``BaseException`` (including ``KeyboardInterrupt``) so that
    a second interrupt during error/cancel handling never produces a
    traceback from the subprocess.
    """
    try:
        mp_queue.put(message)
    except BaseException:
        pass


def _handle_worker_exception(mp_queue: mp.Queue, e: BaseException) -> None:
    """Safely format and send an exception as a terminal message."""
    try:
        from .types import TransferCancelledError
        if isinstance(e, KeyboardInterrupt):
            _safe_put(mp_queue, SubprocessMessage.cancelled(
                message="Transfer interrupted by user (Ctrl+C)",
            ))
        elif isinstance(e, TransferCancelledError):
            _safe_put(mp_queue, SubprocessMessage.cancelled(
                message="Transfer cancelled by user",
            ))
        else:
            _safe_put(mp_queue, SubprocessMessage.error(
                message=str(e),
                error_type=type(e).__name__,
            ))
    except BaseException:
        pass


# ── Serialization Helpers ────────────────────────────────────────

def _serialize_xet_file_data(xet_file_data: Any) -> Dict[str, Any]:
    """Convert an XetFileData object to a plain dict for pickling.

    ``XetFileData`` is a namedtuple from ``huggingface_hub`` with fields
    ``file_hash`` and ``refresh_route``. We serialize it to a dict so
    it can cross the process boundary.
    """
    if xet_file_data is None:
        return {}
    return {
        "file_hash": getattr(xet_file_data, "file_hash", ""),
        "refresh_route": getattr(xet_file_data, "refresh_route", ""),
    }


def _deserialize_xet_file_data(data: Dict[str, Any]) -> Any:
    """Reconstruct an XetFileData-like object from a dict.

    Returns a simple namespace object that has ``file_hash`` and
    ``refresh_route`` attributes, compatible with ``refresh_xet_connection_info``.
    """
    if not data:
        return None

    class _XetFileDataProxy:
        __slots__ = ("file_hash", "refresh_route")

        def __init__(self, file_hash: str, refresh_route: str):
            self.file_hash = file_hash
            self.refresh_route = refresh_route

    return _XetFileDataProxy(
        file_hash=data.get("file_hash", ""),
        refresh_route=data.get("refresh_route", ""),
    )


# ── Internal Callback (runs in child process) ────────────────────

def _make_progress_callback(
    filename: str,
    total_bytes: int,
    transfer_id: str,
    mp_queue: mp.Queue,
    cancel_event: mp.Event,
    direction: str = "download",
    file_index: int = 0,
    total_files: int = 1,
) -> Any:
    """Create a progress callback suitable for ``hf_xet`` detailed mode.

    The returned callable has the signature ``callback(total_update, item_updates)``
    which matches what the Rust runtime detects via ``inspect.signature()``.
    """
    from .types import EventType, ProgressPhase, TransferDirection

    _start_time = time.time()

    def progress_updater(total_update, item_updates):
        # Check cancellation
        if cancel_event.is_set():
            from .types import TransferCancelledError
            raise TransferCancelledError("Transfer cancelled by user")

        # Extract values from Rust PyTotalProgressUpdate
        bytes_completed = getattr(total_update, "total_bytes_completed", 0)
        total = getattr(total_update, "total_bytes", 0) or total_bytes
        speed = getattr(total_update, "total_bytes_completion_rate", 0) or 0
        transfer_completed = getattr(total_update, "total_transfer_bytes_completed", 0)
        transfer_total = getattr(total_update, "total_transfer_bytes", 0)
        transfer_speed = getattr(total_update, "total_transfer_bytes_completion_rate", 0) or 0

        # Per-file progress for multi-file transfers
        if total_files > 1 and item_updates:
            item_update = next(
                (item for item in item_updates if getattr(item, "item_name", "") == filename),
                None,
            )
            if item_update is not None:
                bytes_completed = getattr(item_update, "bytes_completed", 0)
                total = getattr(item_update, "total_bytes", 0) or total_bytes

        # Choose display bytes: prefer transfer_completed for smooth progress
        display_completed = bytes_completed
        if direction == "download" and transfer_completed > 0:
            if bytes_completed < total:
                display_completed = transfer_completed
            elif bytes_completed >= total > 0:
                display_completed = bytes_completed

        active_speed = transfer_speed or speed

        # NTH-001: Progress estimation when bytes = 0 but speed > 0
        if display_completed == 0 and active_speed > 0:
            elapsed = time.time() - _start_time
            estimated = int(active_speed * elapsed)
            if total > 0:
                estimated = min(estimated, int(total * 0.99))
            display_completed = estimated

        percentage = ((display_completed / total * 100) if total > 0 else 0)

        event_dict = {
            "event_type": EventType.PROGRESS.value,
            "transfer_id": transfer_id,
            "direction": direction,
            "filename": filename,
            "phase": ProgressPhase.DOWNLOADING.value if direction == "download" else ProgressPhase.UPLOADING.value,
            "bytes_completed": display_completed,
            "total_bytes": total,
            "percentage": percentage,
            "speed": active_speed,
            "file_index": file_index,
            "total_files": total_files,
            "transfer_bytes_completed": transfer_completed if total_files == 1 else 0,
            "transfer_bytes_total": transfer_total if total_files == 1 else 0,
            "transfer_speed": transfer_speed if total_files == 1 else 0,
        }
        try:
            mp_queue.put_nowait(SubprocessMessage.event(event_dict))
        except BaseException:
            pass  # Drop event if queue is full or closed

    return progress_updater


# ── Download Worker ───────────────────────────────────────────────

def _download_worker(params: Dict[str, Any], mp_queue: mp.Queue, cancel_event: mp.Event) -> None:
    """Worker function for single-file Xet downloads.

    Runs in a child process. Imports ``hf_xet`` locally.

    Args:
        params: Dict with keys: file_hash, file_size, dest_path,
            xet_file_data (dict), token, endpoint, transfer_id,
            report_interval, request_headers.
        mp_queue: Queue for sending SubprocessMessage back to main process.
        cancel_event: Event set by main process to signal cancellation.
    """
    _init_worker()
    from .types import EventType, ProgressPhase, TransferCancelledError, TransferDirection

    transfer_id = params["transfer_id"]
    filename = os.path.basename(params["dest_path"])
    file_size = params["file_size"]

    try:
        import hf_xet
        from huggingface_hub.utils._xet import refresh_xet_connection_info
    except ImportError as e:
        _safe_put(mp_queue, SubprocessMessage.error(
            message=str(e), error_type="ImportError", retryable=False,
        ))
        return

    # Reconstruct XetFileData from serialized dict
    xet_file_data = _deserialize_xet_file_data(params.get("xet_file_data", {}))

    # Get credentials
    try:
        headers = params.get("request_headers", {})
        connection_info = refresh_xet_connection_info(file_data=xet_file_data, headers=headers)

        def token_refresher():
            ci = refresh_xet_connection_info(file_data=xet_file_data, headers=headers)
            return ci.access_token, ci.expiration_unix_epoch
    except Exception as e:
        _safe_put(mp_queue, SubprocessMessage.error(
            message=f"Failed to get download credentials: {e}",
            error_type=type(e).__name__,
        ))
        return

    # Build download info
    download_info = [
        hf_xet.PyXetDownloadInfo(
            destination_path=str(os.path.abspath(params["dest_path"])),
            hash=params["file_hash"],
            file_size=file_size,
        )
    ]

    # Build progress callback
    callback = _make_progress_callback(
        filename=filename,
        total_bytes=file_size,
        transfer_id=transfer_id,
        mp_queue=mp_queue,
        cancel_event=cancel_event,
        direction="download",
    )

    try:
        kwargs: Dict[str, Any] = dict(
            endpoint=connection_info.endpoint,
            token_info=(connection_info.access_token, connection_info.expiration_unix_epoch),
            token_refresher=token_refresher,
            progress_updater=[callback],
        )
        if params.get("request_headers"):
            kwargs["request_headers"] = params["request_headers"]

        hf_xet.download_files(download_info, **kwargs)

        _safe_put(mp_queue, SubprocessMessage.result(
            filename=filename,
            destination_path=params["dest_path"],
            file_size=file_size,
            transfer_id=transfer_id,
        ))

    except (KeyboardInterrupt, Exception) as e:
        _handle_worker_exception(mp_queue, e)


def _download_batch_worker(params: Dict[str, Any], mp_queue: mp.Queue, cancel_event: mp.Event) -> None:
    """Worker function for multi-file Xet downloads.

    Args:
        params: Dict with keys: file_specs (list of dicts), token,
            endpoint, transfer_id, report_interval, request_headers.
        mp_queue: Queue for sending SubprocessMessage back to main process.
        cancel_event: Event set by main process to signal cancellation.
    """
    _init_worker()
    from .types import EventType, ProgressPhase, TransferCancelledError, TransferDirection

    transfer_id = params["transfer_id"]
    file_specs: List[Dict[str, Any]] = params["file_specs"]
    total_files = len(file_specs)

    try:
        import hf_xet
        from huggingface_hub.utils._xet import refresh_xet_connection_info
    except ImportError as e:
        _safe_put(mp_queue, SubprocessMessage.error(
            message=str(e), error_type="ImportError", retryable=False,
        ))
        return

    # Use first file's xet_file_data for credentials
    xet_file_data = _deserialize_xet_file_data(file_specs[0].get("xet_file_data", {}))

    try:
        headers = params.get("request_headers", {})
        connection_info = refresh_xet_connection_info(file_data=xet_file_data, headers=headers)

        def token_refresher():
            ci = refresh_xet_connection_info(file_data=xet_file_data, headers=headers)
            return ci.access_token, ci.expiration_unix_epoch
    except Exception as e:
        _safe_put(mp_queue, SubprocessMessage.error(
            message=f"Failed to get download credentials: {e}",
            error_type=type(e).__name__,
        ))
        return

    # Build download infos and callbacks
    download_infos = []
    callbacks = []
    for i, spec in enumerate(file_specs):
        filename = os.path.basename(spec["dest_path"])
        download_infos.append(
            hf_xet.PyXetDownloadInfo(
                destination_path=str(os.path.abspath(spec["dest_path"])),
                hash=spec["hash"],
                file_size=spec["file_size"],
            )
        )
        callbacks.append(
            _make_progress_callback(
                filename=filename,
                total_bytes=spec["file_size"],
                transfer_id=transfer_id,
                mp_queue=mp_queue,
                cancel_event=cancel_event,
                direction="download",
                file_index=i,
                total_files=total_files,
            )
        )

    try:
        kwargs: Dict[str, Any] = dict(
            endpoint=connection_info.endpoint,
            token_info=(connection_info.access_token, connection_info.expiration_unix_epoch),
            token_refresher=token_refresher,
            progress_updater=callbacks,
        )
        if params.get("request_headers"):
            kwargs["request_headers"] = params["request_headers"]

        hf_xet.download_files(download_infos, **kwargs)

        # Send individual result messages for each file
        for i, spec in enumerate(file_specs):
            filename = os.path.basename(spec["dest_path"])
            _safe_put(mp_queue, SubprocessMessage.result(
                filename=filename,
                destination_path=spec["dest_path"],
                file_size=spec["file_size"],
                transfer_id=transfer_id,
                file_index=i,
                total_files=total_files,
            ))

    except (KeyboardInterrupt, Exception) as e:
        _handle_worker_exception(mp_queue, e)


# ── Upload Workers ────────────────────────────────────────────────

def _upload_file_worker(params: Dict[str, Any], mp_queue: mp.Queue, cancel_event: mp.Event) -> None:
    """Worker function for Xet file uploads.

    Args:
        params: Dict with keys: file_path, repo_id, token, repo_type,
            revision, endpoint, transfer_id, report_interval.
        mp_queue: Queue for sending SubprocessMessage back to main process.
        cancel_event: Event set by main process to signal cancellation.
    """
    _init_worker()
    from .types import EventType, ProgressPhase, TransferCancelledError, TransferDirection

    transfer_id = params["transfer_id"]
    file_path = params["file_path"]
    filename = os.path.basename(file_path)

    try:
        import hf_xet
        from huggingface_hub.utils._xet import (
            XetTokenType,
            fetch_xet_connection_info_from_repo_info,
        )
    except ImportError as e:
        _safe_put(mp_queue, SubprocessMessage.error(
            message=str(e), error_type="ImportError", retryable=False,
        ))
        return

    try:
        file_size = os.path.getsize(file_path)
    except OSError as e:
        _safe_put(mp_queue, SubprocessMessage.error(
            message=f"File not found: {file_path}", error_type=type(e).__name__,
        ))
        return

    # Get upload credentials
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=params["token"], endpoint=params.get("endpoint"))
        headers = api._build_hf_headers()

        connection_info = fetch_xet_connection_info_from_repo_info(
            token_type=XetTokenType.WRITE,
            repo_id=params["repo_id"],
            repo_type=params.get("repo_type", "model"),
            revision=params.get("revision"),
            headers=headers,
            endpoint=params.get("endpoint"),
        )

        def token_refresher():
            info = fetch_xet_connection_info_from_repo_info(
                token_type=XetTokenType.WRITE,
                repo_id=params["repo_id"],
                repo_type=params.get("repo_type", "model"),
                revision=params.get("revision"),
                headers=headers,
                endpoint=params.get("endpoint"),
            )
            return info.access_token, info.expiration_unix_epoch
    except Exception as e:
        _safe_put(mp_queue, SubprocessMessage.error(
            message=f"Failed to get upload credentials: {e}",
            error_type=type(e).__name__,
        ))
        return

    # Build progress callback
    callback = _make_progress_callback(
        filename=filename,
        total_bytes=file_size,
        transfer_id=transfer_id,
        mp_queue=mp_queue,
        cancel_event=cancel_event,
        direction="upload",
    )

    try:
        results = hf_xet.upload_files(
            [file_path],
            connection_info.endpoint,
            (connection_info.access_token, connection_info.expiration_unix_epoch),
            token_refresher,
            callback,
            params.get("repo_type", "model"),
        )

        result_info = results[0] if results else None
        _safe_put(mp_queue, SubprocessMessage.result(
            filename=filename,
            file_size=file_size,
            transfer_id=transfer_id,
            hash=getattr(result_info, "hash", "") if result_info else "",
            url=getattr(result_info, "url", None) if result_info else None,
        ))

    except (KeyboardInterrupt, Exception) as e:
        _handle_worker_exception(mp_queue, e)


def _upload_bytes_worker(params: Dict[str, Any], mp_queue: mp.Queue, cancel_event: mp.Event) -> None:
    """Worker function for Xet bytes uploads.

    For large payloads (>10MB), the caller should write bytes to a temp file
    and pass ``file_path`` instead of ``file_content`` to avoid excessive
    pickle serialization cost.

    Args:
        params: Dict with keys: file_content (bytes) OR file_path (str),
            filename, repo_id, token, repo_type, revision, endpoint,
            transfer_id, report_interval.
        mp_queue: Queue for sending SubprocessMessage back to main process.
        cancel_event: Event set by main process to signal cancellation.
    """
    _init_worker()
    from .types import EventType, ProgressPhase, TransferCancelledError, TransferDirection

    transfer_id = params["transfer_id"]
    filename = params["filename"]

    # Determine content source
    file_content = params.get("file_content")
    file_path = params.get("file_path")

    if file_content is not None:
        file_size = len(file_content)
    elif file_path is not None:
        file_size = os.path.getsize(file_path)
    else:
        _safe_put(mp_queue, SubprocessMessage.error(
            message="Either file_content or file_path must be provided",
            error_type="ValueError",
        ))
        return

    try:
        import hf_xet
        from huggingface_hub.utils._xet import (
            XetTokenType,
            fetch_xet_connection_info_from_repo_info,
        )
    except ImportError as e:
        _safe_put(mp_queue, SubprocessMessage.error(
            message=str(e), error_type="ImportError", retryable=False,
        ))
        return

    # Get upload credentials
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=params["token"], endpoint=params.get("endpoint"))
        headers = api._build_hf_headers()

        connection_info = fetch_xet_connection_info_from_repo_info(
            token_type=XetTokenType.WRITE,
            repo_id=params["repo_id"],
            repo_type=params.get("repo_type", "model"),
            revision=params.get("revision"),
            headers=headers,
            endpoint=params.get("endpoint"),
        )

        def token_refresher():
            info = fetch_xet_connection_info_from_repo_info(
                token_type=XetTokenType.WRITE,
                repo_id=params["repo_id"],
                repo_type=params.get("repo_type", "model"),
                revision=params.get("revision"),
                headers=headers,
                endpoint=params.get("endpoint"),
            )
            return info.access_token, info.expiration_unix_epoch
    except Exception as e:
        _safe_put(mp_queue, SubprocessMessage.error(
            message=f"Failed to get upload credentials: {e}",
            error_type=type(e).__name__,
        ))
        return

    # Build progress callback
    callback = _make_progress_callback(
        filename=filename,
        total_bytes=file_size,
        transfer_id=transfer_id,
        mp_queue=mp_queue,
        cancel_event=cancel_event,
        direction="upload",
    )

    try:
        if file_content is not None:
            results = hf_xet.upload_bytes(
                [file_content],
                connection_info.endpoint,
                (connection_info.access_token, connection_info.expiration_unix_epoch),
                token_refresher,
                callback,
                params.get("repo_type", "model"),
            )
        else:
            results = hf_xet.upload_files(
                [file_path],
                connection_info.endpoint,
                (connection_info.access_token, connection_info.expiration_unix_epoch),
                token_refresher,
                callback,
                params.get("repo_type", "model"),
            )

        result_info = results[0] if results else None
        _safe_put(mp_queue, SubprocessMessage.result(
            filename=filename,
            file_size=file_size,
            transfer_id=transfer_id,
            hash=getattr(result_info, "hash", "") if result_info else "",
            url=getattr(result_info, "url", None) if result_info else None,
        ))

    except (KeyboardInterrupt, Exception) as e:
        _handle_worker_exception(mp_queue, e)