"""Standard (non-Xet) download progress tracking via tqdm_class."""

from __future__ import annotations

import contextlib
import queue
from typing import Callable, Optional

from .callbacks import DownloadProgressTqdm, state_manager
from .types import (
    EventType,
    ProgressEvent,
    ProgressPhase,
    TransferDirection,
    generate_transfer_id,
)


@contextlib.contextmanager
def patch_download_chunk_size(chunk_size: int = 256 * 1024):
    from huggingface_hub import constants as hf_constants
    import huggingface_hub.file_download as file_download

    original_constants = hf_constants.DOWNLOAD_CHUNK_SIZE
    original_fd = getattr(file_download.constants, 'DOWNLOAD_CHUNK_SIZE', None)

    hf_constants.DOWNLOAD_CHUNK_SIZE = chunk_size
    if original_fd is not None:
        file_download.constants.DOWNLOAD_CHUNK_SIZE = chunk_size
        
    try:
        yield
    finally:
        hf_constants.DOWNLOAD_CHUNK_SIZE = original_constants
        if original_fd is not None:
            file_download.constants.DOWNLOAD_CHUNK_SIZE = original_fd


@contextlib.contextmanager
def patch_xet_get():
    try:
        import huggingface_hub.file_download as fd
        original_xet_get = fd.xet_get
    except ImportError:
        yield
        return

    def patched_xet_get(*args, **kwargs):
        # xet_get is a keyword-only function natively. Accessing args[0] throws IndexError.
        incomplete_path = kwargs.get("incomplete_path")
        xet_file_data = kwargs.get("xet_file_data")
        headers = kwargs.get("headers", {})
        expected_size = kwargs.get("expected_size")
        displayed_filename = kwargs.get("displayed_filename")
        tqdm_class = kwargs.get("tqdm_class")
        _tqdm_bar = kwargs.get("_tqdm_bar")

        if incomplete_path is None or xet_file_data is None:
            # Gracefully fallback if upstream huggingface_hub changes its kwargs heavily
            return original_xet_get(*args, **kwargs)

        try:
            from hf_xet import PyXetDownloadInfo, download_files
            from huggingface_hub.utils import refresh_xet_connection_info
        except ImportError:
            return original_xet_get(*args, **kwargs)

        connection_info = refresh_xet_connection_info(file_data=xet_file_data, headers=headers)
        def token_refresher():
            ci = refresh_xet_connection_info(file_data=xet_file_data, headers=headers)
            return ci.access_token, ci.expiration_unix_epoch

        xet_download_info = [
            PyXetDownloadInfo(
                destination_path=str(incomplete_path.absolute()),
                hash=xet_file_data.file_hash,
                file_size=expected_size
            )
        ]

        if not displayed_filename:
            displayed_filename = incomplete_path.name
        if len(displayed_filename) > 40:
            displayed_filename = f"{displayed_filename[:40]}(…)"

        progress_cm = fd._get_progress_bar_context(
            desc=displayed_filename,
            log_level=fd.logger.getEffectiveLevel(),
            total=expected_size,
            initial=0,
            name="huggingface_hub.xet_get",
            tqdm_class=tqdm_class,
            _tqdm_bar=_tqdm_bar,
        )

        xet_headers = headers.copy()
        xet_headers.pop("authorization", None)

        with progress_cm as progress:
            state = {"last_bytes": 0}
            
            def progress_updater(total_update, item_updates):
                # Enforce termination if cancelled from within the progress context
                is_cancelled = getattr(progress, "_is_cancelled", None)
                if is_cancelled is not None and is_cancelled():
                    raise RuntimeError("Transfer cancelled by user")

                transfer_completed = getattr(total_update, "total_transfer_bytes_completed", 0)
                bytes_completed = getattr(total_update, "total_bytes_completed", 0)
                total_bytes = getattr(total_update, "total_bytes", 0) or expected_size
                
                display_bytes = transfer_completed if transfer_completed > 0 else bytes_completed
                if bytes_completed >= total_bytes and total_bytes > 0:
                    display_bytes = bytes_completed
                    
                delta = display_bytes - state["last_bytes"]
                if delta > 0:
                    progress.update(delta)
                    state["last_bytes"] = display_bytes

            download_files(
                xet_download_info,
                endpoint=connection_info.endpoint,
                token_info=(connection_info.access_token, connection_info.expiration_unix_epoch),
                token_refresher=token_refresher,
                progress_updater=[progress_updater],
                request_headers=xet_headers,
            )
            
    fd.xet_get = patched_xet_get
    try:
        yield
    finally:
        fd.xet_get = original_xet_get


def download_file(
    repo_id: str,
    filename: str,
    token: Optional[str],
    event_queue: queue.Queue,
    repo_type: str = "model",
    revision: Optional[str] = None,
    endpoint: Optional[str] = None,
    local_dir: Optional[str] = None,
    transfer_id: Optional[str] = None,
    report_interval: float = 0.1,
    is_cancelled: Optional[Callable[[], bool]] = None,
    **kwargs,
) -> str:
    from huggingface_hub import hf_hub_download

    transfer_id = transfer_id or generate_transfer_id()

    tqdm_class = DownloadProgressTqdm.bind(
        event_queue=event_queue,
        transfer_id=transfer_id,
        filename=filename,
        report_interval=report_interval,
        is_cancelled=is_cancelled,
    )

    event_queue.put(
        ProgressEvent(
            event_type=EventType.START,
            transfer_id=transfer_id,
            direction=TransferDirection.DOWNLOAD,
            filename=filename,
            phase=ProgressPhase.DOWNLOADING,
            total_bytes=0,
        )
    )

    try:
        download_kwargs = dict(
            repo_id=repo_id,
            filename=filename,
            repo_type=repo_type,
            revision=revision,
            token=token,
            endpoint=endpoint,
            tqdm_class=tqdm_class,
        )
        if local_dir is not None:
            download_kwargs["local_dir"] = local_dir
        download_kwargs.update(kwargs)

        with patch_download_chunk_size(), patch_xet_get():
            result = hf_hub_download(**download_kwargs)
        return result

    except Exception as e:
        event_queue.put(
            ProgressEvent(
                event_type=EventType.ERROR,
                transfer_id=transfer_id,
                direction=TransferDirection.DOWNLOAD,
                filename=filename,
                phase=ProgressPhase.ERROR,
                error=str(e),
            )
        )
        raise
    finally:
        state_manager.clear_state(transfer_id)


def download_snapshot(
    repo_id: str,
    token: Optional[str],
    event_queue: queue.Queue,
    allow_patterns=None,
    ignore_patterns=None,
    repo_type: str = "model",
    revision: Optional[str] = None,
    endpoint: Optional[str] = None,
    local_dir: Optional[str] = None,
    transfer_id: Optional[str] = None,
    report_interval: float = 0.1,
    is_cancelled: Optional[Callable[[], bool]] = None,
    **kwargs,
) -> str:
    from huggingface_hub import snapshot_download

    transfer_id = transfer_id or generate_transfer_id()

    tqdm_class = DownloadProgressTqdm.bind(
        event_queue=event_queue,
        transfer_id=transfer_id,
        filename=f"{repo_id}",
        report_interval=report_interval,
        is_cancelled=is_cancelled,
    )

    try:
        download_kwargs = dict(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            token=token,
            endpoint=endpoint,
            tqdm_class=tqdm_class,
        )
        if local_dir is not None:
            download_kwargs["local_dir"] = local_dir
        download_kwargs.update(kwargs)

        with patch_download_chunk_size(), patch_xet_get():
            result = snapshot_download(**download_kwargs)
            
        # Manually synthesize the COMPLETE event for snapshot downloads because 
        # huggingface_hub's _AggregatedTqdm never calls .close() on the byte tracking bar.
        stats = state_manager.get_state(transfer_id)
        event_queue.put(
            ProgressEvent(
                event_type=EventType.COMPLETE,
                transfer_id=transfer_id,
                direction=TransferDirection.DOWNLOAD,
                filename=f"{repo_id}",
                phase=ProgressPhase.COMPLETE,
                bytes_completed=stats.get("bytes_completed", 0),
                total_bytes=stats.get("total_bytes", 0),
                percentage=100.0,
                file_index=stats.get("files_completed", 0),
                total_files=stats.get("total_files", 0),
            )
        )
        return result

    except Exception as e:
        event_queue.put(
            ProgressEvent(
                event_type=EventType.ERROR,
                transfer_id=transfer_id,
                direction=TransferDirection.DOWNLOAD,
                filename=f"{repo_id}",
                phase=ProgressPhase.ERROR,
                error=str(e),
            )
        )
        raise
    finally:
        state_manager.clear_state(transfer_id)