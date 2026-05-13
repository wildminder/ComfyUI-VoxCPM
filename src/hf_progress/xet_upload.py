"""Xet direct upload functions with progress tracking via subprocess isolation.

All ``hf_xet`` calls are executed in isolated child processes using
``XetSubprocessRunner``, so they can be safely terminated without
affecting the main process.
"""

from __future__ import annotations

import os
import queue
import tempfile
from dataclasses import dataclass
from typing import Callable, Optional

from ._xet_worker import _upload_bytes_worker, _upload_file_worker
from .subprocess_runner import XetSubprocessRunner
from .token import is_xet_available
from .types import (
    EventType,
    ProgressEvent,
    ProgressPhase,
    TransferCancelledError,
    TransferDirection,
    TransferError,
    TransferProgressError,
    generate_transfer_id,
)

# Threshold for auto-writing bytes to temp file instead of pickling across processes
_LARGE_PAYLOAD_THRESHOLD = 10 * 1024 * 1024  # 10 MB


@dataclass
class XetUploadResult:
    success: bool
    filename: str
    hash: str = ""
    file_size: int = 0
    transfer_id: str = ""
    url: Optional[str] = None


def _run_upload_in_subprocess(
    *,
    filename: str,
    file_size: int,
    repo_id: str,
    token: str,
    event_queue: queue.Queue,
    repo_type: str = "model",
    revision: Optional[str] = None,
    endpoint: Optional[str] = None,
    transfer_id: Optional[str] = None,
    report_interval: float = 0.1,
    is_cancelled: Optional[Callable[[], bool]] = None,
    worker_func: Callable,
    params: dict,
) -> XetUploadResult:
    """Shared subprocess upload logic for both file and bytes uploads.

    Args:
        filename: Name of the file being uploaded.
        file_size: Size of the file in bytes.
        repo_id: Target repository ID.
        token: HuggingFace API token.
        event_queue: Queue for progress events.
        repo_type: Repository type.
        revision: Optional git revision.
        endpoint: Optional custom endpoint.
        transfer_id: Optional pre-existing transfer ID.
        report_interval: Event reporting interval.
        is_cancelled: Optional cancellation hook.
        worker_func: The worker function to run in subprocess.
        params: Parameters dict for the worker.

    Returns:
        XetUploadResult on success.

    Raises:
        TransferCancelledError: If the transfer is cancelled.
        TransferError: If the upload fails.
    """
    transfer_id = transfer_id or generate_transfer_id()

    # Emit START event from main process
    event_queue.put(
        ProgressEvent(
            event_type=EventType.START,
            transfer_id=transfer_id,
            direction=TransferDirection.UPLOAD,
            filename=filename,
            phase=ProgressPhase.UPLOADING,
            total_bytes=file_size,
        )
    )

    runner = XetSubprocessRunner()
    runner.start(
        worker_func=worker_func,
        params=params,
        event_queue=event_queue,
    )

    try:
        while True:
            result = runner.wait(timeout=1.0)
            if result is not None:
                break
            if is_cancelled is not None and is_cancelled():
                runner.terminate()
                event_queue.put(
                    ProgressEvent.cancelled_event(
                        transfer_id=transfer_id,
                        direction=TransferDirection.UPLOAD,
                        filename=filename,
                    )
                )
                raise TransferCancelledError("Upload cancelled by user")

        if result.get("status") == "success":
            return XetUploadResult(
                success=True,
                filename=result.get("filename", filename),
                hash=result.get("hash", ""),
                file_size=result.get("file_size", file_size),
                transfer_id=transfer_id,
                url=result.get("url"),
            )
        elif (
            result.get("status") == "cancelled"
            or result.get("error_type") == "TransferCancelledError"
            or "cancelled" in result.get("message", "").lower()
            or "interrupted" in result.get("message", "").lower()
        ):
            raise TransferCancelledError(result.get("message", "Upload cancelled by user"))
        else:
            error_msg = result.get("message", "Upload failed")
        raise TransferProgressError(error_msg)

    except KeyboardInterrupt:
        runner.terminate()
        event_queue.put(
            ProgressEvent.cancelled_event(
                transfer_id=transfer_id,
                direction=TransferDirection.UPLOAD,
                filename=filename,
            )
        )
        raise TransferCancelledError("Upload interrupted by user (Ctrl+C)")
    except TransferCancelledError:
        raise
    except TransferProgressError:
        raise
    except Exception as e:
        runner.terminate()
        event_queue.put(
            ProgressEvent(
                event_type=EventType.ERROR,
                transfer_id=transfer_id,
                direction=TransferDirection.UPLOAD,
                filename=filename,
                phase=ProgressPhase.ERROR,
                error=TransferError(message=str(e), error_type=type(e).__name__),
            )
        )
        raise
    finally:
        runner.terminate()


def upload_file_with_xet(
    file_path: str,
    repo_id: str,
    token: str,
    event_queue: queue.Queue,
    repo_type: str = "model",
    revision: Optional[str] = None,
    endpoint: Optional[str] = None,
    transfer_id: Optional[str] = None,
    report_interval: float = 0.1,
    is_cancelled: Optional[Callable[[], bool]] = None,
) -> XetUploadResult:
    """Upload a file via Xet in an isolated subprocess.

    Args:
        file_path: Local path to the file to upload.
        repo_id: Target repository ID.
        token: HuggingFace API token.
        event_queue: Queue for progress events.
        repo_type: Repository type.
        revision: Optional git revision.
        endpoint: Optional custom endpoint.
        transfer_id: Optional pre-existing transfer ID.
        report_interval: Event reporting interval.
        is_cancelled: Optional cancellation hook.

    Returns:
        XetUploadResult on success.

    Raises:
        ImportError: If hf_xet is not installed.
    """
    if not is_xet_available():
        raise ImportError("hf_xet is not installed.")

    filename = os.path.basename(file_path)
    file_size = os.path.getsize(file_path)

    params = {
        "file_path": file_path,
        "repo_id": repo_id,
        "token": token,
        "repo_type": repo_type,
        "revision": revision,
        "endpoint": endpoint,
        "transfer_id": transfer_id,
        "report_interval": report_interval,
        "filename": filename,
        "direction": "upload",
    }

    return _run_upload_in_subprocess(
        filename=filename,
        file_size=file_size,
        repo_id=repo_id,
        token=token,
        event_queue=event_queue,
        repo_type=repo_type,
        revision=revision,
        endpoint=endpoint,
        transfer_id=transfer_id,
        report_interval=report_interval,
        is_cancelled=is_cancelled,
        worker_func=_upload_file_worker,
        params=params,
    )


def upload_bytes_with_xet(
    file_content: bytes,
    filename: str,
    repo_id: str,
    token: str,
    event_queue: queue.Queue,
    repo_type: str = "model",
    revision: Optional[str] = None,
    endpoint: Optional[str] = None,
    transfer_id: Optional[str] = None,
    report_interval: float = 0.1,
    is_cancelled: Optional[Callable[[], bool]] = None,
) -> XetUploadResult:
    """Upload bytes via Xet in an isolated subprocess.

    For payloads larger than 10MB, the bytes are automatically written
    to a temporary file to avoid excessive pickle serialization cost
    across the process boundary.

    Args:
        file_content: Raw bytes to upload.
        filename: Name for the uploaded file.
        repo_id: Target repository ID.
        token: HuggingFace API token.
        event_queue: Queue for progress events.
        repo_type: Repository type.
        revision: Optional git revision.
        endpoint: Optional custom endpoint.
        transfer_id: Optional pre-existing transfer ID.
        report_interval: Event reporting interval.
        is_cancelled: Optional cancellation hook.

    Returns:
        XetUploadResult on success.

    Raises:
        ImportError: If hf_xet is not installed.
    """
    if not is_xet_available():
        raise ImportError("hf_xet is not installed.")

    file_size = len(file_content)
    temp_path = None

    try:
        if file_size > _LARGE_PAYLOAD_THRESHOLD:
            # Write to temp file to avoid expensive pickle of large bytes
            with tempfile.NamedTemporaryFile(
                suffix=f"_{filename}", delete=False
            ) as f:
                f.write(file_content)
                temp_path = f.name

            params = {
                "file_path": temp_path,
                "filename": filename,
                "repo_id": repo_id,
                "token": token,
                "repo_type": repo_type,
                "revision": revision,
                "endpoint": endpoint,
                "transfer_id": transfer_id,
                "report_interval": report_interval,
                "direction": "upload",
            }
        else:
            params = {
                "file_content": file_content,
                "filename": filename,
                "repo_id": repo_id,
                "token": token,
                "repo_type": repo_type,
                "revision": revision,
                "endpoint": endpoint,
                "transfer_id": transfer_id,
                "report_interval": report_interval,
                "direction": "upload",
            }

        return _run_upload_in_subprocess(
            filename=filename,
            file_size=file_size,
            repo_id=repo_id,
            token=token,
            event_queue=event_queue,
            repo_type=repo_type,
            revision=revision,
            endpoint=endpoint,
            transfer_id=transfer_id,
            report_interval=report_interval,
            is_cancelled=is_cancelled,
            worker_func=_upload_bytes_worker,
            params=params,
        )
    finally:
        if temp_path is not None:
            try:
                os.unlink(temp_path)
            except OSError:
                pass


def upload_bytes_via_temp_file(
    file_content: bytes,
    filename: str,
    repo_id: str,
    token: str,
    event_queue: queue.Queue,
    repo_type: str = "model",
    revision: Optional[str] = None,
    endpoint: Optional[str] = None,
    transfer_id: Optional[str] = None,
    report_interval: float = 0.1,
    is_cancelled: Optional[Callable[[], bool]] = None,
) -> XetUploadResult:
    """Upload bytes via temp file, choosing Xet or LFS based on availability.

    This is a convenience wrapper that handles the Xet/LFS routing
    for bytes uploads.
    """
    transfer_id = transfer_id or generate_transfer_id()

    with tempfile.NamedTemporaryFile(
        suffix=f"_{filename}", delete=False
    ) as f:
        f.write(file_content)
        temp_path = f.name

    try:
        if is_xet_available():
            return upload_file_with_xet(
                file_path=temp_path,
                repo_id=repo_id,
                token=token,
                event_queue=event_queue,
                repo_type=repo_type,
                revision=revision,
                endpoint=endpoint,
                transfer_id=transfer_id,
                report_interval=report_interval,
                is_cancelled=is_cancelled,
            )
        else:
            from .standard_upload import upload_file as _upload_file
            return _upload_file(
                file_path=temp_path,
                repo_id=repo_id,
                token=token,
                event_queue=event_queue,
                path_in_repo=filename,
                repo_type=repo_type,
                revision=revision,
                endpoint=endpoint,
                transfer_id=transfer_id,
                report_interval=report_interval,
                is_cancelled=is_cancelled,
            )
    finally:
        os.unlink(temp_path)
