"""Xet direct upload functions with progress tracking."""

from __future__ import annotations

import os
import queue
import tempfile
from dataclasses import dataclass
from typing import Callable, List, Optional

from .callbacks import XetUploadProgressCallback
from .token import XetCredentials, XetTokenManager, is_xet_available
from .types import (
    EventType,
    ProgressEvent,
    ProgressPhase,
    TransferDirection,
    TransferResult,
    generate_transfer_id,
)


@dataclass
class XetUploadResult:
    success: bool
    filename: str
    hash: str = ""
    file_size: int = 0
    transfer_id: str = ""


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
    if not is_xet_available():
        raise ImportError("hf_xet is not installed.")

    import hf_xet

    transfer_id = transfer_id or generate_transfer_id()
    filename = os.path.basename(file_path)
    file_size = os.path.getsize(file_path)

    token_manager = XetTokenManager(token, endpoint)
    creds = token_manager.get_upload_credentials(repo_id, repo_type, revision)

    callback = XetUploadProgressCallback(
        filename=filename,
        total_bytes=file_size,
        event_queue=event_queue,
        report_interval=report_interval,
        transfer_id=transfer_id,
        is_cancelled=is_cancelled,
    )

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

    try:
        results = hf_xet.upload_files(
            [file_path],
            creds.endpoint,
            creds.token_info,
            creds.token_refresher,
            callback.get_wrapper(),
            repo_type,
        )

        result_info = results[0] if results else None
        event_queue.put(
            ProgressEvent(
                event_type=EventType.COMPLETE,
                transfer_id=transfer_id,
                direction=TransferDirection.UPLOAD,
                filename=filename,
                phase=ProgressPhase.COMPLETE,
                bytes_completed=file_size,
                total_bytes=file_size,
                percentage=100.0,
            )
        )

        return XetUploadResult(
            success=True,
            filename=filename,
            hash=getattr(result_info, "hash", "") if result_info else "",
            file_size=getattr(result_info, "file_size", file_size) if result_info else file_size,
            transfer_id=transfer_id,
        )

    except Exception as e:
        event_queue.put(
            ProgressEvent(
                event_type=EventType.ERROR,
                transfer_id=transfer_id,
                direction=TransferDirection.UPLOAD,
                filename=filename,
                phase=ProgressPhase.ERROR,
                error=str(e),
            )
        )
        raise


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
    if not is_xet_available():
        raise ImportError("hf_xet is not installed.")

    import hf_xet

    transfer_id = transfer_id or generate_transfer_id()
    file_size = len(file_content)

    token_manager = XetTokenManager(token, endpoint)
    creds = token_manager.get_upload_credentials(repo_id, repo_type, revision)

    callback = XetUploadProgressCallback(
        filename=filename,
        total_bytes=file_size,
        event_queue=event_queue,
        report_interval=report_interval,
        transfer_id=transfer_id,
        is_cancelled=is_cancelled,
    )

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

    try:
        results = hf_xet.upload_bytes(
            [file_content],
            creds.endpoint,
            creds.token_info,
            creds.token_refresher,
            callback.get_wrapper(),
            repo_type,
        )

        result_info = results[0] if results else None
        event_queue.put(
            ProgressEvent(
                event_type=EventType.COMPLETE,
                transfer_id=transfer_id,
                direction=TransferDirection.UPLOAD,
                filename=filename,
                phase=ProgressPhase.COMPLETE,
                bytes_completed=file_size,
                total_bytes=file_size,
                percentage=100.0,
            )
        )

        return XetUploadResult(
            success=True,
            filename=filename,
            hash=getattr(result_info, "hash", "") if result_info else "",
            file_size=getattr(result_info, "file_size", file_size) if result_info else file_size,
            transfer_id=transfer_id,
        )

    except Exception as e:
        event_queue.put(
            ProgressEvent(
                event_type=EventType.ERROR,
                transfer_id=transfer_id,
                direction=TransferDirection.UPLOAD,
                filename=filename,
                phase=ProgressPhase.ERROR,
                error=str(e),
            )
        )
        raise


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