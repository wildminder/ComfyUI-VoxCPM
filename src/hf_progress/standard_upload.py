"""Standard (non-Xet) upload progress tracking via tqdm monkey-patching."""

from __future__ import annotations

import logging
import os
import queue
import tempfile
from typing import Callable, Optional

from .callbacks import tqdm_upload_patcher, state_manager
from .types import (
    EventType,
    ProgressEvent,
    ProgressPhase,
    TransferCancelledError,
    TransferDirection,
    TransferError,
    generate_transfer_id,
)

logger = logging.getLogger(__name__)


# Narrow set of expected exceptions from the HuggingFace API.
# Any unexpected exception still propagates after the ERROR event.
_ExpectedUploadErrors = (OSError, ValueError, ConnectionError, RuntimeError)


def upload_file(
    file_path: str,
    repo_id: str,
    token: str,
    event_queue: queue.Queue,
    path_in_repo: Optional[str] = None,
    repo_type: str = "model",
    revision: Optional[str] = None,
    endpoint: Optional[str] = None,
    transfer_id: Optional[str] = None,
    report_interval: float = 0.1,
    is_cancelled: Optional[Callable[[], bool]] = None,
) -> str:
    from huggingface_hub import HfApi

    transfer_id = transfer_id or generate_transfer_id()
    filename = os.path.basename(file_path)
    path_in_repo = path_in_repo or filename
    total_bytes = os.path.getsize(file_path)

    try:
        with tqdm_upload_patcher(
            event_queue=event_queue,
            transfer_id=transfer_id,
            filename=filename,
            report_interval=report_interval,
            is_cancelled=is_cancelled,
            total_bytes=total_bytes,
        ):
            api = HfApi(token=token, endpoint=endpoint)
            result = api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type=repo_type,
                revision=revision,
            )

            # Ensure COMPLETE event is fired if the chunked file parts missed the strict bound
            state = state_manager.get_state(transfer_id)
            if not state.get("completed_emitted", False):
                state_manager.mark_upload_completed(transfer_id)
                event_queue.put(
                    ProgressEvent(
                        event_type=EventType.COMPLETE,
                        transfer_id=transfer_id,
                        direction=TransferDirection.UPLOAD,
                        filename=filename,
                        phase=ProgressPhase.COMPLETE,
                        bytes_completed=total_bytes,
                        total_bytes=total_bytes,
                        percentage=100.0,
                    )
                )

            return result
    
    except KeyboardInterrupt:
        event_queue.put(
            ProgressEvent.cancelled_event(
                transfer_id=transfer_id,
                direction=TransferDirection.UPLOAD,
                filename=filename,
            )
        )
        raise TransferCancelledError("Upload interrupted by user (Ctrl+C)")
    except TransferCancelledError:
        event_queue.put(
            ProgressEvent.cancelled_event(
                transfer_id=transfer_id,
                direction=TransferDirection.UPLOAD,
                filename=filename,
            )
        )
        raise
    except _ExpectedUploadErrors as e:
        # Broad except purposely narrow to expected HF API errors.
        # Emit ERROR event so consumers see the failure before re-raising.
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
        state_manager.clear_state(transfer_id)


def upload_bytes(
    file_content: bytes,
    filename: str,
    repo_id: str,
    token: str,
    event_queue: queue.Queue,
    path_in_repo: Optional[str] = None,
    repo_type: str = "model",
    revision: Optional[str] = None,
    endpoint: Optional[str] = None,
    transfer_id: Optional[str] = None,
    report_interval: float = 0.1,
    is_cancelled: Optional[Callable[[], bool]] = None,
) -> str:
    transfer_id = transfer_id or generate_transfer_id()
    path_in_repo = path_in_repo or filename

    with tempfile.NamedTemporaryFile(
        suffix=f"_{filename}", delete=False
    ) as f:
        f.write(file_content)
        temp_path = f.name

    try:
        return upload_file(
            file_path=temp_path,
            repo_id=repo_id,
            token=token,
            event_queue=event_queue,
            path_in_repo=path_in_repo,
            repo_type=repo_type,
            revision=revision,
            endpoint=endpoint,
            transfer_id=transfer_id,
            report_interval=report_interval,
            is_cancelled=is_cancelled,
        )
    # Intentionally narrow: tempfile cleanup must run.
    finally:
        os.unlink(temp_path)


def upload_folder(
    folder_path: str,
    repo_id: str,
    token: str,
    event_queue: queue.Queue,
    path_in_repo: Optional[str] = None,
    repo_type: str = "model",
    revision: Optional[str] = None,
    allow_patterns: Optional[list[str] | str] = None,
    ignore_patterns: Optional[list[str] | str] = None,
    delete_patterns: Optional[list[str] | str] = None,
    endpoint: Optional[str] = None,
    transfer_id: Optional[str] = None,
    report_interval: float = 0.1,
    is_cancelled: Optional[Callable[[], bool]] = None,
) -> str:
    from huggingface_hub import HfApi

    transfer_id = transfer_id or generate_transfer_id()
    
    # Pre-calculate approximate folder size for progress display
    total_bytes = 0
    for root, _, files in os.walk(folder_path):
        for name in files:
            try:
                total_bytes += os.path.getsize(os.path.join(root, name))
            except OSError:
                logger.warning("Could not get size of %s — skipping", os.path.join(root, name))

    try:
        with tqdm_upload_patcher(
            event_queue=event_queue,
            transfer_id=transfer_id,
            filename=f"folder:{os.path.basename(folder_path)}",
            report_interval=report_interval,
            is_cancelled=is_cancelled,
            total_bytes=total_bytes,
        ):
            api = HfApi(token=token, endpoint=endpoint)
            result = api.upload_folder(
                folder_path=folder_path,
                repo_id=repo_id,
                path_in_repo=path_in_repo,
                repo_type=repo_type,
                revision=revision,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
                delete_patterns=delete_patterns,
            )

            state = state_manager.get_state(transfer_id)
            if not state.get("completed_emitted", False):
                state_manager.mark_upload_completed(transfer_id)
                event_queue.put(
                    ProgressEvent(
                        event_type=EventType.COMPLETE,
                        transfer_id=transfer_id,
                        direction=TransferDirection.UPLOAD,
                        filename=f"folder:{os.path.basename(folder_path)}",
                        phase=ProgressPhase.COMPLETE,
                        bytes_completed=total_bytes,
                        total_bytes=total_bytes,
                        percentage=100.0,
                    )
                )

            return getattr(result, "commit_url", str(result))
    
    except KeyboardInterrupt:
        event_queue.put(
            ProgressEvent.cancelled_event(
                transfer_id=transfer_id,
                direction=TransferDirection.UPLOAD,
                filename=f"folder:{os.path.basename(folder_path)}",
            )
        )
        raise TransferCancelledError("Upload interrupted by user (Ctrl+C)")
    except TransferCancelledError:
        event_queue.put(
            ProgressEvent.cancelled_event(
                transfer_id=transfer_id,
                direction=TransferDirection.UPLOAD,
                filename=f"folder:{os.path.basename(folder_path)}",
            )
        )
        raise
    except _ExpectedUploadErrors as e:
        event_queue.put(
            ProgressEvent(
                event_type=EventType.ERROR,
                transfer_id=transfer_id,
                direction=TransferDirection.UPLOAD,
                filename=f"folder:{os.path.basename(folder_path)}",
                phase=ProgressPhase.ERROR,
                error=TransferError(message=str(e), error_type=type(e).__name__),
            )
        )
        raise
    finally:
        state_manager.clear_state(transfer_id)