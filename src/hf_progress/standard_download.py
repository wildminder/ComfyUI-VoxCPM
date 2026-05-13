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
    TransferCancelledError,
    TransferDirection,
    TransferError,
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

        with patch_download_chunk_size():
            result = hf_hub_download(**download_kwargs) # nosec B615
            return result

    except KeyboardInterrupt:
        event_queue.put(
            ProgressEvent.cancelled_event(
                transfer_id=transfer_id,
                direction=TransferDirection.DOWNLOAD,
                filename=filename,
            )
        )
        raise TransferCancelledError("Download interrupted by user (Ctrl+C)")
    except TransferCancelledError:
        event_queue.put(
            ProgressEvent.cancelled_event(
                transfer_id=transfer_id,
                direction=TransferDirection.DOWNLOAD,
                filename=filename,
            )
        )
        raise
    except Exception as e:
        event_queue.put(
            ProgressEvent(
                event_type=EventType.ERROR,
                transfer_id=transfer_id,
                direction=TransferDirection.DOWNLOAD,
                filename=filename,
                phase=ProgressPhase.ERROR,
                error=TransferError(message=str(e), error_type=type(e).__name__),
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

        with patch_download_chunk_size():
            result = snapshot_download(**download_kwargs)  # nosec B615
            
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

    except KeyboardInterrupt:
        stats = state_manager.get_state(transfer_id)
        event_queue.put(
            ProgressEvent.cancelled_event(
                transfer_id=transfer_id,
                direction=TransferDirection.DOWNLOAD,
                filename=f"{repo_id}",
                bytes_completed=stats.get("bytes_completed", 0),
                total_bytes=stats.get("total_bytes", 0),
            )
        )
        raise TransferCancelledError("Download interrupted by user (Ctrl+C)")
    except TransferCancelledError:
        stats = state_manager.get_state(transfer_id)
        event_queue.put(
            ProgressEvent.cancelled_event(
                transfer_id=transfer_id,
                direction=TransferDirection.DOWNLOAD,
                filename=f"{repo_id}",
                bytes_completed=stats.get("bytes_completed", 0),
                total_bytes=stats.get("total_bytes", 0),
            )
        )
        raise
    except Exception as e:
        event_queue.put(
            ProgressEvent(
                event_type=EventType.ERROR,
                transfer_id=transfer_id,
                direction=TransferDirection.DOWNLOAD,
                filename=f"{repo_id}",
                phase=ProgressPhase.ERROR,
                error=TransferError(message=str(e), error_type=type(e).__name__),
            )
        )
        raise
    finally:
        state_manager.clear_state(transfer_id)