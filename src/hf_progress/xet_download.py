"""Xet direct download functions with progress tracking via subprocess isolation.

All ``hf_xet`` calls are executed in isolated child processes using
``XetSubprocessRunner``, so they can be safely terminated without
affecting the main process.
"""

from __future__ import annotations

import os
import queue
from dataclasses import dataclass
from typing import Callable, List, Optional

from ._xet_worker import _download_batch_worker, _download_worker, _serialize_xet_file_data
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


@dataclass
class XetDownloadResult:
    success: bool
    filename: str
    destination_path: str = ""
    file_size: int = 0
    transfer_id: str = ""


def download_file_with_xet(
    file_hash: str,
    file_size: int,
    dest_path: str,
    xet_file_data,
    token: str,
    event_queue: queue.Queue,
    endpoint: Optional[str] = None,
    transfer_id: Optional[str] = None,
    report_interval: float = 0.1,
    request_headers: Optional[dict] = None,
    is_cancelled: Optional[Callable[[], bool]] = None,
) -> XetDownloadResult:
    """Download a single file via Xet in an isolated subprocess.

    The actual ``hf_xet.download_files()`` call runs in a child process,
    which can be safely terminated via ``runner.terminate()`` if the
    transfer hangs or the user cancels.

    Args:
        file_hash: Xet file hash.
        file_size: Expected file size in bytes.
        dest_path: Local destination path.
        xet_file_data: XetFileData object from HfFileMetadata.
        token: HuggingFace API token.
        event_queue: Queue for ProgressEvent objects.
        endpoint: Optional custom Xet endpoint.
        transfer_id: Optional pre-existing transfer ID.
        report_interval: Event reporting interval (seconds).
        request_headers: Optional HTTP headers for Xet requests.
        is_cancelled: Optional cancellation hook (checked in main process).

    Returns:
        XetDownloadResult on success.

    Raises:
        ImportError: If hf_xet is not installed.
        TransferCancelledError: If the transfer is cancelled.
        TransferError: If the download fails.
    """
    if not is_xet_available():
        raise ImportError("hf_xet is not installed.")

    transfer_id = transfer_id or generate_transfer_id()
    filename = os.path.basename(dest_path)

    # Emit START event from main process
    event_queue.put(
        ProgressEvent(
            event_type=EventType.START,
            transfer_id=transfer_id,
            direction=TransferDirection.DOWNLOAD,
            filename=filename,
            phase=ProgressPhase.DOWNLOADING,
            total_bytes=file_size,
        )
    )

    # Serialize xet_file_data for cross-process boundary
    xet_file_data_dict = _serialize_xet_file_data(xet_file_data)

    params = {
        "file_hash": file_hash,
        "file_size": file_size,
        "dest_path": dest_path,
        "xet_file_data": xet_file_data_dict,
        "token": token,
        "endpoint": endpoint,
        "transfer_id": transfer_id,
        "report_interval": report_interval,
        "request_headers": request_headers or {},
        "direction": "download",
        "filename": filename,
    }

    runner = XetSubprocessRunner()
    runner.start(
        worker_func=_download_worker,
        params=params,
        event_queue=event_queue,
    )

    try:
        # Poll for cancellation from main process while waiting
        while True:
            result = runner.wait(timeout=1.0)
            if result is not None:
                break
            # Check main-process cancellation hook
            if is_cancelled is not None and is_cancelled():
                runner.terminate()
                event_queue.put(
                    ProgressEvent.cancelled_event(
                        transfer_id=transfer_id,
                        direction=TransferDirection.DOWNLOAD,
                        filename=filename,
                    )
                )
                raise TransferCancelledError("Download cancelled by user")

        # Process the result
        if result.get("status") == "success":
            return XetDownloadResult(
                success=True,
                filename=result.get("filename", filename),
                destination_path=result.get("destination_path", dest_path),
                file_size=result.get("file_size", file_size),
                transfer_id=transfer_id,
            )
        elif (
            result.get("status") == "cancelled"
            or result.get("error_type") == "TransferCancelledError"
            or "cancelled" in result.get("message", "").lower()
            or "interrupted" in result.get("message", "").lower()
        ):
            raise TransferCancelledError(result.get("message", "Download cancelled by user"))
        else:
            # Error result from worker
            error_msg = result.get("message", "Download failed")
            raise TransferProgressError(error_msg)

    except KeyboardInterrupt:
        runner.terminate()
        event_queue.put(
            ProgressEvent.cancelled_event(
                transfer_id=transfer_id,
                direction=TransferDirection.DOWNLOAD,
                filename=filename,
            )
        )
        raise TransferCancelledError("Download interrupted by user (Ctrl+C)")
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
                direction=TransferDirection.DOWNLOAD,
                filename=filename,
                phase=ProgressPhase.ERROR,
                error=TransferError(message=str(e), error_type=type(e).__name__),
            )
        )
        raise
    finally:
        runner.terminate()


def download_files_with_xet(
    file_specs: List[dict],
    token: str,
    event_queue: queue.Queue,
    endpoint: Optional[str] = None,
    transfer_id: Optional[str] = None,
    report_interval: float = 0.1,
    request_headers: Optional[dict] = None,
    is_cancelled: Optional[Callable[[], bool]] = None,
) -> List[XetDownloadResult]:
    """Download multiple files via Xet in an isolated subprocess.

    Args:
        file_specs: List of dicts with keys: dest_path, hash, file_size, xet_file_data.
        token: HuggingFace API token.
        event_queue: Queue for ProgressEvent objects.
        endpoint: Optional custom Xet endpoint.
        transfer_id: Optional pre-existing transfer ID.
        report_interval: Event reporting interval (seconds).
        request_headers: Optional HTTP headers for Xet requests.
        is_cancelled: Optional cancellation hook.

    Returns:
        List of XetDownloadResult on success.

    Raises:
        ImportError: If hf_xet is not installed.
        TransferCancelledError: If the transfer is cancelled.
    """
    if not is_xet_available():
        raise ImportError("hf_xet is not installed.")

    transfer_id = transfer_id or generate_transfer_id()
    total_files = len(file_specs)

    # Emit START events from main process
    for i, spec in enumerate(file_specs):
        filename = os.path.basename(spec["dest_path"])
        event_queue.put(
            ProgressEvent(
                event_type=EventType.START,
                transfer_id=transfer_id,
                direction=TransferDirection.DOWNLOAD,
                filename=filename,
                phase=ProgressPhase.DOWNLOADING,
                total_bytes=spec["file_size"],
                file_index=i,
                total_files=total_files,
            )
        )

    # Serialize xet_file_data for each file spec
    serialized_specs = []
    for spec in file_specs:
        serialized_spec = dict(spec)
        serialized_spec["xet_file_data"] = _serialize_xet_file_data(spec.get("xet_file_data"))
        serialized_specs.append(serialized_spec)

    params = {
        "file_specs": serialized_specs,
        "token": token,
        "endpoint": endpoint,
        "transfer_id": transfer_id,
        "report_interval": report_interval,
        "request_headers": request_headers or {},
    }

    runner = XetSubprocessRunner()
    runner.start(
        worker_func=_download_batch_worker,
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
                for i, spec in enumerate(file_specs):
                    filename = os.path.basename(spec["dest_path"])
                    event_queue.put(
                        ProgressEvent.cancelled_event(
                            transfer_id=transfer_id,
                            direction=TransferDirection.DOWNLOAD,
                            filename=filename,
                            file_index=i,
                            total_files=total_files,
                        )
                    )
                raise TransferCancelledError("Download cancelled by user")

        if result.get("status") == "success":
            # Batch worker sends individual results — reconstruct from result
            download_results = []
            for i, spec in enumerate(file_specs):
                filename = os.path.basename(spec["dest_path"])
                download_results.append(
                    XetDownloadResult(
                        success=True,
                        filename=filename,
                        destination_path=spec["dest_path"],
                        file_size=spec["file_size"],
                        transfer_id=transfer_id,
                    )
                )
            return download_results
        elif (
            result.get("status") == "cancelled"
            or result.get("error_type") == "TransferCancelledError"
            or "cancelled" in result.get("message", "").lower()
            or "interrupted" in result.get("message", "").lower()
        ):
            raise TransferCancelledError(result.get("message", "Download cancelled by user"))
        else:
            error_msg = result.get("message", "Batch download failed")
            error_type = result.get("error_type", "Exception")
            # Emit ERROR events for all files
            for i, spec in enumerate(file_specs):
                filename = os.path.basename(spec["dest_path"])
                event_queue.put(
                    ProgressEvent(
                        event_type=EventType.ERROR,
                        transfer_id=transfer_id,
                        direction=TransferDirection.DOWNLOAD,
                        filename=filename,
                        phase=ProgressPhase.ERROR,
                        error=TransferError(message=error_msg, error_type=error_type),
                        file_index=i,
                        total_files=total_files,
                    )
                )
            raise TransferProgressError(error_msg)

    except KeyboardInterrupt:
        runner.terminate()
        for i, spec in enumerate(file_specs):
            filename = os.path.basename(spec["dest_path"])
            event_queue.put(
                ProgressEvent.cancelled_event(
                    transfer_id=transfer_id,
                    direction=TransferDirection.DOWNLOAD,
                    filename=filename,
                    file_index=i,
                    total_files=total_files,
                )
            )
        raise TransferCancelledError("Download interrupted by user (Ctrl+C)")
    except TransferCancelledError:
        raise
    except TransferProgressError:
        raise
    except Exception as e:
        runner.terminate()
        for i, spec in enumerate(file_specs):
            filename = os.path.basename(spec["dest_path"])
            event_queue.put(
                ProgressEvent(
                    event_type=EventType.ERROR,
                    transfer_id=transfer_id,
                    direction=TransferDirection.DOWNLOAD,
                    filename=filename,
                    phase=ProgressPhase.ERROR,
                    error=TransferError(message=str(e), error_type=type(e).__name__),
                    file_index=i,
                    total_files=total_files,
                )
            )
        raise
    finally:
        runner.terminate()
