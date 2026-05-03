"""Xet direct download functions with progress tracking."""

from __future__ import annotations

import os
import queue
from dataclasses import dataclass
from typing import Callable, List, Optional

from .callbacks import XetDownloadProgressCallback
from .token import XetTokenManager, is_xet_available
from .types import (
    EventType,
    ProgressEvent,
    ProgressPhase,
    TransferDirection,
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
    if not is_xet_available():
        raise ImportError("hf_xet is not installed.")

    import hf_xet

    transfer_id = transfer_id or generate_transfer_id()
    filename = os.path.basename(dest_path)

    token_manager = XetTokenManager(token, endpoint)
    creds = token_manager.get_download_credentials(xet_file_data)

    download_info = [
        hf_xet.PyXetDownloadInfo(
            destination_path=str(os.path.abspath(dest_path)),
            hash=file_hash,
            file_size=file_size,
        )
    ]

    callback = XetDownloadProgressCallback(
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
            direction=TransferDirection.DOWNLOAD,
            filename=filename,
            phase=ProgressPhase.DOWNLOADING,
            total_bytes=file_size,
        )
    )

    try:
        kwargs = dict(
            endpoint=creds.endpoint,
            token_info=creds.token_info,
            token_refresher=creds.token_refresher,
            progress_updater=[callback.get_wrapper()],
        )
        if request_headers:
            kwargs["request_headers"] = request_headers

        results = hf_xet.download_files(download_info, **kwargs)

        event_queue.put(
            ProgressEvent(
                event_type=EventType.COMPLETE,
                transfer_id=transfer_id,
                direction=TransferDirection.DOWNLOAD,
                filename=filename,
                phase=ProgressPhase.COMPLETE,
                bytes_completed=file_size,
                total_bytes=file_size,
                percentage=100.0,
            )
        )

        return XetDownloadResult(
            success=True,
            filename=filename,
            destination_path=dest_path,
            file_size=file_size,
            transfer_id=transfer_id,
        )

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
    if not is_xet_available():
        raise ImportError("hf_xet is not installed.")

    import hf_xet

    transfer_id = transfer_id or generate_transfer_id()
    total_files = len(file_specs)

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
            XetDownloadProgressCallback(
                filename=filename,
                total_bytes=spec["file_size"],
                event_queue=event_queue,
                report_interval=report_interval,
                transfer_id=transfer_id,
                file_index=i,
                total_files=total_files,
                is_cancelled=is_cancelled,
            )
        )

    token_manager = XetTokenManager(token, endpoint)
    creds = token_manager.get_download_credentials(file_specs[0]["xet_file_data"])

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

    try:
        kwargs = dict(
            endpoint=creds.endpoint,
            token_info=creds.token_info,
            token_refresher=creds.token_refresher,
            progress_updater=[cb.get_wrapper() for cb in callbacks],
        )
        if request_headers:
            kwargs["request_headers"] = request_headers

        results = hf_xet.download_files(download_infos, **kwargs)

        download_results = []
        for i, spec in enumerate(file_specs):
            filename = os.path.basename(spec["dest_path"])
            event_queue.put(
                ProgressEvent(
                    event_type=EventType.COMPLETE,
                    transfer_id=transfer_id,
                    direction=TransferDirection.DOWNLOAD,
                    filename=filename,
                    phase=ProgressPhase.COMPLETE,
                    bytes_completed=spec["file_size"],
                    total_bytes=spec["file_size"],
                    percentage=100.0,
                    file_index=i,
                    total_files=total_files,
                )
            )
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

    except Exception as e:
        for i, spec in enumerate(file_specs):
            filename = os.path.basename(spec["dest_path"])
            event_queue.put(
                ProgressEvent(
                    event_type=EventType.ERROR,
                    transfer_id=transfer_id,
                    direction=TransferDirection.DOWNLOAD,
                    filename=filename,
                    phase=ProgressPhase.ERROR,
                    error=str(e),
                    file_index=i,
                    total_files=total_files,
                )
            )
        raise