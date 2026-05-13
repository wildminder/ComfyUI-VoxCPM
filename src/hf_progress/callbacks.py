"""Progress callback classes for intercepting transfer progress."""

from __future__ import annotations

import contextlib
import logging
import queue
import threading
import time
from typing import Callable, Dict, Optional

from tqdm.auto import tqdm as base_tqdm

from .types import (
    EventType,
    ProgressEvent,
    ProgressPhase,
    TransferCancelledError,
    TransferDirection,
)

logger = logging.getLogger(__name__)

class TransferStateManager:
    """Thread-safe manager for aggregate transfer states."""
    
    def __init__(self):
        self._states: Dict[str, dict] = {}
        self._lock = threading.Lock()

    def init_download(self, transfer_id: str):
        with self._lock:
            if transfer_id not in self._states:
                self._states[transfer_id] = {
                    "files_completed": 0,
                    "total_files": 0,
                    "bytes_completed": 0,
                    "total_bytes": 0,
                }

    def init_upload(self, transfer_id: str, filename: str, total_bytes: int, event_queue: queue.Queue):
        with self._lock:
            if transfer_id not in self._states:
                self._states[transfer_id] = {
                    "filename": filename,
                    "total_bytes": total_bytes,
                    "bytes_completed": 0,
                    "event_queue": event_queue,
                    "completed_emitted": False,
                    "start_time": time.time(),
                }

    def update_download_files(self, transfer_id: str, files_completed: int, total_files: int):
        with self._lock:
            if transfer_id in self._states:
                self._states[transfer_id]["files_completed"] = files_completed
                if total_files > 0:
                    self._states[transfer_id]["total_files"] = total_files

    def update_download_bytes(self, transfer_id: str, bytes_completed: int, total_bytes: int):
        with self._lock:
            if transfer_id in self._states:
                self._states[transfer_id]["bytes_completed"] = bytes_completed
                self._states[transfer_id]["total_bytes"] = total_bytes

    def add_upload_bytes(self, transfer_id: str, byte_increment: int) -> dict:
        """Accumulates bytes for multipart uploads and returns the current state."""
        with self._lock:
            state = self._states.get(transfer_id)
            if state:
                state["bytes_completed"] += byte_increment
                # Return a copy for safe event emission
                return dict(state)
            return {}

    def mark_upload_completed(self, transfer_id: str):
        with self._lock:
            state = self._states.get(transfer_id)
            if state:
                state["completed_emitted"] = True

    def get_state(self, transfer_id: str) -> dict:
        with self._lock:
            return dict(self._states.get(transfer_id, {}))

    def clear_state(self, transfer_id: str):
        with self._lock:
            self._states.pop(transfer_id, None)


# Global thread-safe state manager
state_manager = TransferStateManager()


class _DummyFile:
    """A dummy file-like object to silently sink tqdm output."""
    def write(self, x: str) -> int:
        return len(x)
    def flush(self) -> None:
        pass


_dummy_file = _DummyFile()


class XetProgressCallback:
    """Unified progress callback for Xet uploads and downloads.

    Args:
        filename: Name of the file being transferred.
        total_bytes: Expected total bytes for this file.
        event_queue: Queue to emit ProgressEvents into.
        direction: TransferDirection.UPLOAD or DOWNLOAD.
        phase: ProgressPhase.UPLOADING or DOWNLOADING.
        report_interval: Minimum seconds between progress reports (deprecated logic).
        transfer_id: Unique transfer identifier.
        file_index: Index of this file in a multi-file transfer.
        total_files: Total number of files in the transfer.
        is_cancelled: Optional callable returning True to cancel.
    """

    def __init__(
        self,
        filename: str,
        total_bytes: int,
        event_queue: queue.Queue,
        direction: TransferDirection = TransferDirection.UPLOAD,
        phase: ProgressPhase = ProgressPhase.UPLOADING,
        report_interval: float = 0.1,
        transfer_id: str = "",
        file_index: int = 0,
        total_files: int = 1,
        is_cancelled: Optional[Callable[[], bool]] = None,
    ):
        self.filename = filename
        self.total_bytes = total_bytes
        self.event_queue = event_queue
        self.direction = direction
        self.phase = phase
        self.report_interval = report_interval
        self.transfer_id = transfer_id
        self.file_index = file_index
        self.total_files = total_files
        self.is_cancelled = is_cancelled
        self._dropped_count = 0
        self._start_time = time.time()

    def get_wrapper(self):
        def progress_updater(total_update, item_updates):
            return self(total_update, item_updates)
        return progress_updater

    def _resolve_display_completed(self, bytes_completed, total_bytes, transfer_completed):
        """Choose which byte count to display as progress.

        For uploads, always show per-file bytes_completed.
        For downloads, prefer transfer_completed when per-file
        progress hasn't caught up yet (e.g. dedup-aware display).
        """
        if self.direction == TransferDirection.UPLOAD:
            return bytes_completed
        # Download: show transfer-level progress when per-file is stale
        if bytes_completed >= total_bytes > 0:
            return bytes_completed
        if transfer_completed > 0:
            return transfer_completed
        return bytes_completed

    def __call__(self, total_update, item_updates):
        if self.is_cancelled and self.is_cancelled():
            raise TransferCancelledError("Transfer cancelled by user")

        item_update = next((item for item in item_updates if getattr(item, "item_name", "") == self.filename), None)

        bytes_completed = getattr(total_update, "total_bytes_completed", 0)
        total_bytes = getattr(total_update, "total_bytes", 0) or self.total_bytes
        speed = getattr(total_update, "total_bytes_completion_rate", 0) or 0
        transfer_completed = getattr(total_update, "total_transfer_bytes_completed", 0)
        transfer_total = getattr(total_update, "total_transfer_bytes", 0)
        transfer_speed = getattr(total_update, "total_transfer_bytes_completion_rate", 0) or 0

        if self.total_files > 1 and item_update is not None:
            bytes_completed = getattr(item_update, "bytes_completed", 0)
            total_bytes = getattr(item_update, "total_bytes", 0) or self.total_bytes

        display_completed = self._resolve_display_completed(
            bytes_completed, total_bytes, transfer_completed
        )

        active_speed = transfer_speed or speed

        # NTH-001: Progress Estimation when bytes = 0 but speed > 0
        if display_completed == 0 and active_speed > 0:
            elapsed = time.time() - self._start_time
            estimated = int(active_speed * elapsed)
            if total_bytes > 0:
                # Cap the estimation at 99% to prevent jumping to completion
                estimated = min(estimated, int(total_bytes * 0.99))
            display_completed = estimated

        percentage = ((display_completed / total_bytes * 100) if total_bytes > 0 else 0)
        dedup_saved = max(0, bytes_completed - transfer_completed) if self.total_files == 1 else 0

        self._emit(display_completed, total_bytes, percentage, active_speed, transfer_completed, transfer_total, transfer_speed, dedup_saved)

    def _emit(self, bytes_completed, total_bytes, percentage, speed, transfer_completed, transfer_total, transfer_speed, dedup_saved):
        # Only include Xet-specific transfer stats for single-file transfers
        include_transfer_stats = self.total_files == 1
        event = ProgressEvent(
            event_type=EventType.PROGRESS,
            transfer_id=self.transfer_id,
            direction=self.direction,
            filename=self.filename,
            phase=self.phase,
            bytes_completed=bytes_completed,
            total_bytes=total_bytes,
            percentage=percentage,
            speed=speed,
            file_index=self.file_index,
            total_files=self.total_files,
            transfer_bytes_completed=transfer_completed if include_transfer_stats else 0,
            transfer_bytes_total=transfer_total if include_transfer_stats else 0,
            transfer_speed=transfer_speed if include_transfer_stats else 0,
            dedup_saved_bytes=dedup_saved,
        )
        try:
            self.event_queue.put_nowait(event)
        except queue.Full:
            self._dropped_count += 1
            logger.warning("Progress event queue full — dropping %s event for %s; %d dropped total", event.event_type.value, event.filename, self._dropped_count)


class XetUploadProgressCallback(XetProgressCallback):
    """Backward-compatible subclass for Xet upload progress."""
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("direction", TransferDirection.UPLOAD)
        kwargs.setdefault("phase", ProgressPhase.UPLOADING)
        super().__init__(*args, **kwargs)


class XetDownloadProgressCallback(XetProgressCallback):
    """Backward-compatible subclass for Xet download progress."""
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("direction", TransferDirection.DOWNLOAD)
        kwargs.setdefault("phase", ProgressPhase.DOWNLOADING)
        super().__init__(*args, **kwargs)


class DownloadProgressTqdm(base_tqdm):
    def __init__(self, *args, **kwargs):
        self._event_queue = kwargs.pop("event_queue", None)
        self._transfer_id = kwargs.pop("transfer_id", "")
        self._filename = kwargs.pop("filename", "")
        self._report_interval = kwargs.pop("report_interval", 0.1)
        self._is_cancelled = kwargs.pop("is_cancelled", None)
        self._start_time = time.time()
        self._closed = False
        
        self.is_bytes_bar = (kwargs.get("unit", "it") in ("B", "iB"))
        kwargs.pop("name", None)
        super().__init__(*args, **kwargs)
        
        if not self._filename:
            self._filename = getattr(self, "desc", "unknown") or "unknown"
        
        state_manager.init_download(self._transfer_id)
        if not self.is_bytes_bar:
            state_manager.update_download_files(self._transfer_id, 0, getattr(self, "total", 0) or 0)

    def update(self, n=1):
        if self._is_cancelled is not None and self._is_cancelled():
            raise TransferCancelledError("Transfer cancelled by user")

        result = super().update(n)

        if self._event_queue is None or n == 0:
            return result

        if not getattr(self, "is_bytes_bar", False):
            state_manager.update_download_files(self._transfer_id, getattr(self, "n", 0), getattr(self, "total", 0) or 0)
            return result

        now = time.time()
        bytes_completed = getattr(self, "n", 0)
        total_bytes = getattr(self, "total", 0) or 0
        percentage = ((bytes_completed / total_bytes * 100) if total_bytes > 0 else 0)

        state_manager.update_download_bytes(self._transfer_id, bytes_completed, total_bytes)

        speed = getattr(self, "format_dict", {}).get("rate") or 0
        if not speed and bytes_completed > 0:
            elapsed = now - self._start_time
            speed = bytes_completed / elapsed if elapsed > 0 else 0

        state = state_manager.get_state(self._transfer_id)
        files_completed = state.get("files_completed", 0)
        total_files = state.get("total_files", 0)

        event = ProgressEvent(
            event_type=EventType.PROGRESS,
            transfer_id=self._transfer_id,
            direction=TransferDirection.DOWNLOAD,
            filename=self._filename,
            phase=ProgressPhase.DOWNLOADING,
            bytes_completed=bytes_completed,
            total_bytes=total_bytes,
            percentage=percentage,
            speed=speed or 0,
            file_index=files_completed,
            total_files=total_files,
        )
        try:
            self._event_queue.put_nowait(event)
        except queue.Full:
            logger.warning("Download progress event queue full — dropping %s event", event.event_type.value)
        return result

    def close(self):
        if self._closed:
            super().close()
            return
        self._closed = True

        if self._event_queue is not None and getattr(self, "total", None) is not None and getattr(self, "is_bytes_bar", False):
            n_val = getattr(self, "n", 0)
            total_val = getattr(self, "total", 0)
            
            is_complete = n_val >= total_val
            is_xet_cached = n_val == 0 and total_val > 0

            if is_complete or is_xet_cached:
                final_bytes = total_val if is_xet_cached else n_val
                state = state_manager.get_state(self._transfer_id)
                total_files = state.get("total_files", 0)
                files_completed = total_files if total_files > 0 else state.get("files_completed", 0)

                event = ProgressEvent(
                    event_type=EventType.COMPLETE,
                    transfer_id=self._transfer_id,
                    direction=TransferDirection.DOWNLOAD,
                    filename=self._filename,
                    phase=ProgressPhase.COMPLETE,
                    bytes_completed=final_bytes,
                    total_bytes=total_val,
                    percentage=100.0,
                    file_index=files_completed,
                    total_files=total_files,
                )
                self._event_queue.put(event)
                
        super().close()

    @classmethod
    def bind(
        cls,
        event_queue: queue.Queue,
        transfer_id: str,
        filename: str = "",
        report_interval: float = 0.1,
        is_cancelled: Optional[Callable[[], bool]] = None,
    ) -> type:
        _queue = event_queue
        _tid = transfer_id
        _fname = filename
        _interval = report_interval
        _cancel_hook = is_cancelled

        class BoundDownloadTqdm(cls):
            def __init__(self, *args, **kwargs):
                kwargs.setdefault("event_queue", _queue)
                kwargs.setdefault("transfer_id", _tid)
                kwargs.setdefault("filename", _fname)
                kwargs.setdefault("report_interval", _interval)
                kwargs.setdefault("is_cancelled", _cancel_hook)
                kwargs["file"] = _dummy_file
                super().__init__(*args, **kwargs)

        BoundDownloadTqdm.__name__ = f"BoundDownloadTqdm_{transfer_id[:8]}"
        BoundDownloadTqdm.__qualname__ = f"BoundDownloadTqdm_{transfer_id[:8]}"
        return BoundDownloadTqdm


@contextlib.contextmanager
def tqdm_upload_patcher(
    event_queue: queue.Queue,
    transfer_id: str = "",
    filename: str = "",
    report_interval: float = 0.1,
    is_cancelled: Optional[Callable[[], bool]] = None,
    total_bytes: int = 0,
):
    import tqdm.auto as tqdm_auto_module

    original_tqdm = tqdm_auto_module.tqdm
    _patch_active = True

    # Register this upload in the state manager.
    state_manager.init_upload(transfer_id, filename, total_bytes, event_queue)

    class UploadProgressTqdm(original_tqdm):
        def __init__(self, *args, **kwargs):
            kwargs["file"] = _dummy_file
            super().__init__(*args, **kwargs)
            
            self._managed_state = state_manager.get_state(transfer_id)
            
            self._upload_is_file_bar = (
                getattr(self, "total", None) is not None
                and getattr(self, "total", 0) > 0
                and getattr(self, "unit", "") in ("B", "iB")
            )
            
            self._last_n = 0

            if getattr(self, "_upload_is_file_bar", False) and self._managed_state and self._managed_state["bytes_completed"] == 0:
                event = ProgressEvent(
                    event_type=EventType.START,
                    transfer_id=transfer_id,
                    direction=TransferDirection.UPLOAD,
                    filename=self._managed_state["filename"],
                    phase=ProgressPhase.UPLOADING,
                    total_bytes=self._managed_state["total_bytes"],
                )
                try:
                    event_queue.put_nowait(event)
                except queue.Full:
                    logger.warning("Upload progress event queue full — dropping %s event", event.event_type.value)

        def update(self, n=1):
            if is_cancelled is not None and is_cancelled():
                raise TransferCancelledError("Transfer cancelled by user")

            result = super().update(n)

            if not getattr(self, "_upload_is_file_bar", False) or not _patch_active or n == 0:
                return result

            current_n = getattr(self, "n", 0)
            delta = current_n - self._last_n
            self._last_n = current_n

            if delta > 0:
                state = state_manager.add_upload_bytes(transfer_id, delta)
                if state:
                    bytes_completed = state["bytes_completed"]
                    total_bytes = state["total_bytes"]
                    percentage = ((bytes_completed / total_bytes * 100) if total_bytes > 0 else 0)

                    now = time.time()
                    elapsed = now - state["start_time"]
                    speed = bytes_completed / elapsed if elapsed > 0 else 0

                    event = ProgressEvent(
                        event_type=EventType.PROGRESS,
                        transfer_id=transfer_id,
                        direction=TransferDirection.UPLOAD,
                        filename=state["filename"],
                        phase=ProgressPhase.UPLOADING,
                        bytes_completed=bytes_completed,
                        total_bytes=total_bytes,
                        percentage=percentage,
                        speed=speed,
                    )
                    try:
                        state["event_queue"].put_nowait(event)
                    except queue.Full:
                        logger.warning("Upload progress event queue full — dropping %s event", event.event_type.value)
            return result

        def close(self):
            if not getattr(self, "_upload_is_file_bar", False) or not _patch_active:
                super().close()
                return
                
            state = state_manager.get_state(transfer_id)
            if state and not state.get("completed_emitted", False):
                if state["bytes_completed"] >= state["total_bytes"]:
                    state_manager.mark_upload_completed(transfer_id)
                    event = ProgressEvent(
                        event_type=EventType.COMPLETE,
                        transfer_id=transfer_id,
                        direction=TransferDirection.UPLOAD,
                        filename=state["filename"],
                        phase=ProgressPhase.COMPLETE,
                        bytes_completed=state["bytes_completed"],
                        total_bytes=state["total_bytes"],
                        percentage=100.0,
                    )
                    state["event_queue"].put(event)
                    
            super().close()

    tqdm_auto_module.tqdm = UploadProgressTqdm
    try:
        yield
    finally:
        _patch_active = False
        tqdm_auto_module.tqdm = original_tqdm