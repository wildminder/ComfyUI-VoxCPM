"""Download manager for VoxCPM models with real-time progress tracking and cancel support."""

from __future__ import annotations

import logging
import os
import queue
import shutil
import sys
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Xet Configuration Flag ─────────────────────────────────────────
# Set to False to enable hf_xet downloading (if installed). 
# Set to True to enforce standard HTTP/tqdm downloads (no-xet).
DISABLE_XET_DOWNLOADS = True


def _suppress_hf_http_logs() -> None:
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        from huggingface_hub import logging as hf_logging
        hf_logging.set_verbosity_error()
        hf_logging.disable_propagation()
        _hf_logger = hf_logging.get_logger()
        for _handler in _hf_logger.handlers:
            _handler.setLevel(logging.ERROR)
    except ImportError:
        pass
    for _logger_name in ("urllib3.connectionpool", "filelock"):
        _lh = logging.getLogger(_logger_name)
        if _lh.level == logging.NOTSET or _lh.level < logging.WARNING:
            _lh.setLevel(logging.WARNING)

def _restore_hf_http_logs() -> None:
    logging.getLogger("httpx").setLevel(logging.NOTSET)
    try:
        from huggingface_hub import logging as hf_logging
        hf_logging.set_verbosity_warning()
        hf_logging.enable_propagation()
        _hf_logger = hf_logging.get_logger()
        for _handler in _hf_logger.handlers:
            _handler.setLevel(logging.NOTSET)
    except ImportError:
        pass
    for _logger_name in ("urllib3.connectionpool", "filelock"):
        logging.getLogger(_logger_name).setLevel(logging.NOTSET)

def _format_bytes(num_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} TB"

def _format_speed(bytes_per_sec: float) -> str:
    return f"{_format_bytes(bytes_per_sec)}/s"

_EVENT_POLL_TIMEOUT = 0.05

# Maximum download attempts (1 initial + retries on transient network errors)
MAX_DOWNLOAD_RETRIES = 2

class DownloadCancelledError(Exception):
    def __init__(self, model_name: str):
        self.model_name = model_name
        super().__init__(f"Download cancelled for model '{model_name}'")

class DownloadError(Exception):
    def __init__(self, model_name: str, message: str):
        self.model_name = model_name
        self.message = message
        super().__init__(f"Download failed for model '{model_name}': {message}")

class DownloadStatus(str, Enum):
    IDLE = "idle"
    DOWNLOADING = "downloading"
    COMPLETE = "complete"
    CANCELLED = "cancelled"
    ERROR = "error"

@dataclass
class DownloadState:
    model_name: str
    repo_id: str
    local_dir: str
    status: DownloadStatus = DownloadStatus.IDLE
    progress_percentage: float = 0.0
    current_file: str = ""
    file_index: int = 0
    total_files: int = 0
    speed: float = 0.0
    bytes_completed: int = 0
    total_bytes: int = 0
    transfer_bytes_completed: int = 0
    transfer_bytes_total: int = 0
    transfer_speed: float = 0.0
    dedup_saved_bytes: int = 0
    error_message: Optional[str] = None
    cancel_requested_at: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "model_name": self.model_name,
            "repo_id": self.repo_id,
            "local_dir": self.local_dir,
            "status": self.status.value,
            "progress_percentage": round(self.progress_percentage, 2),
            "current_file": self.current_file,
            "file_index": self.file_index,
            "total_files": self.total_files,
            "speed": self.speed,
            "bytes_completed": self.bytes_completed,
            "total_bytes": self.total_bytes,
        }
        if self.transfer_bytes_completed or self.transfer_bytes_total:
            d["transfer_bytes_completed"] = self.transfer_bytes_completed
            d["transfer_bytes_total"] = self.transfer_bytes_total
            d["transfer_speed"] = self.transfer_speed
        if self.dedup_saved_bytes:
            d["dedup_saved_bytes"] = self.dedup_saved_bytes
        if self.error_message is not None:
            d["error_message"] = self.error_message
        if self.cancel_requested_at is not None:
            d["cancel_requested_at"] = self.cancel_requested_at
        return d

class DownloadManager:
    def __init__(self):
        self._active_downloads: Dict[str, DownloadState] = {}
        self._active_trackers: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def download_model(
        self,
        model_name: str,
        repo_id: str,
        local_dir: str,
        client_id: Optional[str] = None,
    ) -> str:
        # Dynamically apply our flag to both the OS environment and huggingface_hub's 
        # internal cache before initializing the tracker or fetching Xet status.
        if DISABLE_XET_DOWNLOADS:
            os.environ["HF_HUB_DISABLE_XET"] = "1"
            try:
                import huggingface_hub.constants
                huggingface_hub.constants.HF_HUB_DISABLE_XET = True
            except ImportError:
                pass
        else:
            os.environ.pop("HF_HUB_DISABLE_XET", None)
            try:
                import huggingface_hub.constants
                huggingface_hub.constants.HF_HUB_DISABLE_XET = False
            except ImportError:
                pass

        from src.hf_progress import HfProgressTracker, EventType, TransferCancelledError

        return self._download_model_inner(
            model_name, repo_id, local_dir, client_id,
            HfProgressTracker, EventType, TransferCancelledError,
        )

    def _download_model_inner(
        self,
        model_name: str,
        repo_id: str,
        local_dir: str,
        client_id: Optional[str],
        HfProgressTracker,
        EventType,
        TransferCancelledError,
    ) -> str:
        from src.hf_progress import is_xet_available

        token = self._get_token()
        tracker = HfProgressTracker(token=token)

        xet_avail = not DISABLE_XET_DOWNLOADS and is_xet_available()
        xet_status = "ENABLED (subprocess)" if xet_avail else "DISABLED (standard HTTP)"
        
        logger.info(
            f"Starting download for '{model_name}' from '{repo_id}' "
            f"(Xet: {xet_status})"
        )

        with self._lock:
            self._active_trackers[model_name] = tracker
            self._active_downloads[model_name] = DownloadState(
                model_name=model_name,
                repo_id=repo_id,
                local_dir=local_dir,
                status=DownloadStatus.DOWNLOADING,
            )

        self._emit_download_event(
            model_name=model_name,
            event_type="start",
            phase="downloading",
            client_id=client_id,
            extra_data={
                "repo_id": repo_id,
                "local_dir": local_dir,
            },
        )

        _suppress_hf_http_logs()

        try:
            for attempt in range(1, MAX_DOWNLOAD_RETRIES + 1):
                result_path = None
                error_occurred = None
                was_cancelled = False

                def do_download():
                    nonlocal result_path, error_occurred, was_cancelled
                    try:
                        result_path = tracker.download_snapshot(
                            repo_id=repo_id,
                            local_dir=local_dir,
                            transfer_id=model_name,
                        )
                    except TransferCancelledError:
                        was_cancelled = True
                    except KeyboardInterrupt:
                        was_cancelled = True
                    except Exception as e:
                        error_occurred = e

                # ALWAYS use a background thread for the download logic.
                # This prevents blocking the main ComfyUI thread, ensuring the while-loop
                # correctly drains tracker.event_queue and pushes UI updates in real-time.
                download_thread = threading.Thread(target=do_download, daemon=True)
                download_thread.start()

                try:
                    while download_thread.is_alive() or not tracker.event_queue.empty():
                        if tracker.is_cancelled(transfer_id=model_name) or was_cancelled:
                            self._update_state(model_name, status=DownloadStatus.CANCELLED)
                            self._emit_download_event(
                                model_name=model_name,
                                event_type="error",
                                phase="error",
                                client_id=client_id,
                                extra_data={"error": "Download cancelled by user"},
                            )
                            self._print_console_progress(model_name, cancelled=True)
                            tracker.cancel(transfer_id=model_name)
                            download_thread.join(timeout=5.0)
                            if download_thread.is_alive():
                                logger.warning(
                                    f"Download thread for '{model_name}' did not exit within 5s "
                                    f"after cancellation. Resources may temporarily leak."
                                )
                            self._cleanup(model_name, local_dir)
                            raise DownloadCancelledError(model_name)

                        try:
                            event = tracker.event_queue.get(timeout=_EVENT_POLL_TIMEOUT)
                        except queue.Empty:
                            continue

                        self._update_state_from_event(model_name, event)
                        self._emit_progress_event(model_name, event, client_id)

                        if event.event_type == EventType.PROGRESS:
                            self._print_console_progress(model_name)

                        if event.event_type in (EventType.COMPLETE, EventType.ERROR, EventType.CANCELLED):
                            if event.event_type == EventType.ERROR:
                                error_occurred = event.error or "Unknown download error"
                            if event.event_type == EventType.CANCELLED:
                                was_cancelled = True
                            break

                    download_thread.join(timeout=30.0)

                    # Drain any remaining events from the queue post-completion
                    while not tracker.event_queue.empty():
                        try:
                            event = tracker.event_queue.get_nowait()
                            self._update_state_from_event(model_name, event)
                            self._emit_progress_event(model_name, event, client_id)
                        except queue.Empty:
                            break

                    # Final cancel check
                    if tracker.is_cancelled(transfer_id=model_name) or was_cancelled:
                        self._update_state(model_name, status=DownloadStatus.CANCELLED)
                        self._emit_download_event(
                            model_name=model_name,
                            event_type="error",
                            phase="error",
                            client_id=client_id,
                            extra_data={"error": "Download cancelled by user"},
                        )
                        self._print_console_progress(model_name, cancelled=True)
                        self._cleanup(model_name, local_dir)
                        raise DownloadCancelledError(model_name)

                    if error_occurred:
                        # If we have retries left, send a retry notification and continue
                        if attempt < MAX_DOWNLOAD_RETRIES:
                            logger.warning(
                                f"Download attempt {attempt}/{MAX_DOWNLOAD_RETRIES} failed for "
                                f"'{model_name}': {error_occurred}. Retrying..."
                            )
                            # Send retry notification to frontend
                            try:
                                from server import PromptServer
                                if PromptServer.instance is not None:
                                    PromptServer.instance.send_sync("voxcpm.status", {
                                        "severity": "warn",
                                        "summary": f"Retrying download: {model_name}",
                                        "detail": f"Attempt {attempt + 1}/{MAX_DOWNLOAD_RETRIES} after network error: {error_occurred}",
                                        "life": 5000,
                                    }, client_id)
                            except Exception:
                                pass
                            # Clean up partial download before retrying
                            self._cleanup(model_name, local_dir)
                            continue  # Retry

                        # Max retries reached — raise error
                        self._update_state(
                            model_name=model_name,
                            status=DownloadStatus.ERROR,
                            error_message=str(error_occurred),
                        )
                        self._emit_download_event(
                            model_name=model_name,
                            event_type="error",
                            phase="error",
                            client_id=client_id,
                            extra_data={"error": str(error_occurred)},
                        )
                        self._print_console_progress(model_name, error=True)
                        raise DownloadError(model_name, str(error_occurred))

                    # Success
                    self._update_state(model_name, status=DownloadStatus.COMPLETE)
                    self._emit_download_event(
                        model_name=model_name,
                        event_type="complete",
                        phase="complete",
                        client_id=client_id,
                        extra_data={"local_dir": local_dir},
                    )
                    self._print_console_progress(model_name, complete=True)

                    return result_path or local_dir

                except DownloadCancelledError:
                    self._cleanup(model_name, local_dir)
                    raise

            # Should not reach here, but just in case
            raise DownloadError(model_name, "Download failed after all retries")
        finally:
            _restore_hf_http_logs()

    def cancel_download(self, model_name: str) -> bool:
        with self._lock:
            tracker = self._active_trackers.get(model_name)
            if tracker is None:
                return False

            tracker.cancel(transfer_id=model_name)

            state = self._active_downloads.get(model_name)
            if state is not None:
                import time as _time
                state.cancel_requested_at = _time.time()

            logger.info(f"Cancel requested for model '{model_name}'")
            return True

    def get_status(self, model_name: str) -> Optional[DownloadState]:
        with self._lock:
            return self._active_downloads.get(model_name)

    def get_all_statuses(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return {
                name: state.to_dict()
                for name, state in self._active_downloads.items()
            }

    @staticmethod
    def _get_token() -> Optional[str]:
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if token:
            return token
        try:
            from huggingface_hub import HfFolder
            token = HfFolder.get_token()
            if token:
                return token
        except Exception:
            pass
        return None

    @staticmethod
    def _emit_download_event(
        model_name: str,
        event_type: str,
        phase: str,
        client_id: Optional[str] = None,
        extra_data: Optional[Dict[str, Any]] = None,
    ):
        try:
            from server import PromptServer
            if PromptServer.instance is None:
                return
            payload: Dict[str, Any] = {
                "model_name": model_name,
                "event_type": event_type,
                "phase": phase,
                "transfer_id": model_name,
            }
            if extra_data:
                payload.update(extra_data)

            if client_id:
                PromptServer.instance.send_sync("voxcpm.download_progress", payload, client_id)
            else:
                PromptServer.instance.send_sync("voxcpm.download_progress", payload)
        except Exception:
            pass
    
    @staticmethod
    def _emit_progress_event(
        model_name: str,
        event: Any,
        client_id: Optional[str] = None,
    ):
        try:
            from server import PromptServer
            if PromptServer.instance is None:
                return
            payload = event.to_dict()
            payload["model_name"] = model_name
            if client_id:
                PromptServer.instance.send_sync("voxcpm.download_progress", payload, client_id)
            else:
                PromptServer.instance.send_sync("voxcpm.download_progress", payload)
        except Exception:
            pass

    def _update_state(
        self,
        model_name: str,
        status: Optional[DownloadStatus] = None,
        error_message: Optional[str] = None,
    ):
        with self._lock:
            state = self._active_downloads.get(model_name)
            if state is None:
                return
            if status is not None:
                state.status = status
            if error_message is not None:
                state.error_message = error_message

    def _update_state_from_event(self, model_name: str, event: Any):
        with self._lock:
            state = self._active_downloads.get(model_name)
            if state is None:
                return
            state.progress_percentage = event.percentage
            state.current_file = event.filename
            state.file_index = event.file_index
            state.total_files = event.total_files

            from src.hf_progress import EventType as _EventType
            if event.event_type == _EventType.COMPLETE:
                state.status = DownloadStatus.COMPLETE
                return
            if event.event_type == _EventType.CANCELLED:
                state.status = DownloadStatus.CANCELLED
                return

            state.speed = event.speed
            state.bytes_completed = event.bytes_completed
            state.total_bytes = event.total_bytes

            state.transfer_bytes_completed = event.transfer_bytes_completed
            state.transfer_bytes_total = event.transfer_bytes_total
            state.transfer_speed = event.transfer_speed
            state.dedup_saved_bytes = event.dedup_saved_bytes
        
            if event.error:
                state.error_message = event.error
                state.status = DownloadStatus.ERROR

    def _print_console_progress(
        self,
        model_name: str,
        complete: bool = False,
        error: bool = False,
        cancelled: bool = False,
    ) -> None:
        """Log download progress to the logger (replaces print statements).

        Previously used print() with \r for console progress bars, but this
        caused inconsistent output and interfered with logging. Now uses
        logger.info/error for consistent, filterable log output.
        """
        with self._lock:
            state = self._active_downloads.get(model_name)
            if state is None:
                return

        if cancelled:
            logger.info(f"Download cancelled: {model_name}")
            return

        if error:
            msg = state.error_message or "unknown error"
            logger.error(f"Download failed: {model_name} - {msg}")
            return

        if complete:
            total = _format_bytes(state.total_bytes) if state.total_bytes else "unknown size"
            logger.info(f"Download complete: {model_name} ({total})")
            return

        pct = state.progress_percentage
        speed_str = _format_speed(state.speed) if state.speed > 0 else "-"

        file_info = ""
        if state.total_files > 1:
            file_info = f" file {state.file_index + 1}/{state.total_files}"

        # Log progress at debug level to avoid spamming the console.
        # The frontend receives real-time updates via WebSocket events.
        logger.debug(
            f"Downloading {model_name}: {pct:.1f}% {speed_str}{file_info}"
        )

    def _cleanup(self, model_name: str, local_dir: str):
        if os.path.exists(local_dir):
            try:
                for item in os.listdir(local_dir):
                    item_path = os.path.join(local_dir, item)
                    if os.path.isfile(item_path):
                        os.remove(item_path)
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)
            except OSError as e:
                logger.warning(f"Failed to clean up download directory {local_dir}: {e}")

        with self._lock:
            self._active_downloads.pop(model_name, None)
            self._active_trackers.pop(model_name, None)

_download_manager: Optional[DownloadManager] = None
_manager_lock = threading.Lock()

def get_download_manager() -> DownloadManager:
    global _download_manager
    if _download_manager is None:
        with _manager_lock:
            if _download_manager is None:
                _download_manager = DownloadManager()
    return _download_manager