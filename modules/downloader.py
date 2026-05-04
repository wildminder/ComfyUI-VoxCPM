"""Download manager for VoxCPM models with real-time progress tracking and cancel support.

Wraps the ``hf_progress.HfProgressTracker`` to provide:
- Snapshot download with real-time progress events
- Cancel support via ``tracker.cancel(transfer_id)``
- Progress event forwarding to ComfyUI frontend via ``PromptServer``
- Thread-safe state management
- Token acquisition from environment or HuggingFace credentials
- Xet-specific progress fields (dedup, transfer speed)

Architecture follows the hf_progress reference pattern:
- Download runs in a daemon thread
- Main thread polls ``tracker.event_queue`` with short timeout (0.05s)
- Cancellation via ``tracker.cancel(transfer_id)``
- Events forwarded individually to frontend via ``PromptServer.send_sync()``
"""

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


def _suppress_hf_http_logs() -> None:
    """Suppress verbose HTTP request logs during model downloads.

    During snapshot downloads, multiple libraries produce noisy log output:

    - **httpx**: Logs every HTTP request at INFO level via
      ``logging.getLogger("httpx")`` (e.g. ``HTTP Request: GET https://...``).
      This is the primary source of "HTTP Request:" messages.
    - **huggingface_hub**: Uses its own custom logging system
      (``huggingface_hub.logging``) with a dedicated ``StreamHandler``.
      Must be suppressed via the library's own API.
    - **urllib3** / **filelock**: May also produce noise during downloads.

    ComfyUI typically configures the root logger at INFO or DEBUG level,
    which causes the ``httpx`` logger's INFO messages to propagate up and
    appear in the console. We suppress by setting the ``httpx`` logger level
    to WARNING (the standard Python logging approach), and also suppress
    ``huggingface_hub`` via its own API.
    """
    # Suppress httpx logger — the actual source of "HTTP Request:" messages.
    # httpx._client.py uses: logger = logging.getLogger("httpx")
    # and calls: logger.info('HTTP Request: %s %s "%s %d %s"', ...)
    _httpx_logger = logging.getLogger("httpx")
    _httpx_logger.setLevel(logging.WARNING)

    # Suppress huggingface_hub's custom logging system
    try:
        from huggingface_hub import logging as hf_logging

        hf_logging.set_verbosity_error()
        hf_logging.disable_propagation()
        _hf_logger = hf_logging.get_logger()
        for _handler in _hf_logger.handlers:
            _handler.setLevel(logging.ERROR)
    except ImportError:
        pass

    # Also suppress urllib3 and filelock standard loggers
    for _logger_name in ("urllib3.connectionpool", "filelock"):
        _lh = logging.getLogger(_logger_name)
        if _lh.level == logging.NOTSET or _lh.level < logging.WARNING:
            _lh.setLevel(logging.WARNING)


def _restore_hf_http_logs() -> None:
    """Restore loggers to their original levels after download completes.

    Pairs with :func:`_suppress_hf_http_logs` to avoid permanently
    silencing logs if the user later triggers a non-download HF operation.

    Restores the ``httpx`` logger to NOTSET (default), the
    ``huggingface_hub`` internal logging to WARNING level (the library
    default) and re-enables propagation. Also resets ``urllib3`` and
    ``filelock`` loggers.
    """
    # Restore httpx logger to default level
    logging.getLogger("httpx").setLevel(logging.NOTSET)

    # Restore huggingface_hub's custom logging
    try:
        from huggingface_hub import logging as hf_logging

        hf_logging.set_verbosity_warning()
        hf_logging.enable_propagation()
        _hf_logger = hf_logging.get_logger()
        for _handler in _hf_logger.handlers:
            _handler.setLevel(logging.NOTSET)
    except ImportError:
        pass

    # Reset standard loggers
    for _logger_name in ("urllib3.connectionpool", "filelock"):
        logging.getLogger(_logger_name).setLevel(logging.NOTSET)


def _format_bytes(num_bytes: float) -> str:
    """Format bytes into a human-readable string (e.g. '1.23 GB')."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} TB"


def _format_speed(bytes_per_sec: float) -> str:
    """Format transfer speed into a human-readable string (e.g. '12.3 MB/s')."""
    return f"{_format_bytes(bytes_per_sec)}/s"

# Polling timeout for event queue (seconds).
# Short timeout keeps the UI responsive while avoiding busy-waiting.
# Matches the hf_progress reference example pattern.
_EVENT_POLL_TIMEOUT = 0.05


# ── Exceptions ────────────────────────────────────────────────────


class DownloadCancelledError(Exception):
    """Raised when a download is cancelled by the user."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        super().__init__(f"Download cancelled for model '{model_name}'")


class DownloadError(Exception):
    """Raised when a download fails."""

    def __init__(self, model_name: str, message: str):
        self.model_name = model_name
        self.message = message
        super().__init__(f"Download failed for model '{model_name}': {message}")


# ── State Types ───────────────────────────────────────────────────


class DownloadStatus(str, Enum):
    """Status of a model download."""

    IDLE = "idle"
    DOWNLOADING = "downloading"
    COMPLETE = "complete"
    CANCELLED = "cancelled"
    ERROR = "error"


@dataclass
class DownloadState:
    """Tracks the state of an active model download.

    Attributes:
        model_name: Display name of the model.
        repo_id: HuggingFace repository ID.
        local_dir: Local directory for the download.
        status: Current download status.
        progress_percentage: Completion percentage (0.0 - 100.0).
        current_file: Name of the file currently being downloaded.
        file_index: 0-based index of the current file in multi-file transfers.
        total_files: Total number of files in the transfer.
        speed: Transfer speed in bytes/second.
        bytes_completed: Bytes processed so far.
        total_bytes: Total bytes to process.
        transfer_bytes_completed: Network bytes received (Xet only — may differ due to dedup).
        transfer_bytes_total: Total bytes scheduled for network transfer (Xet only).
        transfer_speed: Network transfer speed in bytes/second (Xet only).
        dedup_saved_bytes: Bytes saved by deduplication (Xet only).
        error_message: Error message if status is ERROR.
        is_xet: Whether the Xet backend is active for this download.
    """

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
    is_xet: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a dict suitable for JSON responses.

        Xet-specific fields are only included when they have values,
        matching the convention of ``ProgressEvent.to_dict()``.
        """
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
            "is_xet": self.is_xet,
        }
        # Include Xet-specific fields only when they have values
        if self.transfer_bytes_completed or self.transfer_bytes_total:
            d["transfer_bytes_completed"] = self.transfer_bytes_completed
            d["transfer_bytes_total"] = self.transfer_bytes_total
            d["transfer_speed"] = self.transfer_speed
        if self.dedup_saved_bytes:
            d["dedup_saved_bytes"] = self.dedup_saved_bytes
        if self.error_message is not None:
            d["error_message"] = self.error_message
        return d


# ── Download Manager ──────────────────────────────────────────────


class DownloadManager:
    """Manages model downloads with progress tracking and cancel support.

    Uses ``hf_progress.HfProgressTracker`` for actual download operations
    and forwards progress events to the ComfyUI frontend via
    ``PromptServer.send_sync()``.

    Cancellation is delegated to ``HfProgressTracker.cancel(transfer_id)``
    which sets an ``is_cancelled`` hook that the download thread checks
    on every callback invocation. The polling loop also checks
    ``tracker.is_cancelled()`` as a secondary detection mechanism.

    Thread-safe: downloads run in background threads while the main
    thread polls for progress events and forwards them to the frontend.

    Example::

        manager = DownloadManager()
        path = manager.download_model(
            model_name="VoxCPM2",
            repo_id="openbmb/VoxCPM2",
            local_dir="/models/tts/VoxCPM/VoxCPM2",
            client_id="abc123",
        )
    """

    def __init__(self):
        self._active_downloads: Dict[str, DownloadState] = {}
        self._active_trackers: Dict[str, Any] = {}  # HfProgressTracker instances
        self._lock = threading.Lock()

    # ── Public API ────────────────────────────────────────────────

    def download_model(
        self,
        model_name: str,
        repo_id: str,
        local_dir: str,
        client_id: Optional[str] = None,
    ) -> str:
        """Download a model with progress tracking.

        Blocks until download completes or is cancelled.
        Emits progress events via PromptServer.

        Args:
            model_name: Display name of the model (e.g. "VoxCPM2").
                Also used as the ``transfer_id`` for ``HfProgressTracker``.
            repo_id: HuggingFace repository ID (e.g. "openbmb/VoxCPM2").
            local_dir: Local directory to download the model to.
            client_id: Optional ComfyUI client ID for targeted WebSocket delivery.

        Returns:
            local_dir path on success.

        Raises:
            DownloadCancelledError: If the download is cancelled by the user.
            DownloadError: If the download fails for any other reason.
        """
        from src.hf_progress import HfProgressTracker, EventType, is_xet_available

        token = self._get_token()
        tracker = HfProgressTracker(token=token)

        xet_available = is_xet_available()
        logger.info(
            f"Starting download for '{model_name}' from '{repo_id}' "
            f"(Xet backend: {'available' if xet_available else 'not available'})"
        )

        with self._lock:
            self._active_trackers[model_name] = tracker
            self._active_downloads[model_name] = DownloadState(
                model_name=model_name,
                repo_id=repo_id,
                local_dir=local_dir,
                status=DownloadStatus.DOWNLOADING,
                is_xet=xet_available,
            )

        # Emit initial "start" event to frontend
        self._emit_download_event(
            model_name=model_name,
            event_type="start",
            phase="downloading",
            client_id=client_id,
            extra_data={
                "repo_id": repo_id,
                "local_dir": local_dir,
                "is_xet": xet_available,
            },
        )

        # Suppress verbose HTTP request logs from huggingface_hub
        # (e.g. "HTTP Request: GET https://huggingface.co/...")
        _suppress_hf_http_logs()

        result_path = None
        error_occurred = None

        def do_download():
            """Run the snapshot download in a background thread."""
            nonlocal result_path, error_occurred
            try:
                result_path = tracker.download_snapshot(
                    repo_id=repo_id,
                    local_dir=local_dir,
                    transfer_id=model_name,
                )
            except Exception as e:
                error_occurred = e

        # Run download in a daemon thread (matches hf_progress example pattern)
        download_thread = threading.Thread(target=do_download, daemon=True)
        download_thread.start()

        # Poll for progress events and forward to frontend.
        # Uses short timeout (0.05s) per event to stay responsive,
        # matching the hf_progress reference example pattern.
        try:
            while download_thread.is_alive() or not tracker.event_queue.empty():
                # Check cancellation via tracker's built-in mechanism
                if tracker.is_cancelled(transfer_id=model_name):
                    self._update_state(model_name, status=DownloadStatus.CANCELLED)
                    self._emit_download_event(
                        model_name=model_name,
                        event_type="error",
                        phase="error",
                        client_id=client_id,
                        extra_data={"error": "Download cancelled by user"},
                    )
                    self._print_console_progress(model_name, cancelled=True)
                    raise DownloadCancelledError(model_name)

                # Get a single event from the queue (short timeout)
                try:
                    event = tracker.event_queue.get(timeout=_EVENT_POLL_TIMEOUT)
                except queue.Empty:
                    continue

                # Update state and forward to frontend
                self._update_state_from_event(model_name, event)
                self._emit_progress_event(model_name, event, client_id)

                # Print console progress bar on PROGRESS events
                if event.event_type == EventType.PROGRESS:
                    self._print_console_progress(model_name)

                if event.event_type in (EventType.COMPLETE, EventType.ERROR):
                    if event.event_type == EventType.ERROR:
                        error_occurred = event.error or "Unknown download error"
                    break

            # Wait for the download thread to finish
            download_thread.join(timeout=5.0)

            # Drain any remaining events from the queue
            while not tracker.event_queue.empty():
                try:
                    event = tracker.event_queue.get_nowait()
                    self._update_state_from_event(model_name, event)
                    self._emit_progress_event(model_name, event, client_id)
                except queue.Empty:
                    break

            # Handle cancellation detected in the download thread
            if tracker.is_cancelled(transfer_id=model_name):
                self._update_state(model_name, status=DownloadStatus.CANCELLED)
                self._emit_download_event(
                    model_name=model_name,
                    event_type="error",
                    phase="error",
                    client_id=client_id,
                    extra_data={"error": "Download cancelled by user"},
                )
                self._print_console_progress(model_name, cancelled=True)
                raise DownloadCancelledError(model_name)

            if error_occurred:
                self._update_state(
                    model_name,
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
        finally:
            # Restore huggingface_hub loggers to their original levels
            _restore_hf_http_logs()

    def cancel_download(self, model_name: str) -> bool:
        """Request cancellation of an active download.

        Delegates to ``HfProgressTracker.cancel(transfer_id)`` which
        sets an ``is_cancelled`` hook that the download thread checks
        on every callback invocation.

        Args:
            model_name: Name of the model to cancel (also used as transfer_id).

        Returns:
            True if the download was found and cancel was requested,
            False if no active download was found for this model.
        """
        with self._lock:
            tracker = self._active_trackers.get(model_name)
            if tracker is None:
                return False

        tracker.cancel(transfer_id=model_name)
        logger.info(f"Cancel requested for model '{model_name}'")
        return True

    def get_status(self, model_name: str) -> Optional[DownloadState]:
        """Get current download status for a model.

        Args:
            model_name: Name of the model.

        Returns:
            DownloadState if the model has a download record, None otherwise.
        """
        with self._lock:
            return self._active_downloads.get(model_name)

    def get_all_statuses(self) -> Dict[str, Dict[str, Any]]:
        """Get all download statuses.

        Returns:
            Dict mapping model_name to serialized DownloadState dicts.
        """
        with self._lock:
            return {
                name: state.to_dict()
                for name, state in self._active_downloads.items()
            }

    # ── Token Acquisition ─────────────────────────────────────────

    @staticmethod
    def _get_token() -> Optional[str]:
        """Get HuggingFace token from environment or stored credentials.

        Resolution order:
        1. ``HF_TOKEN`` environment variable
        2. ``HUGGING_FACE_HUB_TOKEN`` environment variable
        3. ``HfFolder.get_token()`` (stored via ``huggingface-cli login``)
        4. ``None`` (anonymous download — public models only)

        Returns:
            Token string or None for anonymous access.
        """
        # Try environment variables first
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if token:
            return token

        # Try huggingface_hub's stored token
        try:
            from huggingface_hub import HfFolder

            token = HfFolder.get_token()
            if token:
                return token
        except Exception:
            pass

        return None  # Anonymous download (public models only)

    # ── Internal: Event Emission ──────────────────────────────────

    @staticmethod
    def _emit_download_event(
        model_name: str,
        event_type: str,
        phase: str,
        client_id: Optional[str] = None,
        extra_data: Optional[Dict[str, Any]] = None,
    ):
        """Emit a download lifecycle event to the frontend via PromptServer.

        This is used for start/complete/error events that are not
        directly from HfProgressTracker.

        Args:
            model_name: Name of the model.
            event_type: Event type string ("start", "complete", "error").
            phase: Phase string ("downloading", "complete", "error").
            client_id: Optional client ID for targeted delivery.
            extra_data: Additional data to include in the event payload.
        """
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
                PromptServer.instance.send_sync(
                    "voxcpm.download_progress", payload, client_id
                )
            else:
                PromptServer.instance.send_sync("voxcpm.download_progress", payload)

        except Exception as e:
            logger.warning(f"Failed to emit download event: {e}")

    @staticmethod
    def _emit_progress_event(
        model_name: str,
        event: Any,
        client_id: Optional[str] = None,
    ):
        """Forward a HfProgressTracker ProgressEvent to the frontend.

        Uses ``event.to_dict()`` for serialization, which converts enum
        values to their string representations. The resulting payload
        is JSON-safe and suitable for WebSocket delivery.

        Args:
            model_name: Name of the model.
            event: ProgressEvent from HfProgressTracker.
            client_id: Optional client ID for targeted delivery.
        """
        try:
            from server import PromptServer

            if PromptServer.instance is None:
                return

            payload = event.to_dict()
            # Ensure model_name is in the payload for frontend mapping
            payload["model_name"] = model_name

            if client_id:
                PromptServer.instance.send_sync(
                    "voxcpm.download_progress", payload, client_id
                )
            else:
                PromptServer.instance.send_sync("voxcpm.download_progress", payload)

        except ImportError:
            # PromptServer not available (e.g., in test environment)
            pass
        except Exception as e:
            logger.warning(f"Failed to emit progress event: {e}")

    # ── Internal: State Management ────────────────────────────────

    def _update_state(
        self,
        model_name: str,
        status: Optional[DownloadStatus] = None,
        error_message: Optional[str] = None,
    ):
        """Update the download state for a model.

        Args:
            model_name: Name of the model.
            status: New status (if changing).
            error_message: Error message (if status is ERROR).
        """
        with self._lock:
            state = self._active_downloads.get(model_name)
            if state is None:
                return
            if status is not None:
                state.status = status
            if error_message is not None:
                state.error_message = error_message

    def _update_state_from_event(self, model_name: str, event: Any):
        """Update download state from a HfProgressTracker ProgressEvent.

        Handles both standard (HTTP) and Xet progress events. Xet-specific
        fields are populated when present in the event.

        For COMPLETE events, only status and percentage are updated —
        progress fields (bytes, speed, Xet fields) are preserved from
        the last PROGRESS event to avoid resetting them to zero.

        Args:
            model_name: Name of the model.
            event: ProgressEvent from HfProgressTracker.
        """
        with self._lock:
            state = self._active_downloads.get(model_name)
            if state is None:
                return

            # Always update percentage and file info
            state.progress_percentage = event.percentage
            state.current_file = event.filename
            state.file_index = event.file_index
            state.total_files = event.total_files

            # For COMPLETE events, only update status — preserve
            # progress/Xet fields from the last PROGRESS event
            from src.hf_progress import EventType as _EventType

            if event.event_type == _EventType.COMPLETE:
                state.status = DownloadStatus.COMPLETE
                return

            # For PROGRESS/START events, update all fields
            state.speed = event.speed
            state.bytes_completed = event.bytes_completed
            state.total_bytes = event.total_bytes

            # Xet-specific fields (use getattr for safety)
            state.transfer_bytes_completed = getattr(event, "transfer_bytes_completed", 0)
            state.transfer_bytes_total = getattr(event, "transfer_bytes_total", 0)
            state.transfer_speed = getattr(event, "transfer_speed", 0.0)
            state.dedup_saved_bytes = getattr(event, "dedup_saved_bytes", 0)

            # Detect Xet from non-zero transfer fields
            if state.transfer_bytes_completed or state.transfer_bytes_total:
                state.is_xet = True

            if event.error:
                state.error_message = event.error
                state.status = DownloadStatus.ERROR

    # ── Internal: Console Progress ────────────────────────────────

    def _print_console_progress(
        self,
        model_name: str,
        complete: bool = False,
        error: bool = False,
        cancelled: bool = False,
    ) -> None:
        """Print a single-line progress bar to the console using ``\\r``.

        Overwrites the current terminal line to show real-time download
        progress without flooding the log with HTTP request messages.

        Output format::

            [VoxCPM] Downloading VoxCPM2 ━━━━━━━━━━━━━━━━━━━━ 45.2% 12.3 MB/s file 3/8
            [VoxCPM] ✓ VoxCPM2 download complete (1.23 GB)
            [VoxCPM] ✗ VoxCPM2 download failed: <error message>
            [VoxCPM] ✗ VoxCPM2 download cancelled

        Args:
            model_name: Name of the model being downloaded.
            complete: If True, print the final "complete" line with newline.
            error: If True, print the "error" line with newline.
            cancelled: If True, print the "cancelled" line with newline.
        """
        with self._lock:
            state = self._active_downloads.get(model_name)
            if state is None:
                return

        if cancelled:
            print(f"\r[VoxCPM] ✗ {model_name} download cancelled", flush=True)
            return

        if error:
            msg = state.error_message or "unknown error"
            print(f"\r[VoxCPM] ✗ {model_name} download failed: {msg}", flush=True)
            return

        if complete:
            total = _format_bytes(state.total_bytes) if state.total_bytes else "unknown size"
            print(
                f"\r[VoxCPM] ✓ {model_name} download complete ({total})",
                flush=True,
            )
            return

        # Build progress bar: 20-char bar filled with ━
        pct = state.progress_percentage
        filled = int(pct / 100 * 20)
        bar = "━" * filled + "─" * (20 - filled)

        # Speed
        speed_str = _format_speed(state.speed) if state.speed > 0 else "—"

        # File info
        file_info = ""
        if state.total_files > 1:
            file_info = f" file {state.file_index + 1}/{state.total_files}"

        # Xet badge
        xet_badge = " [xet]" if state.is_xet else ""

        line = (
            f"\r[VoxCPM] Downloading {model_name}{xet_badge} "
            f"{bar} {pct:5.1f}% {speed_str}{file_info}"
        )
        # Pad to clear any previous longer output
        print(f"{line:<80}", end="", flush=True)

    # ── Internal: Cleanup ─────────────────────────────────────────

    def _cleanup(self, model_name: str, local_dir: str):
        """Remove partially downloaded files and clear state.

        Called when a download is cancelled. Removes all files in
        ``local_dir`` but preserves the directory itself.

        Args:
            model_name: Name of the model.
            local_dir: Local directory containing partial downloads.
        """
        # Remove partial files
        if os.path.exists(local_dir):
            try:
                for item in os.listdir(local_dir):
                    item_path = os.path.join(local_dir, item)
                    if os.path.isfile(item_path):
                        os.remove(item_path)
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                logger.info(
                    f"Cleaned up partial download for '{model_name}' in {local_dir}"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to cleanup partial download for '{model_name}': {e}"
                )

        # Clear state
        with self._lock:
            self._active_downloads.pop(model_name, None)
            self._active_trackers.pop(model_name, None)


# ── Module-level singleton ────────────────────────────────────────

# Single instance shared across the application.
# Created lazily on first access to avoid import-time side effects.
_download_manager: Optional[DownloadManager] = None
_manager_lock = threading.Lock()


def get_download_manager() -> DownloadManager:
    """Get the shared DownloadManager instance.

    Thread-safe. Creates the instance on first call.

    Returns:
        The shared DownloadManager singleton.
    """
    global _download_manager
    if _download_manager is None:
        with _manager_lock:
            if _download_manager is None:
                _download_manager = DownloadManager()
    return _download_manager
