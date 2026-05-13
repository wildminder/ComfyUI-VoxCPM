"""Data types for HuggingFace progress tracking.

Provides structured, typed representations of progress events
that can be consumed by any external application.
"""

from __future__ import annotations

import enum
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


class ProgressPhase(str, enum.Enum):
    """Phase of a transfer operation."""

    HASHING = "hashing"
    UPLOADING = "uploading"
    DOWNLOADING = "downloading"
    VERIFYING = "verifying"
    COMPLETE = "complete"
    ERROR = "error"


class TransferDirection(str, enum.Enum):
    """Direction of a transfer operation."""

    UPLOAD = "upload"
    DOWNLOAD = "download"


class EventType(str, enum.Enum):
    """Type of progress event."""

    START = "start"
    PROGRESS = "progress"
    COMPLETE = "complete"
    ERROR = "error"
    CANCELLED = "cancelled"


@dataclass
class TransferError:
    """Structured error information for failed transfers."""
    message: str
    error_type: str = "Exception"
    retryable: bool = False

    def __str__(self) -> str:
        return self.message

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message": self.message,
            "error_type": self.error_type,
            "retryable": self.retryable,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TransferError:
        return cls(
            message=data.get("message", "Unknown error"),
            error_type=data.get("error_type", "Exception"),
            retryable=data.get("retryable", False),
        )


@dataclass
class ProgressEvent:
    """Structured progress event emitted during transfers.

    All events are serializable to dict/JSON for SSE, WebSocket,
    or any message queue integration.

    Attributes:
        event_type: Type of the event (start, progress, complete, error).
        transfer_id: Unique identifier for the transfer operation.
        direction: Upload or download.
        filename: Name of the file being transferred.
        phase: Current phase of the transfer.
        bytes_completed: Bytes processed so far.
        total_bytes: Total bytes to process.
        percentage: Completion percentage (0.0 - 100.0).
        speed: Transfer speed in bytes/second.
        file_index: Index of this file in a multi-file transfer (0-based).
        total_files: Total number of files in the transfer.
        transfer_bytes_completed: Bytes actually transferred over network
            (Xet only -- may differ from bytes_completed due to dedup).
        transfer_bytes_total: Total bytes scheduled for network transfer
            (Xet only).
        transfer_speed: Network transfer speed in bytes/second (Xet only).
        dedup_saved_bytes: Bytes saved by deduplication (Xet upload only).
        error: Structured error object (only set when event_type is "error").
        timestamp: Unix timestamp when the event was created.
        extra: Additional metadata (strategy-specific fields).
    """

    event_type: EventType
    transfer_id: str
    direction: TransferDirection
    filename: str
    phase: ProgressPhase
    bytes_completed: int = 0
    total_bytes: int = 0
    percentage: float = 0.0
    speed: float = 0.0
    file_index: int = 0
    total_files: int = 1
    transfer_bytes_completed: int = 0
    transfer_bytes_total: int = 0
    transfer_speed: float = 0.0
    dedup_saved_bytes: int = 0
    error: Optional[TransferError] = None
    timestamp: float = field(default_factory=time.time)
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def start(
        cls,
        transfer_id: str,
        direction: TransferDirection,
        filename: str,
        phase: ProgressPhase,
        total_bytes: int = 0,
        **kwargs,
    ) -> ProgressEvent:
        """Create a START event with sensible defaults."""
        return cls(
            event_type=EventType.START,
            transfer_id=transfer_id,
            direction=direction,
            filename=filename,
            phase=phase,
            total_bytes=total_bytes,
            **kwargs,
        )

    @classmethod
    def complete(
        cls,
        transfer_id: str,
        direction: TransferDirection,
        filename: str,
        phase: ProgressPhase = ProgressPhase.COMPLETE,
        bytes_completed: int = 0,
        total_bytes: int = 0,
        **kwargs,
    ) -> ProgressEvent:
        """Create a COMPLETE event with percentage auto-set to 100."""
        return cls(
            event_type=EventType.COMPLETE,
            transfer_id=transfer_id,
            direction=direction,
            filename=filename,
            phase=phase,
            bytes_completed=bytes_completed,
            total_bytes=total_bytes,
            percentage=100.0,
            **kwargs,
        )

    @classmethod
    def error_event(
        cls,
        transfer_id: str,
        direction: TransferDirection,
        filename: str,
        error: TransferError,
        phase: ProgressPhase = ProgressPhase.ERROR,
        **kwargs,
    ) -> ProgressEvent:
        """Create an ERROR event."""
        return cls(
            event_type=EventType.ERROR,
            transfer_id=transfer_id,
            direction=direction,
            filename=filename,
            phase=phase,
            error=error,
            **kwargs,
        )

    @classmethod
    def cancelled_event(
        cls,
        transfer_id: str,
        direction: TransferDirection,
        filename: str,
        bytes_completed: int = 0,
        total_bytes: int = 0,
        **kwargs,
    ) -> ProgressEvent:
        """Create a CANCELLED event."""
        return cls(
            event_type=EventType.CANCELLED,
            transfer_id=transfer_id,
            direction=direction,
            filename=filename,
            phase=ProgressPhase.ERROR,
            bytes_completed=bytes_completed,
            total_bytes=total_bytes,
            error=TransferError(message="Transfer cancelled by user", error_type="TransferCancelledError"),
            **kwargs,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict suitable for JSON encoding."""
        d = {
            "event_type": self.event_type.value,
            "transfer_id": self.transfer_id,
            "direction": self.direction.value,
            "filename": self.filename,
            "phase": self.phase.value,
            "bytes_completed": self.bytes_completed,
            "total_bytes": self.total_bytes,
            "percentage": round(self.percentage, 2),
            "speed": self.speed,
            "file_index": self.file_index,
            "total_files": self.total_files,
            "timestamp": self.timestamp,
        }
        # Include Xet-specific fields only when they have values
        if self.transfer_bytes_completed or self.transfer_bytes_total:
            d["transfer_bytes_completed"] = self.transfer_bytes_completed
            d["transfer_bytes_total"] = self.transfer_bytes_total
            d["transfer_speed"] = self.transfer_speed
        if self.dedup_saved_bytes:
            d["dedup_saved_bytes"] = self.dedup_saved_bytes
        if self.error is not None:
            d["error"] = self.error.to_dict()
        if self.extra:
            d["extra"] = self.extra
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ProgressEvent:
        """Deserialize from a plain dict."""
        # Infer default phase from direction if phase is missing
        direction = TransferDirection(data["direction"])
        default_phase = ProgressPhase.UPLOADING if direction == TransferDirection.UPLOAD else ProgressPhase.DOWNLOADING
        
        error_obj = None
        if "error" in data:
            err_data = data["error"]
            if isinstance(err_data, str):
                # Backward compatibility for old JSON payloads
                error_obj = TransferError(message=err_data)
            else:
                error_obj = TransferError.from_dict(err_data)

        return cls(
            event_type=EventType(data["event_type"]),
            transfer_id=data["transfer_id"],
            direction=direction,
            filename=data["filename"],
            phase=ProgressPhase(data.get("phase", default_phase.value)),
            bytes_completed=data.get("bytes_completed", 0),
            total_bytes=data.get("total_bytes", 0),
            percentage=data.get("percentage", 0.0),
            speed=data.get("speed", 0.0),
            file_index=data.get("file_index", 0),
            total_files=data.get("total_files", 1),
            transfer_bytes_completed=data.get("transfer_bytes_completed", 0),
            transfer_bytes_total=data.get("transfer_bytes_total", 0),
            transfer_speed=data.get("transfer_speed", 0.0),
            dedup_saved_bytes=data.get("dedup_saved_bytes", 0),
            error=error_obj,
            timestamp=data.get("timestamp", time.time()),
            extra=data.get("extra", {}),
        )


@dataclass
class TransferResult:
    """Result of a completed transfer operation.

    Attributes:
        success: Whether the transfer completed successfully.
        transfer_id: Unique identifier for the transfer.
        filename: Name of the transferred file.
        url: URL of the uploaded/downloaded resource (if available).
        hash: Content hash of the file (Xet uploads only).
        file_size: Size of the file in bytes.
        direction: Upload or download.
        local_path: Local file path (for downloads).
    """

    success: bool
    transfer_id: str
    filename: str
    url: Optional[str] = None
    hash: Optional[str] = None
    file_size: int = 0
    direction: TransferDirection = TransferDirection.UPLOAD
    local_path: Optional[str] = None


def generate_transfer_id() -> str:
    """Generate a unique transfer ID."""
    return str(uuid.uuid4())


class TransferCancelledError(Exception):
    """Raised when a transfer is cancelled by the user.

    Callers can catch this specific exception to distinguish
    user-initiated cancellation from other runtime errors.
    """


class TransferProgressError(Exception):
    """Raised when a transfer operation encounters a recoverable error."""


class TokenError(Exception):
    """Raised when token resolution or authentication fails."""