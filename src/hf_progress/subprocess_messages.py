"""Message types for subprocess IPC.

Defines the protocol for messages traveling between the main process
and Xet worker subprocesses via ``multiprocessing.Queue``.

Message flow:
- Worker → Main: ``SubprocessMessage`` with ``msg_type`` = "event" | "result" | "error" | "cancelled"
- Payload is always a plain dict (JSON-serializable, picklable)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict


# Message type constants
MSG_EVENT = "event"
MSG_RESULT = "result"
MSG_ERROR = "error"
MSG_CANCELLED = "cancelled"


@dataclass
class SubprocessMessage:
    """A single message sent from a worker subprocess to the main process.

    Attributes:
        msg_type: One of "event", "result", "error", "cancelled".
        payload: Plain dict containing the message data.
            - "event": ``ProgressEvent.to_dict()`` output
            - "result": ``{"status": "success", ...}`` with operation-specific fields
            - "error": ``{"message": str, "error_type": str, "retryable": bool}``
            - "cancelled": ``{"message": str, "bytes_completed": int, "total_bytes": int}``
    """

    msg_type: str
    payload: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict suitable for pickling/JSON."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SubprocessMessage:
        """Deserialize from a plain dict."""
        return cls(msg_type=data["msg_type"], payload=data["payload"])

    # ── Factory Methods ──────────────────────────────────────────

    @classmethod
    def event(cls, event_dict: Dict[str, Any]) -> SubprocessMessage:
        """Create a progress event message.

        Args:
            event_dict: Output of ``ProgressEvent.to_dict()``.
        """
        return cls(msg_type=MSG_EVENT, payload=event_dict)

    @classmethod
    def result(cls, **kwargs: Any) -> SubprocessMessage:
        """Create a success result message.

        Args:
            **kwargs: Operation-specific result fields
                (e.g. ``filename``, ``destination_path``, ``file_size``).
        """
        return cls(msg_type=MSG_RESULT, payload={"status": "success", **kwargs})

    @classmethod
    def error(cls, message: str, error_type: str = "Exception", retryable: bool = False) -> SubprocessMessage:
        """Create an error message.

        Args:
            message: Human-readable error description.
            error_type: Exception class name (e.g. ``"ConnectionError"``).
            retryable: Whether the operation might succeed on retry.
        """
        return cls(
            msg_type=MSG_ERROR,
            payload={
                "status": "error",
                "message": message,
                "error_type": error_type,
                "retryable": retryable,
            },
        )

    @classmethod
    def cancelled(
        cls,
        message: str = "Transfer cancelled by user",
        bytes_completed: int = 0,
        total_bytes: int = 0,
    ) -> SubprocessMessage:
        """Create a cancellation message.

        Args:
            message: Cancellation reason.
            bytes_completed: Bytes transferred before cancellation.
            total_bytes: Total bytes expected.
        """
        return cls(
            msg_type=MSG_CANCELLED,
            payload={
                "status": "cancelled",
                "message": message,
                "bytes_completed": bytes_completed,
                "total_bytes": total_bytes,
            },
        )

    # ── Type Checks ──────────────────────────────────────────────

    @property
    def is_event(self) -> bool:
        """True if this is a progress event message."""
        return self.msg_type == MSG_EVENT

    @property
    def is_result(self) -> bool:
        """True if this is a success result message."""
        return self.msg_type == MSG_RESULT

    @property
    def is_error(self) -> bool:
        """True if this is an error message."""
        return self.msg_type == MSG_ERROR

    @property
    def is_cancelled(self) -> bool:
        """True if this is a cancellation message."""
        return self.msg_type == MSG_CANCELLED

    @property
    def is_terminal(self) -> bool:
        """True if this message signals the end of the worker's lifecycle.

        Terminal messages are: result, error, cancelled.
        Non-terminal: event (more messages may follow).
        """
        return self.msg_type in (MSG_RESULT, MSG_ERROR, MSG_CANCELLED)
