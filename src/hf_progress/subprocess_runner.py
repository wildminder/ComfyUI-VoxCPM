"""Subprocess runner for isolating hf_xet operations.

Manages the lifecycle of a child process that runs an Xet worker function.
Provides:

- ``start()``: Spawn a child process with ``multiprocessing.get_context("spawn")``
- ``terminate()``: Kill the child process (SIGTERM → SIGKILL fallback)
- ``wait()``: Block until the worker sends a terminal message
- Event relay: Background thread translates ``mp.Queue`` messages
  into ``queue.Queue[ProgressEvent]`` for the main process

Key design choices:

- **``spawn`` context only**: Avoids inheriting parent state/locks/CUDA
- **``daemon=True``**: Child dies if main process crashes
- **Relay thread**: Bridges ``mp.Queue`` → ``queue.Queue`` so the
  main process keeps its existing ``ProgressEvent`` API
- **``mp.Event`` cancel signal**: Cross-process cancellation that
  the worker's callback checks on each invocation
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import queue
import threading
import time
import warnings
from typing import Any, Callable, Dict, Optional

from .subprocess_messages import MSG_EVENT, SubprocessMessage
from .types import EventType, ProgressEvent, TransferError

logger = logging.getLogger(__name__)


class XetSubprocessRunner:
    """Manages an isolated subprocess for Xet operations.

    Usage::

        runner = XetSubprocessRunner()
        runner.start(
            worker_func=_download_worker,
            params={"file_hash": "abc", ...},
            event_queue=my_queue,
        )
        # ... events flow into my_queue via relay thread ...
        result = runner.wait(timeout=300)
        runner.terminate()  # always call to clean up

    Args:
        terminate_timeout: Seconds to wait after SIGTERM before SIGKILL (default 3).
        kill_timeout: Seconds to wait after SIGKILL before giving up (default 2).
    """

    def __init__(
        self,
        terminate_timeout: float = 3.0,
        kill_timeout: float = 2.0,
    ):
        self._ctx = mp.get_context("spawn")
        self._process: Optional[mp.Process] = None
        self._mp_queue: Optional[mp.Queue] = None
        self._cancel_event: Optional[mp.Event] = None
        self._relay_thread: Optional[threading.Thread] = None
        self._event_queue: Optional[queue.Queue] = None
        self._stop_event = threading.Event()
        self._result: Optional[Dict[str, Any]] = None
        self._terminate_timeout = terminate_timeout
        self._kill_timeout = kill_timeout
        self._lock = threading.Lock()

    def start(
        self,
        worker_func: Callable,
        params: Dict[str, Any],
        event_queue: queue.Queue,
    ) -> None:
        """Spawn child process and start event relay thread.

        Args:
            worker_func: A top-level function that runs in the child process.
                Must accept ``(params: dict, mp_queue: mp.Queue, cancel_event: mp.Event)``.
            params: Picklable dict of parameters for the worker.
            event_queue: Main-process queue to receive ``ProgressEvent`` objects.

        Raises:
            RuntimeError: If a process is already running.
        """
        with self._lock:
            if self._process is not None and self._process.is_alive():
                raise RuntimeError("A subprocess is already running. Call terminate() first.")

            self._event_queue = event_queue
            self._stop_event.clear()
            self._result = None

            # Create multiprocessing objects in the spawn context
            self._mp_queue = self._ctx.Queue()
            self._cancel_event = self._ctx.Event()

            # Spawn child process
            self._process = self._ctx.Process(
                target=worker_func,
                args=(params, self._mp_queue, self._cancel_event),
                daemon=True,
            )
            self._process.start()

            # Start relay thread
            self._relay_thread = threading.Thread(
                target=self._relay_events,
                name=f"xet-relay-{self._process.pid}",
                daemon=True,
            )
            self._relay_thread.start()

            logger.debug(
                "Subprocess started: pid=%d, worker=%s",
                self._process.pid,
                getattr(worker_func, "__name__", str(worker_func)),
            )

    def _relay_events(self) -> None:
        """Background thread: mp.Queue → queue.Queue translation.

        Reads ``SubprocessMessage`` from the multiprocessing queue,
        converts to ``ProgressEvent``, and puts into the main event queue.
        Stops when:
        - A terminal message (result/error/cancelled) is received
        - The stop_event is set (from terminate())
        - The mp_queue is empty for too long after the process exits
        """
        while not self._stop_event.is_set():
            try:
                msg = self._mp_queue.get(timeout=0.2)  # type: ignore[union-attr]
            except Exception:
                # Queue empty or closed — check if process is still alive
                if self._process is not None and not self._process.is_alive():
                    # Drain any remaining messages before exiting
                    self._drain_queue()
                    break
                continue

            if msg.is_event:
                # Translate SubprocessMessage → ProgressEvent
                try:
                    event = ProgressEvent.from_dict(msg.payload)
                    if self._event_queue is not None:
                        try:
                            self._event_queue.put_nowait(event)
                        except queue.Full:
                            logger.warning(
                                "Event queue full — dropping %s event for %s",
                                event.event_type.value,
                                event.filename,
                            )
                except Exception as e:
                    logger.warning("Failed to translate event message: %s", e)

            elif msg.is_result:
                self._result = msg.payload
                # Emit COMPLETE event from the result
                self._emit_complete_from_result(msg.payload)
                break

            elif msg.is_error:
                self._result = msg.payload
                # Emit ERROR event
                self._emit_error_from_payload(msg.payload)
                break

            elif msg.is_cancelled:
                self._result = msg.payload
                # Emit CANCELLED event
                self._emit_cancelled_from_payload(msg.payload)
                break

    def _drain_queue(self) -> None:
        """Drain any remaining messages from mp_queue after process exits."""
        while True:
            try:
                msg = self._mp_queue.get_nowait()  # type: ignore[union-attr]
                if msg.is_event:
                    try:
                        event = ProgressEvent.from_dict(msg.payload)
                        if self._event_queue is not None:
                            self._event_queue.put_nowait(event)
                    except Exception:
                        pass
                elif msg.is_terminal:
                    self._result = msg.payload
                    if msg.is_result:
                        self._emit_complete_from_result(msg.payload)
                    elif msg.is_error:
                        self._emit_error_from_payload(msg.payload)
                    elif msg.is_cancelled:
                        self._emit_cancelled_from_payload(msg.payload)
                    break
            except Exception:
                break

    def _emit_complete_from_result(self, payload: Dict[str, Any]) -> None:
        """Emit a COMPLETE ProgressEvent from a result payload."""
        if self._event_queue is None:
            return
        try:
            from .types import ProgressPhase, TransferDirection
            event = ProgressEvent(
                event_type=EventType.COMPLETE,
                transfer_id=payload.get("transfer_id", ""),
                direction=TransferDirection(payload.get("direction", "download")),
                filename=payload.get("filename", ""),
                phase=ProgressPhase.COMPLETE,
                bytes_completed=payload.get("file_size", 0),
                total_bytes=payload.get("file_size", 0),
                percentage=100.0,
            )
            self._event_queue.put(event)
        except Exception as e:
            logger.warning("Failed to emit COMPLETE event: %s", e)

    def _emit_error_from_payload(self, payload: Dict[str, Any]) -> None:
        """Emit an ERROR ProgressEvent from an error payload."""
        if self._event_queue is None:
            return
        try:
            from .types import ProgressPhase, TransferDirection
            event = ProgressEvent(
                event_type=EventType.ERROR,
                transfer_id=payload.get("transfer_id", ""),
                direction=TransferDirection(payload.get("direction", "download")),
                filename=payload.get("filename", ""),
                phase=ProgressPhase.ERROR,
                error=TransferError(
                    message=payload.get("message", "Unknown error"),
                    error_type=payload.get("error_type", "Exception"),
                    retryable=payload.get("retryable", False),
                ),
            )
            self._event_queue.put(event)
        except Exception as e:
            logger.warning("Failed to emit ERROR event: %s", e)

    def _emit_cancelled_from_payload(self, payload: Dict[str, Any]) -> None:
        """Emit a CANCELLED ProgressEvent from a cancelled payload."""
        if self._event_queue is None:
            return
        try:
            event = ProgressEvent.cancelled_event(
                transfer_id=payload.get("transfer_id", ""),
                direction=payload.get("direction", "download"),
                filename=payload.get("filename", ""),
                bytes_completed=payload.get("bytes_completed", 0),
                total_bytes=payload.get("total_bytes", 0),
            )
            self._event_queue.put(event)
        except Exception as e:
            logger.warning("Failed to emit CANCELLED event: %s", e)

    def terminate(self) -> None:
        """Terminate the child process: SIGTERM → join → SIGKILL → join.

        Always safe to call — no-op if no process is running.
        Sets the cancel_event before terminating so the worker can
        clean up gracefully if it checks the event.
        """
        with self._lock:
            # Signal cancellation first
            if self._cancel_event is not None:
                try:
                    self._cancel_event.set()
                except Exception:
                    pass

            # Stop the relay thread
            self._stop_event.set()

            if self._process is not None:
                if self._process.is_alive():
                    logger.debug("Terminating subprocess pid=%d", self._process.pid)
                    self._process.terminate()
                    self._process.join(timeout=self._terminate_timeout)

                    if self._process.is_alive():
                        logger.warning(
                            "Subprocess pid=%d did not terminate in %.1fs — killing",
                            self._process.pid,
                            self._terminate_timeout,
                        )
                        self._process.kill()
                        self._process.join(timeout=self._kill_timeout)

                    if self._process.is_alive():
                        logger.error(
                            "Subprocess pid=%d could not be killed!",
                            self._process.pid,
                        )

                # Wait for relay thread to finish
                if self._relay_thread is not None and self._relay_thread.is_alive():
                    self._relay_thread.join(timeout=2.0)

                # Clean up process reference
                self._process = None

            # Clean up mp objects
            self._mp_queue = None
            self._cancel_event = None
            self._relay_thread = None

    def is_alive(self) -> bool:
        """Check if the child process is still running.

        Returns:
            True if the process exists and is alive, False otherwise.
        """
        with self._lock:
            return self._process is not None and self._process.is_alive()

    def wait(self, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Wait for the worker to complete and return the result.

        Blocks until the relay thread receives a terminal message
        (result/error/cancelled) or the timeout expires.

        Args:
            timeout: Maximum seconds to wait. None means wait forever.

        Returns:
            Result payload dict on success/error/cancelled, or None on timeout.
        """
        if self._relay_thread is None:
            return self._result

        self._relay_thread.join(timeout=timeout)

        if self._relay_thread.is_alive():
            # Timeout expired
            return None

        return self._result

    @property
    def pid(self) -> Optional[int]:
        """Process ID of the child, or None if not started."""
        with self._lock:
            if self._process is not None:
                return self._process.pid
            return None

    @property
    def exitcode(self) -> Optional[int]:
        """Exit code of the child process, or None if still running."""
        with self._lock:
            if self._process is not None:
                return self._process.exitcode
            return None

    def __del__(self) -> None:
        """Warn if process wasn't properly joined."""
        if self._process is not None and self._process.is_alive():
            warnings.warn(
                f"XetSubprocessRunner with pid={self._process.pid} was not "
                f"properly terminated. Call terminate() to avoid zombie processes.",
                ResourceWarning,
                stacklevel=1,
            )
