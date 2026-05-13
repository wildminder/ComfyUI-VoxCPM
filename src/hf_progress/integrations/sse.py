"""FastAPI/SSE integration for real-time progress streaming.

Provides ready-to-use FastAPI endpoints that stream progress events
from ``HfProgressTracker`` to frontend clients via Server-Sent Events.

.. warning::

   These endpoints have **no built-in authentication**. Any client that
   can reach the server can start uploads/downloads, observe progress,
   and clear transfer state. You **must** add authentication middleware
   or pass an ``auth_dependency`` to ``create_progress_router`` in
   production deployments.

Usage::

    from fastapi import FastAPI, Depends
    from hf_progress import HfProgressTracker
    from hf_progress.integrations.sse import create_progress_router

    app = FastAPI()
    tracker = HfProgressTracker(token="hf_...")

    # With authentication (recommended for production)
    async def verify_token(token: str = Depends(oauth2_scheme)):
        ...

    router = create_progress_router(tracker, auth_dependency=verify_token)
    app.include_router(router)

    # Without authentication (development only)
    router = create_progress_router(tracker)
    app.include_router(router)

The router adds these endpoints:

- ``POST /hf-progress/upload`` — Start an upload
- ``POST /hf-progress/download`` — Start a download
- ``GET /hf-progress/events/{transfer_id}`` — SSE stream for a transfer
- ``GET /hf-progress/status`` — List active transfers
- ``DELETE /hf-progress/transfer/{transfer_id}`` — Clear transfer status state
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from typing import Dict, Optional

from ..types import EventType


def create_progress_router(tracker, prefix: str = "/hf-progress", auth_dependency=None):
    """Create a FastAPI APIRouter with progress tracking endpoints.

    Args:
        tracker: An ``HfProgressTracker`` instance.
        prefix: URL prefix for all routes (default ``"/hf-progress"``).
        auth_dependency: Optional FastAPI ``Depends`` callable for
            authentication. When provided, it is applied to every
            endpoint. **Strongly recommended for production.**
    """
    try:
        from fastapi import APIRouter, BackgroundTasks, Depends
    except ImportError as err:
        raise ImportError(
            "FastAPI is required for SSE integration. "
            "Install with: pip install 'hf-progress[sse]'"
        ) from err

    router = APIRouter(prefix=prefix, tags=["progress"])

    # Apply authentication to all endpoints if an auth dependency is provided
    _common_deps = [Depends(auth_dependency)] if auth_dependency else []
    
    # Store transfer status. Keys are transfer_ids.
    _active_transfers: Dict[str, dict] = {}
    
    # Simple TTL cache cleanup logic (1 hour expiry)
    _TTL_SECONDS = 3600
    
    async def cleanup_expired_transfers():
        """Background task that sweeps old, completed transfers from memory."""
        now = time.time()
        expired = [
            tid for tid, data in _active_transfers.items()
            if data.get("status") in ("completed", "error") 
            and data.get("completed_at", now) < (now - _TTL_SECONDS)
        ]
        for tid in expired:
            _active_transfers.pop(tid, None)

    @router.post("/upload", dependencies=_common_deps)
    async def start_upload(
        repo_id: str,
        file_path: str,
        background_tasks: BackgroundTasks,
        path_in_repo: Optional[str] = None,
        repo_type: str = "model",
    ):
        """Start a file upload in a background thread."""
        transfer_id = str(uuid.uuid4())
        _active_transfers[transfer_id] = {
            "direction": "upload",
            "repo_id": repo_id,
            "filename": file_path,
            "status": "running",
            "started_at": time.time(),
        }
        
        background_tasks.add_task(cleanup_expired_transfers)

        def do_upload():
            try:
                tracker.upload_file(
                    file_path=file_path,
                    repo_id=repo_id,
                    path_in_repo=path_in_repo,
                    repo_type=repo_type,
                    transfer_id=transfer_id,
                )
                _active_transfers[transfer_id]["status"] = "completed"
            except (OSError, ConnectionError, ValueError) as e:
                _active_transfers[transfer_id]["status"] = "error"
                _active_transfers[transfer_id]["error"] = str(e)
            except Exception as e:
                _active_transfers[transfer_id]["status"] = "error"
                _active_transfers[transfer_id]["error"] = f"Unexpected error: {e}"
            finally:
                _active_transfers[transfer_id]["completed_at"] = time.time()

        thread = threading.Thread(target=do_upload, daemon=True)
        thread.start()

        return {"transfer_id": transfer_id}

    @router.post("/download", dependencies=_common_deps)
    async def start_download(
        repo_id: str,
        filename: str,
        background_tasks: BackgroundTasks,
        repo_type: str = "model",
    ):
        """Start a file download in a background thread."""
        transfer_id = str(uuid.uuid4())
        _active_transfers[transfer_id] = {
            "direction": "download",
            "repo_id": repo_id,
            "filename": filename,
            "status": "running",
            "started_at": time.time(),
        }
        
        background_tasks.add_task(cleanup_expired_transfers)

        def do_download():
            try:
                tracker.download_file(
                    repo_id=repo_id,
                    filename=filename,
                    repo_type=repo_type,
                    transfer_id=transfer_id,
                )
                _active_transfers[transfer_id]["status"] = "completed"
            except (OSError, ConnectionError, ValueError) as e:
                _active_transfers[transfer_id]["status"] = "error"
                _active_transfers[transfer_id]["error"] = str(e)
            except Exception as e:
                _active_transfers[transfer_id]["status"] = "error"
                _active_transfers[transfer_id]["error"] = f"Unexpected error: {e}"
            finally:
                _active_transfers[transfer_id]["completed_at"] = time.time()

        thread = threading.Thread(target=do_download, daemon=True)
        thread.start()

        return {"transfer_id": transfer_id}

    @router.get("/events/{transfer_id}", dependencies=_common_deps)
    async def stream_progress(transfer_id: str):
        """Stream progress events for a transfer via SSE."""
        from sse_starlette.sse import EventSourceResponse

        async def event_stream():
            while True:
                events = tracker.get_events()
                for event in events:
                    if event.transfer_id == transfer_id:
                        data = json.dumps(event.to_dict())
                        yield {"data": data}
                        if event.event_type in (
                            EventType.COMPLETE,
                            EventType.ERROR,
                            EventType.CANCELLED,
                        ):
                            return
                await asyncio.sleep(0.1)

        return EventSourceResponse(event_stream())

    @router.get("/status", dependencies=_common_deps)
    async def get_status(background_tasks: BackgroundTasks):
        """List all active and recent transfers."""
        background_tasks.add_task(cleanup_expired_transfers)
        return {
            "active_transfers": _active_transfers,
            "queue_size": tracker.event_queue.qsize(),
        }

    @router.delete("/transfer/{transfer_id}", dependencies=_common_deps)
    async def clear_transfer(transfer_id: str):
        """Explicitly clear a transfer from memory."""
        if transfer_id in _active_transfers:
            _active_transfers.pop(transfer_id)
            return {"status": "success", "message": "Transfer state cleared."}
        return {"status": "not_found", "message": "Transfer ID not found."}

    return router


def create_raw_sse_stream(tracker, transfer_id: str):
    """Create a raw SSE event generator without FastAPI dependency."""
    import asyncio

    async def event_stream():
        while True:
            events = tracker.get_events()
            for event in events:
                if event.transfer_id == transfer_id:
                    data = json.dumps(event.to_dict())
                    yield f"data: {data}\n\n"
                    if event.event_type in (
                        EventType.COMPLETE,
                        EventType.ERROR,
                        EventType.CANCELLED,
                    ):
                        return
            await asyncio.sleep(0.1)

    return event_stream()