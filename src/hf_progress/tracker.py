"""Unified progress tracker for HuggingFace uploads and downloads."""

from __future__ import annotations

import asyncio
import logging
import os
import queue
import tempfile
import threading
import time
from typing import Callable, Generator, List, Optional

from .token import XetTokenManager, is_xet_available
from .types import (
    EventType,
    ProgressEvent,
    TransferCancelledError,
    generate_transfer_id,
)

logger = logging.getLogger(__name__)


class HfProgressTracker:
    def __init__(
        self,
        token: Optional[str] = None,
        endpoint: Optional[str] = None,
        report_interval: float = 0.1,
    ):
        self._token = token
        self._endpoint = endpoint
        self._report_interval = report_interval
        self.event_queue: queue.Queue[ProgressEvent] = queue.Queue(maxsize=10000)
        self._token_manager = XetTokenManager(token, endpoint)
        self._cancelled_transfers: set[str] = set()
        self._lock = threading.Lock()

    def cancel(self, transfer_id: str) -> None:
        """Cancel an active transfer by its transfer_id."""
        with self._lock:
            self._cancelled_transfers.add(transfer_id)

    def is_cancelled(self, transfer_id: str) -> bool:
        """Check if a transfer has been cancelled."""
        with self._lock:
            return transfer_id in self._cancelled_transfers

    def cleanup_transfer(self, transfer_id: str) -> None:
        """Remove a transfer from tracking sets upon completion."""
        with self._lock:
            self._cancelled_transfers.discard(transfer_id)

    def _prepare_transfer(self, transfer_id: Optional[str]) -> tuple[str, Callable[[], bool]]:
        """Generate a transfer ID and cancellation hook.

        Returns:
            Tuple of (transfer_id, is_cancelled_hook) — used by all
            public download/upload methods.
        """
        transfer_id = transfer_id or generate_transfer_id()

        def _cancelled_hook() -> bool:
            return self.is_cancelled(transfer_id)

        is_cancelled_hook = _cancelled_hook
        return transfer_id, is_cancelled_hook

    # ── Synchronous Download Methods ──────────────────────────────

    def download_file(
        self,
        repo_id: str,
        filename: str,
        repo_type: str = "model",
        revision: Optional[str] = None,
        local_dir: Optional[str] = None,
        transfer_id: Optional[str] = None,
        **kwargs,
    ) -> str:
        transfer_id, is_cancelled_hook = self._prepare_transfer(transfer_id)

        try:
            if is_xet_available():
                try:
                    return self._download_file_xet(
                        repo_id=repo_id,
                        filename=filename,
                        repo_type=repo_type,
                        revision=revision,
                        local_dir=local_dir,
                        transfer_id=transfer_id,
                        is_cancelled=is_cancelled_hook,
                    )
                except TransferCancelledError:
                    raise
                except Exception as xet_err:
                    logger.warning(
                        f"Xet direct download failed for {repo_id}/{filename}, "
                        f"falling back to tqdm_class: {xet_err}"
                    )

            from .standard_download import download_file as _download_file

            return _download_file(
                repo_id=repo_id,
                filename=filename,
                token=self._token,
                event_queue=self.event_queue,
                repo_type=repo_type,
                revision=revision,
                endpoint=self._endpoint,
                local_dir=local_dir,
                transfer_id=transfer_id,
                report_interval=self._report_interval,
                is_cancelled=is_cancelled_hook,
                **kwargs,
            )
        except KeyboardInterrupt:
            raise TransferCancelledError("Download interrupted by user (Ctrl+C)")
        finally:
            self.cleanup_transfer(transfer_id)

    def download_snapshot(
        self,
        repo_id: str,
        allow_patterns=None,
        ignore_patterns=None,
        repo_type: str = "model",
        revision: Optional[str] = None,
        local_dir: Optional[str] = None,
        transfer_id: Optional[str] = None,
        **kwargs,
    ) -> str:
        from .standard_download import download_snapshot as _download_snapshot

        transfer_id, is_cancelled_hook = self._prepare_transfer(transfer_id)

        try:
            return _download_snapshot(
                repo_id=repo_id,
                token=self._token,
                event_queue=self.event_queue,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
                repo_type=repo_type,
                revision=revision,
                endpoint=self._endpoint,
                local_dir=local_dir,
                transfer_id=transfer_id,
                report_interval=self._report_interval,
                is_cancelled=is_cancelled_hook,
                **kwargs,
            )
        except KeyboardInterrupt:
            raise TransferCancelledError("Download interrupted by user (Ctrl+C)")
        finally:
            self.cleanup_transfer(transfer_id)

    # ── Synchronous Upload Methods ────────────────────────────────

    def upload_file(
        self,
        file_path: str,
        repo_id: str,
        path_in_repo: Optional[str] = None,
        repo_type: str = "model",
        revision: Optional[str] = None,
        transfer_id: Optional[str] = None,
    ) -> str:
        transfer_id, is_cancelled_hook = self._prepare_transfer(transfer_id)
        filename = os.path.basename(file_path)
        path_in_repo = path_in_repo or filename

        try:
            if is_xet_available():
                return self._upload_file_xet(
                    file_path=file_path,
                    repo_id=repo_id,
                    path_in_repo=path_in_repo,
                    repo_type=repo_type,
                    revision=revision,
                    transfer_id=transfer_id,
                    filename=filename,
                    is_cancelled=is_cancelled_hook,
                )
            else:
                return self._upload_file_lfs(
                    file_path=file_path,
                    repo_id=repo_id,
                    path_in_repo=path_in_repo,
                    repo_type=repo_type,
                    revision=revision,
                    transfer_id=transfer_id,
                    filename=filename,
                    is_cancelled=is_cancelled_hook,
                )
        except KeyboardInterrupt:
            raise TransferCancelledError("Upload interrupted by user (Ctrl+C)")
        finally:
            self.cleanup_transfer(transfer_id)

    def upload_bytes(
        self,
        file_content: bytes,
        filename: str,
        repo_id: str,
        path_in_repo: Optional[str] = None,
        repo_type: str = "model",
        revision: Optional[str] = None,
        transfer_id: Optional[str] = None,
    ) -> str:
        transfer_id, is_cancelled_hook = self._prepare_transfer(transfer_id)
        path_in_repo = path_in_repo or filename

        try:
            if is_xet_available():
                return self._upload_bytes_xet(
                    file_content=file_content,
                    filename=filename,
                    repo_id=repo_id,
                    path_in_repo=path_in_repo,
                    repo_type=repo_type,
                    revision=revision,
                    transfer_id=transfer_id,
                    is_cancelled=is_cancelled_hook,
                )
            else:
                return self._upload_bytes_via_temp(
                    file_content=file_content,
                    filename=filename,
                    repo_id=repo_id,
                    path_in_repo=path_in_repo,
                    repo_type=repo_type,
                    revision=revision,
                    transfer_id=transfer_id,
                    is_cancelled=is_cancelled_hook,
                )
        except KeyboardInterrupt:
            raise TransferCancelledError("Upload interrupted by user (Ctrl+C)")
        finally:
            self.cleanup_transfer(transfer_id)

    def upload_folder(
        self,
        folder_path: str,
        repo_id: str,
        path_in_repo: Optional[str] = None,
        repo_type: str = "model",
        revision: Optional[str] = None,
        allow_patterns: Optional[list[str] | str] = None,
        ignore_patterns: Optional[list[str] | str] = None,
        delete_patterns: Optional[list[str] | str] = None,
        transfer_id: Optional[str] = None,
    ) -> str:
        from .standard_upload import upload_folder as _upload_folder

        transfer_id, is_cancelled_hook = self._prepare_transfer(transfer_id)

        try:
            return _upload_folder(
                folder_path=folder_path,
                repo_id=repo_id,
                token=self._token,
                event_queue=self.event_queue,
                path_in_repo=path_in_repo,
                repo_type=repo_type,
                revision=revision,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
                delete_patterns=delete_patterns,
                endpoint=self._endpoint,
                transfer_id=transfer_id,
                report_interval=self._report_interval,
                is_cancelled=is_cancelled_hook,
            )
        except KeyboardInterrupt:
            raise TransferCancelledError("Upload interrupted by user (Ctrl+C)")
        finally:
            self.cleanup_transfer(transfer_id)

    # ── Asynchronous Operations ───────────────────────────────────

    async def download_file_async(self, *args, **kwargs) -> str:
        """Asynchronous wrapper for download_file."""
        return await asyncio.to_thread(self.download_file, *args, **kwargs)

    async def download_snapshot_async(self, *args, **kwargs) -> str:
        """Asynchronous wrapper for download_snapshot."""
        return await asyncio.to_thread(self.download_snapshot, *args, **kwargs)

    async def upload_file_async(self, *args, **kwargs) -> str:
        """Asynchronous wrapper for upload_file."""
        return await asyncio.to_thread(self.upload_file, *args, **kwargs)

    async def upload_bytes_async(self, *args, **kwargs) -> str:
        """Asynchronous wrapper for upload_bytes."""
        return await asyncio.to_thread(self.upload_bytes, *args, **kwargs)

    async def upload_folder_async(self, *args, **kwargs) -> str:
        """Asynchronous wrapper for upload_folder."""
        return await asyncio.to_thread(self.upload_folder, *args, **kwargs)

    # ── Event Consumer Methods ────────────────────────────────────

    def get_events(self, timeout: float = 0) -> List[ProgressEvent]:
        events: List[ProgressEvent] = []
        while True:
            try:
                event = self.event_queue.get(timeout=timeout)
                events.append(event)
                timeout = 0
            except queue.Empty:
                break
        return events

    def events(
        self, timeout: float = 1.0, stop_on: Optional[EventType] = None
    ) -> Generator[ProgressEvent, None, None]:
        while True:
            try:
                event = self.event_queue.get(timeout=timeout)
                yield event
                if stop_on and event.event_type == stop_on:
                    return
            except queue.Empty:
                logger.debug("Queue empty in events generator — polling")
                continue

    def wait_for_complete(
        self,
        transfer_id: str,
        timeout: float = 300,
    ) -> Optional[ProgressEvent]:
        deadline = time.time() + timeout
        while time.time() < deadline:
            remaining = deadline - time.time()
            try:
                event = self.event_queue.get(timeout=min(remaining, 0.5))
                if event.transfer_id == transfer_id:
                    if event.event_type in (EventType.COMPLETE, EventType.ERROR, EventType.CANCELLED):
                        return event
            except queue.Empty:
                logger.debug("Queue empty while waiting for transfer %s — polling", transfer_id)
                continue
        return None

    # ── Internal: Xet Download Methods ──────────────────────────

    def _download_file_xet(
        self,
        repo_id: str,
        filename: str,
        repo_type: str,
        revision: Optional[str],
        local_dir: Optional[str],
        transfer_id: str,
        is_cancelled: Callable[[], bool],
    ) -> str:
        from huggingface_hub import HfApi, hf_hub_url
        from .xet_download import download_file_with_xet

        api = HfApi(endpoint=self._endpoint, token=self._token)
        url = hf_hub_url(
            repo_id=repo_id,
            filename=filename,
            repo_type=repo_type,
            revision=revision,
            endpoint=self._endpoint,
        )
        metadata = api.get_hf_file_metadata(url=url, token=self._token)

        if metadata.xet_file_data is None:
            raise ValueError(
                f"File '{filename}' in '{repo_id}' is not stored in Xet storage."
            )

        xet_file_data = metadata.xet_file_data
        file_size = metadata.size or 0

        if local_dir:
            dest_path = os.path.join(local_dir, filename)
        else:
            dest_path = os.path.join(tempfile.gettempdir(), filename)

        headers = api._build_hf_headers()
        xet_headers = {k: v for k, v in headers.items() if k != "authorization"}

        result = download_file_with_xet(
            file_hash=xet_file_data.file_hash,
            file_size=file_size,
            dest_path=dest_path,
            xet_file_data=xet_file_data,
            token=self._token,
            event_queue=self.event_queue,
            endpoint=api.endpoint,
            transfer_id=transfer_id,
            report_interval=self._report_interval,
            request_headers=xet_headers,
            is_cancelled=is_cancelled,
        )

        return result.destination_path

    # ── Internal: Xet Upload Methods ──────────────────────────────

    def _upload_file_xet(
        self,
        file_path: str,
        repo_id: str,
        path_in_repo: str,
        repo_type: str,
        revision: Optional[str],
        transfer_id: str,
        filename: str,
        is_cancelled: Callable[[], bool],
    ) -> str:
        from .xet_upload import upload_file_with_xet

        try:
            result = upload_file_with_xet(
                file_path=file_path,
                repo_id=repo_id,
                token=self._token,
                event_queue=self.event_queue,
                repo_type=repo_type,
                revision=revision,
                endpoint=self._endpoint,
                transfer_id=transfer_id,
                report_interval=self._report_interval,
                is_cancelled=is_cancelled,
            )
            return result.url or f"xet://{repo_id}/{result.filename}"
        except ImportError:
            return self._upload_file_lfs(
                file_path=file_path,
                repo_id=repo_id,
                path_in_repo=path_in_repo,
                repo_type=repo_type,
                revision=revision,
                transfer_id=transfer_id,
                filename=filename,
                is_cancelled=is_cancelled,
            )

    def _upload_bytes_xet(
        self,
        file_content: bytes,
        filename: str,
        repo_id: str,
        path_in_repo: str,
        repo_type: str,
        revision: Optional[str],
        transfer_id: str,
        is_cancelled: Callable[[], bool],
    ) -> str:
        from .xet_upload import upload_bytes_with_xet

        try:
            result = upload_bytes_with_xet(
                file_content=file_content,
                filename=filename,
                repo_id=repo_id,
                token=self._token,
                event_queue=self.event_queue,
                repo_type=repo_type,
                revision=revision,
                endpoint=self._endpoint,
                transfer_id=transfer_id,
                report_interval=self._report_interval,
                is_cancelled=is_cancelled,
            )
            return result.url or f"xet://{repo_id}/{result.filename}"
        except ImportError:
            return self._upload_bytes_via_temp(
                file_content=file_content,
                filename=filename,
                repo_id=repo_id,
                path_in_repo=path_in_repo,
                repo_type=repo_type,
                revision=revision,
                transfer_id=transfer_id,
                is_cancelled=is_cancelled,
            )

    # ── Internal: LFS Upload Methods ──────────────────────────────

    def _upload_file_lfs(
        self,
        file_path: str,
        repo_id: str,
        path_in_repo: str,
        repo_type: str,
        revision: Optional[str],
        transfer_id: str,
        filename: str,
        is_cancelled: Callable[[], bool],
    ) -> str:
        from .standard_upload import upload_file as _upload_file

        return _upload_file(
            file_path=file_path,
            repo_id=repo_id,
            token=self._token,
            event_queue=self.event_queue,
            path_in_repo=path_in_repo,
            repo_type=repo_type,
            revision=revision,
            endpoint=self._endpoint,
            transfer_id=transfer_id,
            report_interval=self._report_interval,
            is_cancelled=is_cancelled,
        )

    def _upload_bytes_via_temp(
        self,
        file_content: bytes,
        filename: str,
        repo_id: str,
        path_in_repo: str,
        repo_type: str,
        revision: Optional[str],
        transfer_id: str,
        is_cancelled: Callable[[], bool],
    ) -> str:
        from .standard_upload import upload_bytes as _upload_bytes

        return _upload_bytes(
            file_content=file_content,
            filename=filename,
            repo_id=repo_id,
            token=self._token,
            event_queue=self.event_queue,
            path_in_repo=path_in_repo,
            repo_type=repo_type,
            revision=revision,
            endpoint=self._endpoint,
            transfer_id=transfer_id,
            report_interval=self._report_interval,
            is_cancelled=is_cancelled,
        )