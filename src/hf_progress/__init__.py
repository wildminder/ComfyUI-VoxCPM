"""hf-progress: Plug-and-play progress tracking for HuggingFace Hub.

A library that provides real-time upload/download progress tracking
for HuggingFace Hub operations. Uses a **direct-first** strategy:

- **Xet uploads**: Direct ``hf_xet`` calls with detailed callbacks
  (dedup, transfer speed, per-file progress)
- **Xet downloads**: Direct ``hf_xet.download_files()`` with
  detailed ``(total_update, item_updates)`` callbacks for speed,
  dedup info, and per-item progress
- **HTTP downloads** (fallback): ``tqdm_class`` override for ``hf_hub_download()``
- **LFS uploads** (fallback): tqdm monkey-patching for ``HfApi.upload_file()``

Quick start::

    from hf_progress import HfProgressTracker, EventType

    tracker = HfProgressTracker(token="hf_...")

    # Upload with progress
    result = tracker.upload_file("model.bin", "username/repo")

    # Download with progress
    path = tracker.download_file("bert-base-uncased", "config.json")

    # Consume events
    for event in tracker.events(stop_on=EventType.COMPLETE):
        print(f"{event.filename}: {event.percentage:.1f}%")

Low-level callbacks::

    from hf_progress import (
        XetUploadProgressCallback,
        XetDownloadProgressCallback,
        DownloadProgressTqdm,
        tqdm_upload_patcher,
    )

SSE integration::

    from hf_progress.integrations.sse import create_progress_router
"""

from __future__ import annotations

# Core types
from .types import (
    EventType,
    ProgressEvent,
    ProgressPhase,
    TransferCancelledError,
    TransferDirection,
    TransferProgressError,
    TransferResult,
    TokenError,
    generate_transfer_id,
)

# Callback classes
from .callbacks import (
    DownloadProgressTqdm,
    XetDownloadProgressCallback,
    XetProgressCallback,
    XetUploadProgressCallback,
    tqdm_upload_patcher,
)

# Token management
from .token import XetCredentials, XetTokenManager, is_xet_available

# Subprocess isolation
from .subprocess_runner import XetSubprocessRunner
from .subprocess_messages import SubprocessMessage

# High-level tracker
from .tracker import HfProgressTracker

__version__ = "0.1.0"

__all__ = [
# Core types
"EventType",
"ProgressEvent",
"ProgressPhase",
"TransferCancelledError",
"TransferDirection",
"TransferProgressError",
"TransferResult",
"generate_transfer_id",
# Callback classes
"DownloadProgressTqdm",
"XetDownloadProgressCallback",
"XetProgressCallback",
"XetUploadProgressCallback",
"tqdm_upload_patcher",
# Token management
"XetCredentials",
"XetTokenManager",
"is_xet_available",
# Subprocess isolation
"XetSubprocessRunner",
"SubprocessMessage",
# High-level tracker
"HfProgressTracker",
]
