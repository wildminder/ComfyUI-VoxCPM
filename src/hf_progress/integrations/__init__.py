"""Integration helpers for external frameworks.

Provides optional integrations with web frameworks and real-time
streaming protocols. Each integration is an opt-in dependency
controlled by extras in pyproject.toml:

- **sse**: FastAPI Server-Sent Events endpoint for progress streaming.
  Install with ``pip install 'hf-progress[sse]'``.
"""

from __future__ import annotations