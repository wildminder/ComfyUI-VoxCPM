"""Xet authentication token management.

Wraps the HuggingFace Xet authentication flow into a reusable class
that handles both upload and download credential management.

The token lifecycle:
- **Uploads**: Use ``XetTokenType.WRITE`` to get a repo-level token
  via ``fetch_xet_connection_info_from_repo_info()``.
- **Downloads**: Use file-level ``XetFileData`` to get a per-file token
  via ``refresh_xet_connection_info()``.
- **Token refresh**: Both paths support a ``token_refresher`` callable
  that the Rust runtime calls when the token expires.
- **Caching**: ``huggingface_hub`` caches tokens internally with a
  1,000-entry limit and 60-second safety margin before expiry.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Optional, Tuple


@dataclass
class XetCredentials:
    """Xet authentication credentials for a single operation.

    Attributes:
        endpoint: Xet storage endpoint URL.
        token_info: Tuple of (access_token, expiration_unix_epoch).
        token_refresher: Optional callable that returns a fresh
            (access_token, expiration_unix_epoch) tuple.
    """

    endpoint: str
    token_info: Tuple[str, int]
    token_refresher: Optional[Callable[[], Tuple[str, int]]] = None


class XetTokenManager:
    """Manages Xet authentication tokens for uploads and downloads.

    This class wraps the complex HuggingFace Xet authentication flow
    into a simple interface. It handles:

    - Repo-level token acquisition for uploads (WRITE scope)
    - File-level token acquisition for downloads (READ scope)
    - Token refresher callable creation for Rust runtime

    Usage::

        from hf_progress.token import XetTokenManager

        manager = XetTokenManager(token="hf_...")
        creds = manager.fetch_upload_credentials("username/repo")
        # creds.endpoint, creds.token_info, creds.token_refresher

    Args:
        token: HuggingFace API token (``hf_...``).
        endpoint: Optional custom HuggingFace API endpoint.
    """

    def __init__(self, token: Optional[str] = None, endpoint: Optional[str] = None):
        self._token = token
        self._endpoint = endpoint
        self._api = None
        self._headers = None

    def _ensure_api(self):
        """Lazily initialize the HfApi instance and headers."""
        if self._api is None:
            from huggingface_hub import HfApi

            self._api = HfApi(token=self._token, endpoint=self._endpoint)
            self._headers = self._api._build_hf_headers()

    def fetch_upload_credentials(
        self,
        repo_id: str,
        repo_type: str = "model",
        revision: Optional[str] = None,
    ) -> XetCredentials:
        """Get credentials for uploading to a repo.

        Uses ``XetTokenType.WRITE`` to obtain a repo-level token
        with write access.

        Args:
            repo_id: Repository ID (e.g. ``"username/model"``).
            repo_type: Repository type (``"model"``, ``"dataset"``, ``"space"``).
            revision: Optional git revision.

        Returns:
            XetCredentials with endpoint, token_info, and token_refresher.
        """
        self._ensure_api()

        from huggingface_hub.utils._xet import (
            XetTokenType,
            fetch_xet_connection_info_from_repo_info,
        )

        connection_info = fetch_xet_connection_info_from_repo_info(
            token_type=XetTokenType.WRITE,
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            headers=self._headers,
            endpoint=self._endpoint,
        )

        return XetCredentials(
            endpoint=connection_info.endpoint,
            token_info=(
                connection_info.access_token,
                connection_info.expiration_unix_epoch,
            ),
            token_refresher=self.fetch_upload_token_refresher(
                repo_id, repo_type, revision
            ),
        )

    def fetch_upload_token_refresher(
        self,
        repo_id: str,
        repo_type: str = "model",
        revision: Optional[str] = None,
    ) -> Callable[[], Tuple[str, int]]:
        """Create a token refresher callable for uploads.

        The returned callable can be passed as ``token_refresher``
        to ``hf_xet.upload_files()`` or ``hf_xet.upload_bytes()``.
        The Rust runtime calls it when the current token expires.

        Args:
            repo_id: Repository ID.
            repo_type: Repository type.
            revision: Optional git revision.

        Returns:
            A callable that returns (access_token, expiration_unix_epoch).
        """
        from huggingface_hub.utils._xet import (
            XetTokenType,
            fetch_xet_connection_info_from_repo_info,
        )

        def token_refresher():
            self._ensure_api()
            info = fetch_xet_connection_info_from_repo_info(
                token_type=XetTokenType.WRITE,
                repo_id=repo_id,
                repo_type=repo_type,
                revision=revision,
                headers=self._headers,
                endpoint=self._endpoint,
            )
            return info.access_token, info.expiration_unix_epoch

        return token_refresher

    def fetch_download_credentials(self, xet_file_data) -> XetCredentials:
        """Get credentials for downloading a file.

        Uses file-level ``XetFileData`` to obtain a per-file token
        with read access.

        Args:
            xet_file_data: An ``XetFileData`` instance obtained from
                ``parse_xet_file_data_from_response()`` or from the
                repo's file metadata.

        Returns:
            XetCredentials with endpoint, token_info, and token_refresher.
        """
        self._ensure_api()

        from huggingface_hub.utils._xet import refresh_xet_connection_info

        connection_info = refresh_xet_connection_info(
            file_data=xet_file_data,
            headers=self._headers,
        )

        return XetCredentials(
            endpoint=connection_info.endpoint,
            token_info=(
                connection_info.access_token,
                connection_info.expiration_unix_epoch,
            ),
            token_refresher=self.fetch_download_token_refresher(xet_file_data),
        )

    def fetch_download_token_refresher(
        self, xet_file_data
    ) -> Callable[[], Tuple[str, int]]:
        """Create a token refresher callable for downloads.

        Args:
            xet_file_data: An ``XetFileData`` instance.

        Returns:
            A callable that returns (access_token, expiration_unix_epoch).
        """
        from huggingface_hub.utils._xet import refresh_xet_connection_info

        def token_refresher():
            self._ensure_api()
            info = refresh_xet_connection_info(
                file_data=xet_file_data,
                headers=self._headers,
            )
            return info.access_token, info.expiration_unix_epoch

        return token_refresher


def is_xet_available() -> bool:
    """Check if hf_xet package is installed and importable.

    Respects the HF_HUB_DISABLE_XET environment variable.

    Returns:
        True if ``hf_xet`` can be imported and is not disabled, False otherwise.
    """
    disable_xet = os.environ.get("HF_HUB_DISABLE_XET", "0").lower()
    if disable_xet in ("1", "true", "yes"):
        return False

    try:
        import hf_xet  # noqa: F401

        return True
    except ImportError:
        return False