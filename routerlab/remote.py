"""Byte-range access to files on the HuggingFace Hub (or any HTTP server).

`Checkpoint("hf://deepseek-ai/DeepSeek-V3")` uses this to read safetensors
headers and individual router tensors without downloading the checkpoint.
Standard library only.

Authentication: ``HF_TOKEN`` (or ``HUGGING_FACE_HUB_TOKEN``), else the token
saved by ``huggingface-cli login``. The token is sent only to the Hub
endpoint, never to the CDN hosts that downloads are redirected to.
``HF_ENDPOINT`` selects a mirror.
"""

from __future__ import annotations

import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

__all__ = ["RemoteFile", "parse_hf_uri", "hf_file", "is_remote"]

_HF_URL = re.compile(r"^https?://(?:www\.)?huggingface\.co/(?P<repo>[^/]+/[^/@]+)(?:/tree/(?P<rev>[^/]+))?/?$")


def is_remote(path) -> bool:
    text = str(path)
    return text.startswith("hf://") or bool(_HF_URL.match(text))


def parse_hf_uri(uri: str) -> tuple[str, str]:
    """'hf://org/name[@revision]' or 'https://huggingface.co/org/name[/tree/rev]' -> (repo, revision)."""
    text = str(uri)
    if text.startswith("hf://"):
        repo, _, revision = text[len("hf://"):].strip("/").partition("@")
    else:
        match = _HF_URL.match(text)
        if not match:
            raise ValueError(f"Not a HuggingFace model URI: {uri!r}")
        repo, revision = match.group("repo"), match.group("rev") or ""
    if repo.count("/") != 1 or not all(repo.split("/")):
        raise ValueError(f"Expected 'org/name' in {uri!r}")
    return repo, revision or "main"


def _endpoint() -> str:
    return os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")


def _token() -> str | None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token.strip()
    home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    path = home / "token"
    return path.read_text().strip() if path.is_file() else None


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None  # handled manually so credentials never follow a cross-host redirect


_OPENER = urllib.request.build_opener(_NoRedirect)


class RemoteFile:
    """A remote file read by HTTP range requests, with retries on transient failures."""

    def __init__(self, url: str, *, token: str | None = None, timeout: float = 60.0, retries: int = 4):
        self.url = url
        self.name = url
        self._token_host = urllib.parse.urlsplit(url).netloc if token else None
        self._token = token
        self.timeout = timeout
        self.retries = retries

    def __repr__(self) -> str:
        return f"RemoteFile({self.url!r})"

    def _request(self, headers: dict[str, str]) -> tuple[int, bytes]:
        """(HTTP status, body), following redirects and retrying transient failures."""
        url, hops = self.url, 0
        for attempt in range(self.retries + 1):
            try:
                while True:
                    request_headers = {"User-Agent": "routerlab", **headers}
                    if self._token and urllib.parse.urlsplit(url).netloc == self._token_host:
                        request_headers["Authorization"] = f"Bearer {self._token}"
                    try:
                        with _OPENER.open(urllib.request.Request(url, headers=request_headers),
                                          timeout=self.timeout) as response:
                            return response.status, response.read()
                    except urllib.error.HTTPError as err:
                        if err.code in (301, 302, 303, 307, 308) and err.headers.get("Location") and hops < 10:
                            url, hops = urllib.parse.urljoin(url, err.headers["Location"]), hops + 1
                            continue
                        raise
            except urllib.error.HTTPError as err:
                if err.code == 404:
                    raise FileNotFoundError(self.url) from None
                if err.code in (401, 403):
                    raise PermissionError(f"{self.url}: access denied (HTTP {err.code}). The repo may not exist, "
                                          "or it is gated/private: set HF_TOKEN or run `huggingface-cli login`, "
                                          "and accept the model's terms on its Hub page") from None
                if err.code < 500 and err.code != 429 or attempt == self.retries:
                    raise OSError(f"{self.url}: HTTP {err.code} {err.reason}") from None
            except (urllib.error.URLError, TimeoutError, ConnectionError) as err:
                if attempt == self.retries:
                    raise OSError(f"{self.url}: {err}") from None
            time.sleep(min(2 ** attempt, 20))
            url, hops = self.url, 0
        raise AssertionError("unreachable")

    def read_upto(self, offset: int, length: int) -> bytes:
        """Up to `length` bytes at `offset` (fewer at end of file)."""
        if length == 0:
            return b""
        status, data = self._request({"Range": f"bytes={offset}-{offset + length - 1}"})
        # 206: the requested range. 200: the server ignored Range and sent the whole file.
        return data[offset:offset + length] if status == 200 else data[:length]

    def read(self, offset: int, length: int) -> bytes:
        """Exactly `length` bytes starting at `offset`."""
        data = self.read_upto(offset, length)
        if len(data) != length:
            raise OSError(f"{self.url}: expected {length} bytes at offset {offset}, got {len(data)}")
        return data

    def read_all(self) -> bytes:
        return self._request({})[1]


def hf_file(repo: str, filename: str, revision: str = "main") -> RemoteFile:
    """RemoteFile for `filename` in a Hub model repo."""
    quoted = urllib.parse.quote(revision, safe="")
    return RemoteFile(f"{_endpoint()}/{repo}/resolve/{quoted}/{filename}", token=_token())
