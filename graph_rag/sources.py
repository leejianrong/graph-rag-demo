"""Turn an uploaded file or a fetched URL into the text the pipeline reads (GRA-34).

``/ingest`` stores one thing per source: UTF-8 text. A PDF or an HTML page is reduced to text
here, before it is stored, so the object store, the pipeline and the ``unchanged`` check all see
the same bytes whatever the source was. A source that yields no text is an error the caller sees
(:class:`SourceError`), not an empty document.

A :class:`Fetcher` gets a URL's bytes. :class:`SafeFetcher` is the real one: it refuses addresses
that are not public (loopback, private, link-local, reserved) and checks every redirect hop the
same way, and it caps the size and the time. Tests inject a fake, so no test touches the network.
One gap remains: the name is resolved once to check it and again to connect, so a hostile DNS
server could answer differently the second time. Run the service where private networks are
unreachable if that matters.
"""

from __future__ import annotations

import io
import ipaddress
import socket
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Protocol

from pypdf import PdfReader

__all__ = ["FetchedPage", "Fetcher", "SafeFetcher", "SourceError", "extract_text"]

_MAX_REDIRECTS = 3


class SourceError(Exception):
    """A source could not be turned into text. The message is safe to show the caller."""


@dataclass(frozen=True)
class FetchedPage:
    """What a fetch returned."""

    data: bytes
    content_type: str = ""


class Fetcher(Protocol):
    """Gets the bytes at a URL, or raises :class:`SourceError`."""

    def fetch(self, url: str, max_bytes: int) -> FetchedPage: ...


class _Text(HTMLParser):
    _SKIP = {"script", "style", "head", "noscript"}
    _BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skipping = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._SKIP:
            self._skipping += 1
        elif tag in self._BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP and self._skipping:
            self._skipping -= 1
        elif tag in self._BLOCK:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skipping:
            self.parts.append(data)


def _html_text(data: bytes) -> str:
    parser = _Text()
    parser.feed(data.decode("utf-8", errors="replace"))
    lines = (" ".join(line.split()) for line in "".join(parser.parts).splitlines())
    return "\n".join(line for line in lines if line)


def _pdf_text(data: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise SourceError("The PDF is encrypted.")
        return "\n\n".join((page.extract_text() or "").strip() for page in reader.pages)
    except SourceError:
        raise
    except Exception as error:  # noqa: BLE001 — pypdf raises many types on malformed input
        raise SourceError(f"The PDF could not be read ({type(error).__name__}).") from error


def extract_text(data: bytes, *, content_type: str = "", name: str = "") -> bytes:
    """The UTF-8 text of ``data``: a PDF or HTML page is reduced to text, anything else kept.

    ``content_type`` and ``name`` (a filename or URL path) only hint; a PDF is recognised by its
    magic bytes too. Raises :class:`SourceError` when no text comes out.
    """
    kind = content_type.split(";")[0].strip().lower()
    path = urllib.parse.urlparse(name).path.lower()
    if data.startswith(b"%PDF-") or kind == "application/pdf":
        text = _pdf_text(data)
        if not text.strip():
            raise SourceError("The PDF has no extractable text (a scan, perhaps).")
    elif kind in ("text/html", "application/xhtml+xml") or (
        not kind and path.endswith((".html", ".htm"))
    ):
        text = _html_text(data)
        if not text.strip():
            raise SourceError("The page has no text.")
    else:
        return data
    return text.encode("utf-8")


def _check_public(host: str, port: int) -> None:
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError as error:
        raise SourceError("The URL's host could not be resolved.") from error
    for info in infos:
        address = ipaddress.ip_address(str(info[4][0]).split("%")[0])
        if not address.is_global:
            raise SourceError("The URL points at a non-public address, which is not fetched.")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args: object, **kwargs: object) -> None:  # type: ignore[override]
        return None


class SafeFetcher:
    """Fetch http(s) URLs from public addresses only, within a size and time cap."""

    def __init__(self, timeout: float = 10.0) -> None:
        self._timeout = timeout
        self._opener = urllib.request.build_opener(_NoRedirect)

    def fetch(self, url: str, max_bytes: int) -> FetchedPage:
        for _ in range(_MAX_REDIRECTS + 1):
            parts = urllib.parse.urlparse(url)
            if parts.scheme not in ("http", "https") or not parts.hostname:
                raise SourceError("Only http and https URLs can be fetched.")
            _check_public(parts.hostname, parts.port or (443 if parts.scheme == "https" else 80))
            request = urllib.request.Request(url, headers={"User-Agent": "graph-rag-ingest"})  # noqa: S310
            try:
                with self._opener.open(request, timeout=self._timeout) as response:  # noqa: S310
                    data = response.read(max_bytes + 1)
                    if len(data) > max_bytes:
                        raise SourceError(f"The page is over the {max_bytes}-byte limit.")
                    return FetchedPage(data, response.headers.get("content-type", ""))
            except urllib.error.HTTPError as error:
                location = error.headers.get("location")
                if error.code in (301, 302, 303, 307, 308) and location:
                    url = urllib.parse.urljoin(url, location)
                    continue
                raise SourceError(f"The URL answered {error.code}.") from error
            except (urllib.error.URLError, TimeoutError, OSError) as error:
                raise SourceError("The URL could not be fetched.") from error
        raise SourceError("The URL redirected too many times.")
