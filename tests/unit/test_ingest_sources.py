"""Plain text, PDF and URL ingest sources (GRA-34). No test touches the network."""

from __future__ import annotations

import email.message
import socket
import urllib.error

import pytest
from fastapi.testclient import TestClient

from graph_rag.api import create_app
from graph_rag.config import Settings
from graph_rag.sources import FetchedPage, SafeFetcher, SourceError, extract_text
from graph_rag.wiring import build_offline_components
from tests.unit.test_agent_api_v1 import Queue, drain

TEXT = "Ada Lovelace works for Acme Corp. Acme Corp is based in London."


def make_pdf(text: str) -> bytes:
    """A one-page PDF with ``text`` on it."""
    stream = f"BT /F1 12 Tf 20 100 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 600 200] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    body = b"%PDF-1.4\n"
    offsets = []
    for number, obj in enumerate(objects, 1):
        offsets.append(len(body))
        body += b"%d 0 obj\n" % number + obj + b"\nendobj\n"
    xref = len(body)
    body += b"xref\n0 6\n0000000000 65535 f \n"
    body += b"".join(b"%010d 00000 n \n" % offset for offset in offsets)
    return body + b"trailer\n<< /Root 1 0 R /Size 6 >>\nstartxref\n%d\n%%%%EOF\n" % xref


class FakeFetcher:
    def __init__(self, page: FetchedPage | None = None, error: str | None = None) -> None:
        self.page, self.error, self.urls = page, error, []

    def fetch(self, url: str, max_bytes: int) -> FetchedPage:
        self.urls.append(url)
        if self.error:
            raise SourceError(self.error)
        assert self.page is not None
        return self.page


def stack(fetcher: FakeFetcher | None = None):
    components = build_offline_components()
    queue = Queue()
    app = create_app(
        components.object_store,
        queue,
        Settings(),
        retriever=components.retriever,
        jobs=components.job_store,
        fetcher=fetcher,
    )
    return TestClient(app), components, queue


def test_a_pdf_upload_is_stored_as_text_and_ingests() -> None:
    client, components, queue = stack()
    pdf = make_pdf("Ada Lovelace works for Acme Corp.")
    response = client.post("/ingest", files={"file": ("ada.pdf", pdf, "application/pdf")})
    assert response.status_code == 200
    drain(components, queue)
    job = client.get(f"/jobs/{response.json()['document_id']}").json()
    assert job["state"] == "done"
    stored = components.object_store.get(_default(), response.json()["bucket"], "ada.pdf")
    assert stored == b"Ada Lovelace works for Acme Corp."


def _default():
    from graph_rag.scope import DEFAULT_SCOPE

    return DEFAULT_SCOPE


def test_a_url_is_fetched_keyed_by_the_url_and_ingests() -> None:
    html = b"<html><head><title>x</title><script>var a=1</script></head><body><p>" + (
        TEXT.encode() + b"</p></body></html>"
    )
    fetcher = FakeFetcher(FetchedPage(html, "text/html; charset=utf-8"))
    client, components, queue = stack(fetcher)
    url = "https://example.com/ada"
    response = client.post("/ingest", data={"url": url})
    assert response.status_code == 200 and response.json()["object_key"] == url
    assert fetcher.urls == [url]
    stored = components.object_store.get(_default(), response.json()["bucket"], url)
    assert stored == TEXT.encode()
    drain(components, queue)
    assert client.get(f"/jobs/{response.json()['document_id']}").json()["state"] == "done"


def test_a_fetched_pdf_is_extracted() -> None:
    fetcher = FakeFetcher(FetchedPage(make_pdf("Hello world"), "application/pdf"))
    client, components, _ = stack(fetcher)
    response = client.post("/ingest", data={"url": "https://example.com/a.pdf"})
    stored = components.object_store.get(
        _default(), response.json()["bucket"], response.json()["object_key"]
    )
    assert stored == b"Hello world"


def test_a_plain_text_upload_is_unchanged() -> None:
    client, components, _ = stack()
    response = client.post("/ingest", files={"file": ("n.txt", TEXT.encode(), "text/plain")})
    stored = components.object_store.get(_default(), response.json()["bucket"], "n.txt")
    assert stored == TEXT.encode()


@pytest.mark.parametrize(
    ("files", "data"),
    [
        ({"file": ("bad.pdf", b"%PDF-1.4 not really", "application/pdf")}, {}),
        ({"file": ("blank.pdf", make_pdf(""), "application/pdf")}, {}),
        ({"file": ("e.html", b"<html><script>x</script></html>", "text/html")}, {}),
    ],
)
def test_an_extraction_failure_is_a_422_and_nothing_is_stored(files, data) -> None:
    client, components, queue = stack()
    response = client.post("/ingest", files=files, data=data)
    assert response.status_code == 422 and response.json()["detail"]
    assert components.object_store.keys(_default()) == [] and queue.triggers == []


def test_a_failed_fetch_is_a_422_and_a_missing_source_is_a_400() -> None:
    client, _, queue = stack(FakeFetcher(error="The URL answered 404."))
    response = client.post("/ingest", data={"url": "https://example.com/x"})
    assert response.status_code == 422 and "404" in response.json()["detail"]
    assert client.post("/ingest").status_code == 400
    both = client.post(
        "/ingest", data={"url": "https://example.com/x"}, files={"file": ("a.txt", b"a")}
    )
    assert both.status_code == 400 and queue.triggers == []


def test_extract_text_keeps_other_bytes_as_they_are() -> None:
    assert extract_text(b"plain", content_type="text/plain") == b"plain"
    assert extract_text(b"{}", name="a.json") == b"{}"


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/",
        "http://10.0.0.5/x",
        "http://169.254.169.254/latest/meta-data",
        "http://[::1]/",
        "http://192.168.1.1/",
        "file:///etc/passwd",
        "ftp://example.com/x",
    ],
)
def test_the_real_fetcher_refuses_non_public_targets(url: str) -> None:
    with pytest.raises(SourceError):
        SafeFetcher().fetch(url, 1000)


def test_the_real_fetcher_refuses_a_redirect_to_a_private_address(monkeypatch) -> None:
    def resolve(host, port, **kw):
        ip = "93.184.216.34" if host == "example.com" else "169.254.169.254"
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port))]

    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    fetcher = SafeFetcher()
    headers = email.message.Message()
    headers["location"] = "http://metadata.internal/latest"
    calls: list[str] = []

    def opener(request, timeout):
        calls.append(request.full_url)
        raise urllib.error.HTTPError(request.full_url, 302, "Found", headers, None)

    monkeypatch.setattr(fetcher._opener, "open", opener)  # noqa: SLF001
    with pytest.raises(SourceError, match="non-public"):
        fetcher.fetch("http://example.com/start", 1000)
    assert calls == ["http://example.com/start"]  # the private hop was never requested


def test_a_url_ingest_is_refused_for_a_tenant_with_no_llm_key() -> None:
    from cryptography.fernet import Fernet

    from graph_rag.auth import ApiKeys, InMemoryKeyStore
    from graph_rag.tenant_keys import InMemoryTenantKeyStore, TenantLlmKeys

    components = build_offline_components()
    queue = Queue()
    api_keys = ApiKeys(InMemoryKeyStore())
    llm_keys = TenantLlmKeys(InMemoryTenantKeyStore(), Fernet.generate_key().decode())
    fetcher = FakeFetcher(FetchedPage(TEXT.encode(), "text/plain"))
    client = TestClient(
        create_app(
            components.object_store,
            queue,
            Settings(require_auth=True, require_tenant_llm_key=True),
            keys=api_keys,
            llm_keys=llm_keys,
            fetcher=fetcher,
        )
    )
    token, _ = api_keys.create("globex")
    auth = {"Authorization": f"Bearer {token}"}
    refused = client.post("/ingest", data={"url": "https://example.com/a"}, headers=auth)
    assert refused.status_code == 400 and "no LLM provider key" in refused.json()["detail"]
    assert fetcher.urls == [] and queue.triggers == []
    llm_keys.set("globex", "globex-provider-key-sentinel")
    ok = client.post("/ingest", data={"url": "https://example.com/a"}, headers=auth)
    assert ok.status_code == 200 and len(queue.triggers) == 1
