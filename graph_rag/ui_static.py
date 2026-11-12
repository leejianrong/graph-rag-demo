"""Serve the built web UI (``ui/``, GRA-53) from the API process, on the same origin.

The UI is static files with a hash router, so one mount at ``/`` is enough, and it is added after
every API route so none of them is shadowed. The files are public: the API key lives in the
browser's memory and every API route still checks it. Only responses for UI files get the
security headers below; API responses are untouched.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from starlette.responses import Response
from starlette.staticfiles import StaticFiles
from starlette.types import Scope as AsgiScope

from graph_rag.config import Settings

__all__ = ["UI_HEADERS", "mount_ui", "resolve_ui_dir"]

# The app's own files, plus the IBM Plex stylesheet and fonts it loads from Google (app.css). Svelte
# compiles component styles to a stylesheet, so no inline style or script is needed.
UI_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self'; style-src 'self' https://fonts.googleapis.com; "
        "font-src https://fonts.gstatic.com; img-src 'self' data:; connect-src 'self'; "
        "object-src 'none'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
}


class _UiFiles(StaticFiles):
    """Static files with security headers; hashed assets are immutable, the page is revalidated."""

    async def get_response(self, path: str, scope: AsgiScope) -> Response:
        response = await super().get_response(path, scope)
        if response.status_code == 200:
            hashed = path.startswith("assets/")
            response.headers["Cache-Control"] = (
                "public, max-age=31536000, immutable" if hashed else "no-cache"
            )
        for name, value in UI_HEADERS.items():
            response.headers[name] = value
        return response


def resolve_ui_dir(settings: Settings) -> Path | None:
    """The directory holding the built UI, or ``None`` when there is none to serve."""
    if not settings.ui_dir:
        return None
    path = Path(settings.ui_dir)
    return path if (path / "index.html").is_file() else None


def mount_ui(app: FastAPI, settings: Settings) -> bool:
    """Serve the built UI at ``/`` if ``settings.ui_dir`` holds one. Call after every API route."""
    directory = resolve_ui_dir(settings)
    if directory is None:
        return False
    app.mount("/", _UiFiles(directory=directory, html=True), name="ui")
    return True
