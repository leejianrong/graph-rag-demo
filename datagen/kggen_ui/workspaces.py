"""Saved workspaces for the generator UI (GRA-194): a name and the settings that made each dataset.

Generation is deterministic, so a workspace stores only its settings (every select and number on
the form), one revision per change, never the generated data: opening one regenerates it from the
latest revision. Plain JSON files, one directory per workspace, so they can be read, copied and put
in git. This module imports nothing from indah.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

MAX_REVISIONS = 50  # the oldest are dropped past this
ENV_DIR = "KGGEN_STUDIO_DIR"


def default_root() -> Path:
    """Where workspaces live: ``$KGGEN_STUDIO_DIR``, else ``~/.kggen/studio``."""
    return Path(os.environ.get(ENV_DIR) or Path.home() / ".kggen" / "studio")


@dataclass(frozen=True)
class Revision:
    """One saved set of settings."""

    id: int
    saved: str  # UTC, "YYYY-MM-DD HH:MM:SS"
    settings: dict

    def to_dict(self) -> dict:
        return {"id": self.id, "saved": self.saved, "settings": self.settings}


@dataclass(frozen=True)
class Workspace:
    slug: str
    name: str
    created: str
    revisions: tuple[Revision, ...]

    @property
    def latest(self) -> Revision | None:
        return self.revisions[-1] if self.revisions else None


class WorkspaceError(ValueError):
    """A request the store cannot honour (unknown workspace, empty name); the message says why."""


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:60]


class Store:
    """Workspaces as ``<root>/<slug>/workspace.json``. Safe to call before the root exists."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root if root is not None else default_root()

    def _path(self, slug: str) -> Path:
        if not slug or slug != slugify(slug):  # no path separators, nothing outside the root
            raise WorkspaceError(f"{slug!r} is not a workspace")
        return self.root / slug / "workspace.json"

    def _write(self, ws: Workspace) -> None:
        path = self._path(ws.slug)
        path.parent.mkdir(parents=True, exist_ok=True)
        body = {
            "name": ws.name,
            "created": ws.created,
            "revisions": [r.to_dict() for r in ws.revisions],
        }
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(body, indent=1, sort_keys=True) + "\n")
        tmp.replace(path)  # a crash mid-write never leaves half a file

    def slugs(self) -> list[str]:
        if not self.root.is_dir():
            return []
        return sorted(p.parent.name for p in self.root.glob("*/workspace.json"))

    def load(self, slug: str) -> Workspace:
        path = self._path(slug)
        if not path.is_file():
            raise WorkspaceError(f"There is no workspace {slug!r}.")
        try:
            data = json.loads(path.read_text())
            revisions = tuple(
                Revision(int(r["id"]), str(r["saved"]), dict(r["settings"]))
                for r in data["revisions"]
            )
            return Workspace(slug, str(data["name"]), str(data["created"]), revisions)
        except (ValueError, KeyError, TypeError) as exc:
            raise WorkspaceError(f"Workspace {slug!r} is unreadable: {exc}") from exc

    def list(self) -> list[Workspace]:
        """Every readable workspace, newest change first; an unreadable one is skipped."""
        found = []
        for slug in self.slugs():
            try:
                found.append(self.load(slug))
            except WorkspaceError:
                continue
        return sorted(found, key=lambda w: w.latest.saved if w.latest else w.created, reverse=True)

    def create(self, name: str, settings: dict | None = None) -> Workspace:
        """A new workspace; its slug comes from the name and is made unique with a number."""
        name = " ".join(name.split())
        base = slugify(name)
        if not base:
            raise WorkspaceError("Give the workspace a name.")
        slug, n = base, 2
        while self._path(slug).exists():
            slug, n = f"{base}-{n}", n + 1
        ws = Workspace(slug, name, _now(), ())
        if settings is not None:
            ws = Workspace(slug, name, ws.created, (Revision(1, ws.created, settings),))
        self._write(ws)
        return ws

    def save(self, slug: str, settings: dict) -> Workspace:
        """Add a revision, unless the settings equal the latest (so generating twice with the same
        form saves once)."""
        ws = self.load(slug)
        if ws.latest is not None and ws.latest.settings == settings:
            return ws
        nxt = (ws.latest.id + 1) if ws.latest else 1
        revisions = (*ws.revisions, Revision(nxt, _now(), settings))[-MAX_REVISIONS:]
        ws = Workspace(ws.slug, ws.name, ws.created, revisions)
        self._write(ws)
        return ws

    def rename(self, slug: str, name: str) -> Workspace:
        ws = self.load(slug)
        name = " ".join(name.split())
        if not name:
            raise WorkspaceError("Give the workspace a name.")
        ws = Workspace(ws.slug, name, ws.created, ws.revisions)
        self._write(ws)
        return ws

    def duplicate(self, slug: str, name: str) -> Workspace:
        """A copy holding only the latest settings, as a fresh history."""
        source = self.load(slug)
        return self.create(name, source.latest.settings if source.latest else None)

    def delete(self, slug: str) -> None:
        path = self._path(slug)
        if not path.is_file():
            raise WorkspaceError(f"There is no workspace {slug!r}.")
        path.unlink()
        try:
            path.parent.rmdir()  # only when nothing else lives there
        except OSError:
            pass
