"""Saved workspaces (GRA-194): the store on disk, and the UI opening, saving and restoring them."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from kggen_ui import workspaces
from kggen_ui.workspaces import Store, WorkspaceError


def test_a_workspace_is_a_name_and_a_slug_that_stays_unique(tmp_path: Path) -> None:
    store = Store(tmp_path)
    a = store.create("Harbour towns, heavy churn")
    b = store.create("harbour towns heavy churn")
    assert a.slug == "harbour-towns-heavy-churn" and b.slug == "harbour-towns-heavy-churn-2"
    assert [w.name for w in store.list()] and store.slugs() == sorted([a.slug, b.slug])
    with pytest.raises(WorkspaceError, match="name"):
        store.create("   ")
    with pytest.raises(WorkspaceError, match="name"):
        store.create("!!!")


def test_saving_adds_a_revision_only_when_the_settings_changed(tmp_path: Path) -> None:
    store = Store(tmp_path)
    ws = store.create("a")
    store.save(ws.slug, {"preset": "small"})
    store.save(ws.slug, {"preset": "small"})  # generating twice with one form saves once
    store.save(ws.slug, {"preset": "medium"})
    loaded = store.load(ws.slug)
    assert [r.id for r in loaded.revisions] == [1, 2]
    assert loaded.latest is not None and loaded.latest.settings == {"preset": "medium"}


def test_revisions_are_capped_and_ids_keep_counting(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(workspaces, "MAX_REVISIONS", 3)
    store = Store(tmp_path)
    ws = store.create("a")
    for n in range(5):
        store.save(ws.slug, {"n": n})
    assert [r.id for r in store.load(ws.slug).revisions] == [3, 4, 5]


def test_rename_duplicate_and_delete(tmp_path: Path) -> None:
    store = Store(tmp_path)
    ws = store.create("a", {"preset": "small"})
    store.save(ws.slug, {"preset": "medium"})
    assert store.rename(ws.slug, "Better name").name == "Better name"
    copy = store.duplicate(ws.slug, "copy")
    assert copy.latest is not None and copy.latest.settings == {"preset": "medium"}
    assert len(copy.revisions) == 1  # a fresh history
    store.delete(ws.slug)
    assert store.slugs() == [copy.slug]
    with pytest.raises(WorkspaceError, match="no workspace"):
        store.load(ws.slug)


def test_a_slug_cannot_leave_the_root(tmp_path: Path) -> None:
    store = Store(tmp_path / "root")
    for bad in ("../x", "a/b", "", ".."):
        with pytest.raises(WorkspaceError):
            store.load(bad)


def test_an_unreadable_workspace_is_skipped_not_fatal(tmp_path: Path) -> None:
    store = Store(tmp_path)
    good = store.create("good")
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "workspace.json").write_text("{not json")
    assert [w.slug for w in store.list()] == [good.slug]
    with pytest.raises(WorkspaceError, match="unreadable"):
        store.load("broken")


def test_the_file_is_plain_json_that_can_be_read_back(tmp_path: Path) -> None:
    store = Store(tmp_path)
    ws = store.create("a", {"preset": "small", "values": {"seed": 7}})
    data = json.loads((tmp_path / ws.slug / "workspace.json").read_text())
    assert data["name"] == "a" and data["revisions"][0]["settings"]["values"] == {"seed": 7}
    assert not list(tmp_path.rglob("*.tmp"))


def test_a_store_that_does_not_exist_yet_lists_nothing(tmp_path: Path) -> None:
    assert Store(tmp_path / "nope").list() == []


def test_the_default_root_follows_the_environment(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv(workspaces.ENV_DIR, str(tmp_path))
    assert workspaces.default_root() == tmp_path


# -- the UI ------------------------------------------------------------------------------------

pytest.importorskip("indah")  # the UI is an optional extra; the fast gate runs without it

from kggen_ui.app import build_ui  # noqa: E402


def ui_with(tmp_path: Path):  # type: ignore[no-untyped-def]
    ui = build_ui(Store(tmp_path))
    ui.difficulty.set("clean")
    ui.preset.set("tiny")
    ui.name_style.set("invented")
    return ui


def test_a_scratch_session_saves_nothing(tmp_path: Path) -> None:
    ui = ui_with(tmp_path)
    ui.generate()
    assert ui.ws_current.value is None and ui.store.slugs() == []
    assert "Scratch" in ui.workspace_title()


def test_a_new_workspace_keeps_the_form_and_each_changed_generate_adds_a_revision(
    tmp_path: Path,
) -> None:
    ui = ui_with(tmp_path)
    ui.ws_name.set("Tiny smoke")
    ui.create_workspace()
    assert ui.ws_current.value == "tiny-smoke" and ui.workspace_title() == "Tiny smoke"
    assert [r["slug"] for r in ui.ws_rows.value] == ["tiny-smoke"]
    ui.generate()
    ui.generate()  # same settings: no new revision
    assert len(ui.rev_rows.value) == 1
    ui.values["seed"].set(99)
    ui.generate()
    assert [r["id"] for r in ui.rev_rows.value] == ["2", "1"]  # newest first
    assert "seed 99" in ui.rev_rows.value[0]["what"]


def test_opening_a_workspace_restores_its_settings_and_makes_the_data_again(
    tmp_path: Path,
) -> None:
    first = ui_with(tmp_path)
    first.ws_name.set("kept")
    first.create_workspace()
    first.values["seed"].set(1234)
    first.evolution.set("heavy")
    first.values["seed"].set(1234)  # the evolution select refilled the numbers from its preset
    first.generate()
    entities = first.entities.value
    # A second viewer on the same disk opens it.
    second = build_ui(Store(tmp_path))
    assert second.ws_current.value is None
    second.open_workspace("kept")
    assert second.ws_current.value == "kept"
    assert second.evolution.value == "heavy" and second.values["seed"].value == 1234
    assert second.entities.value == entities and second.world.value is not None
    assert second.world.value.to_json() == first.world.value.to_json()  # the same dataset


def test_restoring_a_revision_puts_back_its_settings(tmp_path: Path) -> None:
    ui = ui_with(tmp_path)
    ui.ws_name.set("r")
    ui.create_workspace()
    ui.values["seed"].set(11)
    ui.generate()
    ui.values["seed"].set(22)
    ui.generate()
    ui.rev_picked.set("2")  # revision 1 is the form as it was when the workspace was made
    ui.restore_revision()
    assert ui.values["seed"].value == 11
    assert len(ui.rev_rows.value) == 4  # restoring is itself a revision


def test_a_big_preset_is_restored_but_not_generated_until_asked(tmp_path: Path) -> None:
    store = Store(tmp_path)
    store.create("big", {"preset": "large", "difficulty": "clean"})
    ui = build_ui(store)
    ui.open_workspace("big")
    assert ui.preset.value == "large" and ui.world.value is None
    assert "Press Generate" in ui.status.value


def test_rename_duplicate_delete_from_the_ui(tmp_path: Path) -> None:
    ui = ui_with(tmp_path)
    ui.ws_name.set("one")
    ui.create_workspace()
    ui.ws_name.set("Uno")
    ui.rename_workspace()
    assert ui.workspace_title() == "Uno"
    ui.duplicate_workspace()
    assert ui.ws_current.value == "uno-copy" and len(ui.ws_rows.value) == 2
    ui.delete_workspace()
    assert ui.ws_current.value is None and [r["slug"] for r in ui.ws_rows.value] == ["one"]


def test_actions_without_a_workspace_say_so_instead_of_failing(tmp_path: Path) -> None:
    ui = ui_with(tmp_path)
    ui.rename_workspace()
    assert "Open a workspace" in ui.ws_note.value
    ui.restore_revision()
    assert "Pick a revision" in ui.ws_note.value
    ui.ws_name.set("")
    ui.create_workspace()
    assert "name" in ui.ws_note.value and ui.store.slugs() == []
