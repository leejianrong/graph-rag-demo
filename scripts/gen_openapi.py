"""Write (or check) docs/openapi.json, the API's OpenAPI document (GRA-46).

The app is built over in-memory fakes (no containers, no model, no key), so this runs anywhere.
The UI's TypeScript client is generated from the committed file; a fast-suite test fails when the
file no longer matches the app. Regenerate with ``make openapi``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from pydantic_settings import BaseSettings, PydanticBaseSettingsSource

from graph_rag.config import Settings
from graph_rag.offline_server import build_app

SPEC_PATH = Path(__file__).resolve().parent.parent / "docs" / "openapi.json"


class PinnedSettings(Settings):
    """The defaults only: neither the environment nor ``.env`` can change the spec (GRA-151)."""

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (init_settings,)


def render() -> str:
    """The app's OpenAPI document as stable text: sorted keys, two-space indent, final newline."""
    return (
        json.dumps(
            build_app(PinnedSettings()).openapi(), indent=2, sort_keys=True, ensure_ascii=False
        )
        + "\n"
    )


def main() -> int:
    """Rewrite the committed file, or with ``--check`` exit 1 when it is stale."""
    text = render()
    if "--check" in sys.argv:
        return 0 if SPEC_PATH.read_text(encoding="utf-8") == text else 1
    SPEC_PATH.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
