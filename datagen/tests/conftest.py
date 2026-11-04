from __future__ import annotations

import pytest
from llm_fake import serve


@pytest.fixture
def endpoint():
    """A local OpenAI-compatible server whose answers the test chooses (see ``llm_fake``)."""
    with serve() as make:
        yield make
