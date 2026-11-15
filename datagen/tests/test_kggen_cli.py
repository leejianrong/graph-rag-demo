"""The kggen CLI is installed and answers --help / --version."""

from __future__ import annotations

import subprocess
import sys

import pytest
from kggen import __version__
from kggen.cli import main


def test_main_prints_help_and_returns_zero(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    assert "kggen" in capsys.readouterr().out


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_module_entry_point_runs() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "kggen", "--help"], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0
    assert "usage: kggen" in result.stdout


def test_difficulty_makes_a_preset_world_ambiguous_and_clean_leaves_it_alone(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """GRA-183: the same preset, with and without ``--difficulty hard``."""
    assert main(["world", "stats", "--preset", "small"]) == 0
    clean = capsys.readouterr().out
    assert main(["world", "stats", "--preset", "small", "--difficulty", "hard"]) == 0
    hard = capsys.readouterr().out

    assert clean != hard
    assert main(["world", "stats", "--preset", "small", "--difficulty", "clean"]) == 0
    assert capsys.readouterr().out == clean
