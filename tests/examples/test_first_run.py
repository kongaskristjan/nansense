"""Tests for the examples' cold-start notice."""

from __future__ import annotations

from pathlib import Path

import pytest

from examples.first_run import DEFAULT_DATA_DIR, argv_value, note_first_run


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ([], "fallback"),
        (["--dataset", "mnist"], "mnist"),
        (["--dataset=mnist"], "mnist"),
        (["--epochs", "3", "--dataset", "mnist", "--seed", "0"], "mnist"),
        (["--dataset"], "fallback"),  # missing value: argparse reports it
        (["--datasets", "mnist"], "fallback"),  # a longer flag must not match
    ],
)
def test_argv_value(
    monkeypatch: pytest.MonkeyPatch, args: list[str], expected: str
) -> None:
    monkeypatch.setattr("sys.argv", ["main.py", *args])

    assert argv_value("--dataset", "fallback") == expected


def test_notice_when_dataset_is_missing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    monkeypatch.setattr("sys.argv", ["main.py", "--data-dir", str(tmp_path)])

    note_first_run("cifar10")

    out = capsys.readouterr().out
    assert "First run" in out
    assert str(tmp_path) in out


def test_quiet_when_every_dataset_is_cached(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    monkeypatch.setattr("sys.argv", ["main.py", f"--data-dir={tmp_path}"])
    for name in ("cifar-10-batches-py", "MNIST"):
        (tmp_path / name).mkdir()

    note_first_run("cifar10", "mnist")

    assert capsys.readouterr().out == ""


def test_notice_when_only_some_datasets_are_cached(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """Make3D needs four archives; three of them is still a cold start."""
    monkeypatch.setattr("sys.argv", ["main.py", "--data-dir", str(tmp_path)])
    for name in ("Train400Img", "Train400Depth", "Test134"):
        (tmp_path / name).mkdir()

    note_first_run("make3d")

    assert "First run" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("args", "env"),
    [
        (["--help"], {}),
        (["-h"], {}),
        ([], {"RANK": "1"}),  # a follower rank in a torchrun launch
    ],
)
def test_quiet_for_help_and_follower_ranks(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    args: list[str],
    env: dict[str, str],
) -> None:
    monkeypatch.setattr("sys.argv", ["main.py", "--data-dir", str(tmp_path), *args])
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    note_first_run("cifar10")

    assert capsys.readouterr().out == ""


def test_unknown_dataset_name_is_ignored(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """`--playground` is required, so argv can hand the notice an empty name."""
    monkeypatch.setattr("sys.argv", ["main.py", "--data-dir", str(tmp_path)])

    note_first_run("")

    assert capsys.readouterr().out == ""


def test_default_data_dir_is_used_without_the_flag(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["main.py"])

    note_first_run("cifar10")

    assert str(DEFAULT_DATA_DIR) in capsys.readouterr().out
