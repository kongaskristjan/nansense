"""Tests for the examples' fast dataset mirrors (no network access)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from torchvision import datasets

from examples import mirrors


def test_mirror_serves_the_archive_torchvision_expects() -> None:
    """The mirror is only safe because torchvision's integrity metadata still
    applies to it: same filename, same md5 as the canonical archive."""
    assert mirrors.CIFAR10_MIRROR_URL.endswith(datasets.CIFAR10.filename)
    assert mirrors._MirroredCIFAR10.url == mirrors.CIFAR10_MIRROR_URL
    assert mirrors._MirroredCIFAR10.filename == datasets.CIFAR10.filename
    assert mirrors._MirroredCIFAR10.tgz_md5 == datasets.CIFAR10.tgz_md5


def test_mirror_url_is_not_the_slow_canonical_host() -> None:
    assert mirrors.CIFAR10_CANONICAL_URL == datasets.CIFAR10.url
    assert mirrors.CIFAR10_MIRROR_URL != mirrors.CIFAR10_CANONICAL_URL


def _record_calls(
    monkeypatch: pytest.MonkeyPatch, mirror: type[Exception] | None
) -> list[tuple[str, dict[str, Any]]]:
    """Stub both CIFAR-10 classes, recording which one the helper reaches for.

    `mirror` is the exception type the mirrored class raises, or None to let it
    succeed.
    """
    calls: list[tuple[str, dict[str, Any]]] = []

    def stub(name: str, fail: type[Exception] | None) -> Any:
        def build(*args: Any, **kwargs: Any) -> str:
            calls.append((name, kwargs))
            if fail is not None:
                raise fail("boom")
            return name

        return build

    monkeypatch.setattr(mirrors, "_MirroredCIFAR10", stub("mirror", mirror))
    monkeypatch.setattr(datasets, "CIFAR10", stub("canonical", None))
    return calls


def test_mirror_is_used_first(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = _record_calls(monkeypatch, mirror=None)

    assert mirrors.cifar10(tmp_path, train=True) == "mirror"
    assert [name for name, _ in calls] == ["mirror"]


@pytest.mark.parametrize("error", [RuntimeError, OSError])
def test_falls_back_to_the_canonical_host(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    error: type[Exception],
) -> None:
    """A corrupt archive (RuntimeError) or a network failure (OSError) at the
    mirror must not break the example: the slow canonical URL still works."""
    calls = _record_calls(monkeypatch, mirror=error)

    assert mirrors.cifar10(tmp_path, train=False) == "canonical"
    assert [name for name, _ in calls] == ["mirror", "canonical"]
    assert calls[1][1]["download"] is True
    assert "mirror failed" in capsys.readouterr().out


def test_no_fallback_when_downloads_are_disabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`download=False` means "use the cache or fail" — a distributed rank
    waiting on rank 0's copy must not start its own download."""
    calls = _record_calls(monkeypatch, mirror=RuntimeError)

    with pytest.raises(RuntimeError):
        mirrors.cifar10(tmp_path, train=True, download=False)
    assert [name for name, _ in calls] == ["mirror"]
