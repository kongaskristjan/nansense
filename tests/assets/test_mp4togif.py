import os
import tempfile
from pathlib import Path

import pytest

from assets.generators.mp4togif import (
    _extract_number,
    _number_part,
    _resolve_input,
    build_filter,
)


class TestBuildFilter:
    def test_basic(self) -> None:
        result = build_filter(fps=10, width=None, stats_mode="full", dither="sierra2_4a")
        assert "fps=10" in result
        assert "palettegen=stats_mode=full" in result
        assert "paletteuse=dither=sierra2_4a" in result
        assert "scale" not in result

    def test_with_width(self) -> None:
        result = build_filter(fps=15, width=480, stats_mode="diff", dither="none")
        assert "scale=480:-2:flags=lanczos" in result
        assert "fps=15" in result


class TestNumberPart:
    @pytest.mark.parametrize("filename,expected", [
        ("000001.png", "000001"),
        ("01.jpg", "01"),
        ("1.png", "1"),
    ])
    def test_with_numbers(self, filename: str, expected: str) -> None:
        assert _number_part(filename) == expected

    def test_no_leading_digits(self) -> None:
        assert _number_part("abc.png") == ""


class TestExtractNumber:
    @pytest.mark.parametrize("filename,expected", [
        ("000001.png", 1),
        ("01.jpg", 1),
        ("42.png", 42),
    ])
    def test_valid(self, filename: str, expected: int) -> None:
        assert _extract_number(filename) == expected

    def test_no_digits(self) -> None:
        assert _extract_number("abc.png") is None


class TestResolveInput:
    def test_file_passed_through(self, tmp_path: Path) -> None:
        f = tmp_path / "video.mp4"
        f.write_bytes(b"\x00")
        assert _resolve_input(str(f)) == str(f)

    def test_nonexistent_path_exits(self) -> None:
        with pytest.raises(SystemExit, match="neither a file nor a directory"):
            _resolve_input("/nonexistent/path")

    def test_empty_directory_exits(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit, match="no image files found"):
            _resolve_input(str(tmp_path))

    def test_directory_with_sequential_images(self, tmp_path: Path) -> None:
        for i in range(1, 4):
            (tmp_path / f"{i:06d}.png").write_bytes(b"\x00")
        result = _resolve_input(str(tmp_path))
        assert "%06d.png" in result

    def test_directory_with_start_at_zero(self, tmp_path: Path) -> None:
        for i in range(0, 3):
            (tmp_path / f"{i:06d}.png").write_bytes(b"\x00")
        result = _resolve_input(str(tmp_path))
        assert "%06d.png" in result
