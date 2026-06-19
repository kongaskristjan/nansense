#!/usr/bin/env python3
"""Convert a video, existing GIF, or directory of numbered images to an optimized GIF.

Uses ffmpeg's palettegen/paletteuse filter chain in a single pass for
good-quality 256-color GIFs. --mode tunes the palette stats and dithering
for different content types.
"""

import argparse
import os
import shutil
import subprocess
import sys

MODES = {
    "normal": {"stats_mode": "full", "dither": "sierra2_4a"},
    "ui":     {"stats_mode": "diff", "dither": "bayer:bayer_scale=5"},
    "flat":   {"stats_mode": "diff", "dither": "none"},
}


def build_filter(fps: int | None, width: int | None, stats_mode: str, dither: str) -> str:
    steps = []
    if fps is not None:
        steps.append(f"fps={fps}")
    if width is not None:
        steps.append(f"scale={width}:-2:flags=lanczos")
    pre = ",".join(steps) + "," if steps else ""
    return (
        f"{pre}split[s0][s1];"
        f"[s0]palettegen=stats_mode={stats_mode}[p];"
        f"[s1][p]paletteuse=dither={dither}"
    )


def _resolve_input(input_path: str) -> str:
    """Return an ffmpeg-compatible input specifier.

    * If *input_path* is a file  -> return it as-is.
    * If *input_path* is a directory -> return an ``-i``-safe glob pattern
      (``dir/%06d.png`` etc.) based on the numbered images found inside.
    """
    if os.path.isfile(input_path):
        return input_path

    if not os.path.isdir(input_path):
        sys.exit(f"error: {input_path!r} is neither a file nor a directory")

    IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
    files = sorted(
        f
        for f in os.listdir(input_path)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTS
    )
    if not files:
        sys.exit(f"error: no image files found in {input_path!r}")

    ext = os.path.splitext(files[0])[1]
    start = _extract_number(files[0])
    if start is None:
        sys.exit(f"error: cannot determine numbering from {files[0]!r}")

    ndigits = len(_number_part(files[0]))
    return os.path.join(input_path, f"%0{ndigits}d{ext}" if start == 1 else f"%0{ndigits}d{ext}")


def _number_part(filename: str) -> str:
    """Return the leading digit run of *filename* (e.g. '000001' from '000001.png')."""
    digits: list[str] = []
    for ch in filename:
        if ch.isdigit():
            digits.append(ch)
        else:
            break
    return "".join(digits)


def _extract_number(filename: str) -> int | None:
    part = _number_part(filename)
    return int(part) if part else None


def main() -> int:
    p = argparse.ArgumentParser(
        description="Convert a video, GIF, or directory of numbered images to an optimized GIF.",
    )
    p.add_argument("input", help="input video/gif file or directory of numbered images")
    p.add_argument("output", help="output .gif file")
    p.add_argument(
        "--mode", choices=MODES, default="normal",
        help="tuning preset: normal (real video), ui (screen/UI capture), "
             "flat (flat-color diagrams) [default: normal]",
    )
    p.add_argument("--fps", type=int, default=15, help="output frame rate [default: 15]")
    p.add_argument("--width", type=int, default=None,
                   help="output width in px, height auto [default: keep original resolution]")
    p.add_argument("--start-number", type=int, default=None,
                   help="first frame number when using an image directory [default: auto-detect]")
    args = p.parse_args()

    if shutil.which("ffmpeg") is None:
        sys.exit("error: ffmpeg not found on PATH")

    cfg = MODES[args.mode]
    input_spec = _resolve_input(args.input)
    is_dir = os.path.isdir(args.input)

    # For an image directory, set the frame rate on the INPUT (-framerate) so
    # every image becomes one GIF frame. Using the fps filter instead resamples
    # ffmpeg's default 25fps input timeline and silently drops frames.
    vf = build_filter(None if is_dir else args.fps, args.width, cfg["stats_mode"], cfg["dither"])

    cmd = ["ffmpeg", "-y"]
    if is_dir:
        cmd += ["-framerate", str(args.fps)]
    if args.start_number is not None:
        cmd += ["-start_number", str(args.start_number)]
    elif is_dir:
        files = sorted(
            f
            for f in os.listdir(args.input)
            if os.path.splitext(f)[1].lower()
            in {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
        )
        start = _extract_number(files[0]) if files else None
        # ffmpeg's image2 demuxer defaults start_number to 1, so an explicit
        # value is only needed when the sequence starts anywhere but 1.
        if start is not None and start != 1:
            cmd += ["-start_number", str(start)]
    cmd += ["-i", input_spec, "-vf", vf, args.output]

    print("running:", " ".join(cmd), file=sys.stderr)
    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    sys.exit(main())

