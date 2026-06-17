#!/usr/bin/env python3
"""Convert an MP4 (or any ffmpeg-readable video) to an optimized GIF.

Uses ffmpeg's palettegen/paletteuse filter chain in a single pass for
good-quality 256-color GIFs. --mode tunes the palette stats and dithering
for different content types.
"""

import argparse
import shutil
import subprocess
import sys

# Per-mode ffmpeg settings.
#   stats_mode=diff weights the palette toward regions that actually change,
#     which helps mostly-static screen/UI captures.
#   dither trades color banding against file size; flat-color content
#     compresses best and looks sharpest with no dithering.
MODES = {
    "normal": {"stats_mode": "full", "dither": "sierra2_4a"},        # real video
    "ui":     {"stats_mode": "diff", "dither": "bayer:bayer_scale=5"},  # screen/UI capture
    "flat":   {"stats_mode": "diff", "dither": "none"},              # flat-color diagrams
}


def build_filter(fps, width, stats_mode, dither):
    steps = [f"fps={fps}"]
    if width is not None:
        steps.append(f"scale={width}:-2:flags=lanczos")  # else keep original resolution
    pre = ",".join(steps)
    return (
        f"{pre},split[s0][s1];"
        f"[s0]palettegen=stats_mode={stats_mode}[p];"
        f"[s1][p]paletteuse=dither={dither}"
    )


def main():
    p = argparse.ArgumentParser(description="Convert a video to an optimized GIF.")
    p.add_argument("input", help="input video file")
    p.add_argument("output", help="output .gif file")
    p.add_argument(
        "--mode", choices=MODES, default="normal",
        help="tuning preset: normal (real video), ui (screen/UI capture), "
             "flat (flat-color diagrams) [default: normal]",
    )
    p.add_argument("--fps", type=int, default=15, help="output frame rate [default: 15]")
    p.add_argument("--width", type=int, default=None,
                   help="output width in px, height auto [default: keep original resolution]")
    args = p.parse_args()

    if shutil.which("ffmpeg") is None:
        sys.exit("error: ffmpeg not found on PATH")

    cfg = MODES[args.mode]
    vf = build_filter(args.fps, args.width, cfg["stats_mode"], cfg["dither"])
    cmd = ["ffmpeg", "-y", "-i", args.input, "-vf", vf, args.output]

    print("running:", " ".join(cmd), file=sys.stderr)
    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    sys.exit(main())

