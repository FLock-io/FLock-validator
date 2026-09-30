"""Command line tool that builds a video-inconsistency validation package.

Public dev package vs private eval package
------------------------------------------
* The **public dev package** is built with a published seed and purely
  procedural clips so trainers can reproduce it and iterate locally.
* The **private eval package** is what validators score against. Build it with
  a *secret* seed and, ideally, real footage (``--footage-dir``) so submissions
  cannot overfit to the procedural renderer. Never publish the private seed or
  the private zip's clip labels: anyone holding the seed can regenerate the
  exact clips and labels.

Example::

    python -m validator.modules.video_inconsistency.build_package \
        --output dev_package.zip --num-clips 60 --seed 1234
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tempfile
from collections import Counter
from pathlib import Path

from validator.modules.video_inconsistency.issue_types import ISSUE_TYPE_NAMES
from validator.modules.video_inconsistency.package import (
    build_validation_package,
    resolve_validation_package,
)
from validator.modules.video_inconsistency.manifest import DIFFICULTIES
from validator.modules.video_inconsistency.synthesis import (
    DECOY_TYPES,
    SynthesisConfig,
    collect_footage,
)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    defaults = SynthesisConfig()
    parser = argparse.ArgumentParser(
        prog="build_package",
        description="Build a video inconsistency validation package (zip).",
    )
    parser.add_argument("--output", required=True, help="output zip path")
    parser.add_argument("--num-clips", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0, help="keep the private eval seed secret")
    parser.add_argument("--width", type=int, default=defaults.width)
    parser.add_argument("--height", type=int, default=defaults.height)
    parser.add_argument("--fps", type=float, default=defaults.fps)
    parser.add_argument("--min-duration", type=float, default=defaults.min_duration)
    parser.add_argument("--max-duration", type=float, default=defaults.max_duration)
    parser.add_argument("--clean-fraction", type=float, default=defaults.clean_fraction)
    parser.add_argument("--footage-dir", help="directory of real videos (*.mp4/*.mov/*.mkv/*.webm)")
    parser.add_argument(
        "--footage-fraction",
        type=float,
        default=None,
        help="share of clips drawn from --footage-dir (default 0.5 when a footage dir is given)",
    )
    parser.add_argument("--decoy-rate", type=float, default=defaults.decoy_rate,
                        help="expected decoys per 8 s of video (legitimate, unlabelled events)")
    parser.add_argument("--no-degradation", action="store_true",
                        help="disable sensor noise / blur / rescale and use crf 18 (debugging)")
    parser.add_argument("--crf-min", type=int, default=defaults.crf_range[0])
    parser.add_argument("--crf-max", type=int, default=defaults.crf_range[1])
    parser.add_argument(
        "--difficulty-weights",
        help="comma separated name=weight, e.g. easy=0.15,medium=0.3,hard=0.3,expert=0.25",
    )
    parser.add_argument(
        "--issue-types",
        help="comma separated subset of: " + ",".join(ISSUE_TYPE_NAMES),
    )
    return parser.parse_args(argv)


def _config_from_args(args: argparse.Namespace) -> SynthesisConfig:
    footage: tuple[str, ...] = ()
    fraction = 0.0
    if args.footage_dir:
        footage = collect_footage(args.footage_dir)
        if not footage:
            raise SystemExit(f"no video files found under {args.footage_dir}")
        fraction = 0.5 if args.footage_fraction is None else args.footage_fraction
    elif args.footage_fraction:
        raise SystemExit("--footage-fraction needs --footage-dir")
    issue_types = ISSUE_TYPE_NAMES
    if args.issue_types:
        issue_types = tuple(name.strip() for name in args.issue_types.split(",") if name.strip())
    weights = SynthesisConfig().difficulty_weights
    if args.difficulty_weights:
        try:
            weights = tuple(
                (name.strip(), float(value))
                for name, value in (item.split("=") for item in args.difficulty_weights.split(",") if item.strip())
            )
        except ValueError as exc:
            raise SystemExit(f"invalid --difficulty-weights: {exc}") from exc
    return SynthesisConfig(
        difficulty_weights=weights,
        decoy_rate=args.decoy_rate,
        degradation=not args.no_degradation,
        crf_range=(args.crf_min, args.crf_max),
        width=args.width,
        height=args.height,
        fps=args.fps,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        clean_fraction=args.clean_fraction,
        issue_types=issue_types,
        footage_paths=footage,
        footage_fraction=fraction,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        config = _config_from_args(args)
    except ValueError as exc:
        raise SystemExit(f"invalid configuration: {exc}") from exc
    output = build_validation_package(
        args.output, num_clips=args.num_clips, seed=args.seed, config=config
    )
    digest = hashlib.sha256(Path(output).read_bytes()).hexdigest()
    # Re-read the zip exactly as a validator would, to catch a bad package at build time.
    with tempfile.TemporaryDirectory(prefix="video_pkg_check_") as check_cache:
        resolved = resolve_validation_package(str(output), check_cache)
    counts = Counter(issue.type for clip in resolved.manifest.clips for issue in clip.issues)
    clean = sum(1 for clip in resolved.manifest.clips if not clip.issues)
    print(f"output: {output}")
    print(f"size: {Path(output).stat().st_size} bytes")
    print(f"sha256: {digest}")
    print(f"clips: {len(resolved.manifest.clips)} ({clean} clean)")
    for name in ISSUE_TYPE_NAMES:
        print(f"  {name}: {counts.get(name, 0)}")
    difficulty = Counter(clip.difficulty for clip in resolved.manifest.clips)
    print("difficulty:")
    for name in DIFFICULTIES:
        print(f"  {name}: {difficulty.get(name, 0)}")
    decoys = Counter(decoy.type for clip in resolved.manifest.clips for decoy in clip.decoys)
    with_decoys = sum(1 for clip in resolved.manifest.clips if clip.decoys)
    print(f"decoys: {sum(decoys.values())} across {with_decoys} clips")
    for name in DECOY_TYPES:
        print(f"  {name}: {decoys.get(name, 0)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
