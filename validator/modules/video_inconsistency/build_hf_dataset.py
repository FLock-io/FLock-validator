"""Operator tool: build (and optionally upload) the public training dataset for the task.

    python -m validator.modules.video_inconsistency.build_hf_dataset \
        --out-dir hf_dataset --train-clips 8000 --validation-clips 1000 --workers 16
    HF_TOKEN=hf_xxx python -m validator.modules.video_inconsistency.build_hf_dataset \
        --out-dir hf_dataset --skip-build --push-to-hub random-sequence/flock-video-inconsistency

Layout (Hugging Face ``videofolder``-compatible)::

    README.md                  dataset card
    issue_types.json           canonical issue (and decoy) catalogue
    train/metadata.jsonl       one row per clip; ``file_name`` is relative to the file
    train/<clip_id>.mp4
    validation/metadata.jsonl
    validation/<clip_id>.mp4
    dev_package/video_inconsistency_dev_package.zip   local-validation package (see below)
    stats.json                 counts per split

A metadata row is::

    {"file_name": "<clip_id>.mp4", "clip_id", "fps", "num_frames", "width", "height",
     "duration", "difficulty", "source", "crf", "issues": [IssueLabel...], "decoys": [DecoyLabel...]}

Seeds and privacy
-----------------
Train / validation clip seeds are hash-derived from ``--seed`` in their own namespace, so they
are disjoint from each other and from the dev package (whose seeds come from the package
builder). Clip ids are 12 hex characters of a hash of (split, seed, index): they carry no label
information. The validators' PRIVATE evaluation set is generated with a different, secret seed
and must never be published with this dataset.

Determinism and resuming
------------------------
Every clip depends only on (split, seed, index), never on the worker count or on which clips
were built before, so the output is identical for any ``--workers``. A clip whose mp4 and row
already exist (and match the requested size) is skipped, so an interrupted build resumes.
Each mp4 is decoded back and must have exactly ``num_frames`` frames, otherwise the clip is
regenerated from a derived seed (as ``package.py`` does).

Upload
------
The token is read ONLY from the ``HF_TOKEN`` environment variable (never a CLI argument, never
printed). The repo is created private unless ``--public``; if it already exists and is public
while private was requested, the upload is refused. Upload uses ``upload_large_folder`` (resumable).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import sys
import time
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, get_args

from loguru import logger

from validator.modules.video_inconsistency.issue_types import (
    ISSUE_TYPE_NAMES,
    ISSUE_TYPES,
    SUITE_VERSION,
)
from validator.modules.video_inconsistency.manifest import (
    DIFFICULTIES,
    ClipSpec,
    DecoyLabel,
    DecoyType,
    IssueLabel,
    VideoManifest,
)
from validator.modules.video_inconsistency.package import (
    MANIFEST_FILENAME,
    build_validation_package,
)
from validator.modules.video_inconsistency.synthesis import (
    SynthesisConfig,
    collect_footage,
    generate_clip,
)
from validator.modules.video_inconsistency.video_io import encode_video, probe_video


DATASET_VERSION = "video_inconsistency_hf_v2"
DEFAULT_REPO_ID = "random-sequence/flock-video-inconsistency"
DEV_PACKAGE_RELPATH = "dev_package/video_inconsistency_dev_package.zip"
SPLITS: tuple[str, ...] = ("train", "validation")
DECOY_TYPES: tuple[str, ...] = get_args(DecoyType)
_CLIP_ENCODE_ATTEMPTS = 5
_BUILD_DIRNAME = ".build"  # per-clip row sidecars for resuming; removed when a split completes
_UPLOAD_IGNORE = [".build/**", ".cache/**", "**/*.tmp", "**/*.partial"]

DECOY_DESCRIPTIONS: dict[str, str] = {
    "scene_cut": "The shot changes to a different scene for good (an ordinary cut, not an inserted fragment).",
    "exposure_drift": "Brightness drifts smoothly up or down over a stretch (auto-exposure, a passing cloud).",
    "white_balance_drift": "The colour cast drifts smoothly over a stretch (auto white balance catching up).",
    "smooth_zoom": "The camera zooms in or out gradually (a normal push-in or pull-out).",
    "object_enters": "An object moves into the frame from outside, following the scene's motion.",
    "object_exits": "An object moves out of the frame, following the scene's motion.",
    "object_stops": "An object comes to rest while the rest of the scene keeps moving.",
    "camera_stops": "The camera pan slows to a halt and holds still (natural, gradual deceleration).",
}
DIFFICULTY_DESCRIPTIONS: dict[str, str] = {
    "easy": "1 edit per clip; long and strong, obvious to a casual viewer.",
    "medium": "1-2 edits per clip; moderate length and strength.",
    "hard": "2-3 edits per clip; short and subtle.",
    "expert": "2-4 edits per clip; the shortest and subtlest edits, together with the most decoys.",
}


class HubUploadError(RuntimeError):
    """The upload was refused (missing token, visibility mismatch, bad output folder)."""


# ---------------------------------------------------------------------------
# Identifiers and seeds
# ---------------------------------------------------------------------------


def clip_id_for(split: str, seed: int, index: int) -> str:
    """12 hex chars of sha256(split, seed, index): carries no label information."""
    digest = hashlib.sha256(f"video_inconsistency_hf_clip:{split}:{seed}:{index}".encode("utf-8"))
    return digest.hexdigest()[:12]


def clip_seed_for(split: str, seed: int, index: int, attempt: int) -> int:
    digest = hashlib.sha256(
        f"video_inconsistency_hf_seed:{split}:{seed}:{index}:{attempt}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big")


# ---------------------------------------------------------------------------
# Building clips
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Job:
    split: str
    index: int
    seed: int
    out_dir: str
    config: dict[str, Any]  # SynthesisConfig kwargs (picklable, spawn-safe)


def _config_from_kwargs(kwargs: dict[str, Any]) -> SynthesisConfig:
    return SynthesisConfig(**{**kwargs, "footage_paths": tuple(kwargs.get("footage_paths", ()))})


def _sidecar_path(out_dir: Path, split: str, clip_id: str) -> Path:
    return out_dir / _BUILD_DIRNAME / split / f"{clip_id}.json"


def _encode(frames: Any, path: Path, fps: float, crf: int) -> None:
    """Encode via a temporary name and rename, so an existing mp4 is always complete."""
    tmp = path.with_name(path.stem + ".tmp.mp4")
    try:
        encode_video(frames, tmp, fps, crf=crf)
    except TypeError:  # an older video_io without the crf argument
        encode_video(frames, tmp, fps)
    os.replace(tmp, path)


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _render_clip(job: _Job) -> dict[str, Any]:
    """Generate, encode and verify one clip; returns its metadata row (and stores a sidecar)."""
    out_dir = Path(job.out_dir)
    config = _config_from_kwargs(job.config)
    clip_id = clip_id_for(job.split, job.seed, job.index)
    path = out_dir / job.split / f"{clip_id}.mp4"
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(_CLIP_ENCODE_ATTEMPTS):
        clip = generate_clip(clip_seed_for(job.split, job.seed, job.index, attempt), config)
        crf = int(getattr(clip, "crf", 18))
        _encode(clip.frames, path, clip.fps, crf)
        # Label frame indices are only meaningful if the decoded video has exactly the planned length.
        if probe_video(path).num_frames == len(clip.frames):
            num_frames = int(len(clip.frames))
            row = {
                "file_name": f"{clip_id}.mp4",
                "clip_id": clip_id,
                "fps": float(clip.fps),
                "num_frames": num_frames,
                "width": int(clip.frames.shape[2]),
                "height": int(clip.frames.shape[1]),
                "duration": num_frames / float(clip.fps),
                "difficulty": clip.difficulty,
                "source": clip.source,
                "crf": crf,
                "issues": [issue.model_dump(mode="json") for issue in clip.issues],
                "decoys": [decoy.model_dump(mode="json") for decoy in getattr(clip, "decoys", [])],
            }
            _atomic_write(_sidecar_path(out_dir, job.split, clip_id), json.dumps(row))
            return row
        logger.warning(
            "{} clip {} attempt {}: decoded frame count differs from the plan; retrying with a new seed",
            job.split, job.index, attempt,
        )
    raise RuntimeError(f"could not produce a frame-exact encode for {job.split} clip {job.index}")


def validate_row(row: dict[str, Any]) -> None:
    """Raise if a metadata row does not round-trip through the manifest models."""
    for issue in row["issues"]:
        IssueLabel.model_validate(issue)
    for decoy in row["decoys"]:
        DecoyLabel.model_validate(decoy)
    ClipSpec(
        clip_id=row["clip_id"], video_path=row["file_name"], fps=row["fps"],
        num_frames=row["num_frames"], width=row["width"], height=row["height"],
        difficulty=row["difficulty"], source=row["source"],
        issues=[IssueLabel.model_validate(i) for i in row["issues"]],
        decoys=[DecoyLabel.model_validate(d) for d in row["decoys"]],
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _reusable_rows(out_dir: Path, split: str, config: SynthesisConfig) -> dict[str, dict[str, Any]]:
    """Rows of clips already built for this split (metadata.jsonl or resume sidecars)."""
    found: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(out_dir / split / "metadata.jsonl"):
        found[row["clip_id"]] = row
    sidecars = out_dir / _BUILD_DIRNAME / split
    if sidecars.is_dir():
        for path in sidecars.glob("*.json"):
            try:
                row = json.loads(path.read_text(encoding="utf-8"))
                found[row["clip_id"]] = row
            except (OSError, ValueError, KeyError):
                continue  # half-written sidecar: the clip is simply rebuilt
    usable = {}
    for clip_id, row in found.items():
        video = out_dir / split / f"{clip_id}.mp4"
        matches = (
            video.is_file() and video.stat().st_size > 0
            and row.get("file_name") == f"{clip_id}.mp4"
            and row.get("width") == config.width and row.get("height") == config.height
            and abs(float(row.get("fps", -1.0)) - config.fps) < 1e-9
        )
        if matches:
            usable[clip_id] = row
    return usable


def build_split(
    out_dir: Path,
    split: str,
    num_clips: int,
    seed: int,
    config_kwargs: dict[str, Any],
    workers: int,
) -> list[dict[str, Any]]:
    """Build (or resume) one split; writes ``<split>/metadata.jsonl`` and returns its rows."""
    config = _config_from_kwargs(config_kwargs)
    split_dir = out_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    ids = [clip_id_for(split, seed, index) for index in range(num_clips)]
    if len(set(ids)) != len(ids):
        raise RuntimeError(f"clip id collision in split {split!r}; choose another --seed")

    have = _reusable_rows(out_dir, split, config)
    pending = [
        _Job(split, index, seed, str(out_dir), config_kwargs)
        for index, clip_id in enumerate(ids)
        if clip_id not in have
    ]
    if pending:
        logger.info("{}: building {} of {} clips ({} already present)", split, len(pending), num_clips, num_clips - len(pending))
    started = time.time()
    built: dict[str, dict[str, Any]] = {}
    if workers <= 1 or len(pending) <= 1:
        results = map(_render_clip, pending)
        pool = None
    else:
        # spawn: no inherited state, so behaviour is identical on macOS and Linux
        pool = mp.get_context("spawn").Pool(min(workers, len(pending)))
        results = pool.imap_unordered(_render_clip, pending, chunksize=1)
    try:
        for count, row in enumerate(results, start=1):
            built[row["clip_id"]] = row
            if count % 100 == 0 or count == len(pending):
                print(f"[build] {split}: {count}/{len(pending)} clips ({time.time() - started:.0f}s)", flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    rows = [built.get(clip_id) or have[clip_id] for clip_id in ids]
    for row in rows:
        validate_row(row)
    _atomic_write(split_dir / "metadata.jsonl", "".join(json.dumps(row) + "\n" for row in rows))
    # Drop videos that are not part of this build (stale earlier runs): an mp4 without a
    # metadata row would break `videofolder` loading.
    expected = {f"{clip_id}.mp4" for clip_id in ids}
    for path in split_dir.glob("*.mp4"):
        if path.name not in expected:
            path.unlink()
    for path in split_dir.glob("*.tmp.mp4"):
        path.unlink()
    shutil.rmtree(out_dir / _BUILD_DIRNAME / split, ignore_errors=True)
    return rows


# ---------------------------------------------------------------------------
# Stats, catalogue and dataset card
# ---------------------------------------------------------------------------


def compute_stats(rows: Sequence[dict[str, Any]], total_bytes: int) -> dict[str, Any]:
    issues = Counter(issue["type"] for row in rows for issue in row["issues"])
    decoys = Counter(decoy["type"] for row in rows for decoy in row["decoys"])
    difficulty = Counter(row["difficulty"] for row in rows)
    return {
        "clips": len(rows),
        "clean_clips": sum(1 for row in rows if not row["issues"]),
        "issues_total": int(sum(issues.values())),
        "issues_per_type": {name: int(issues.get(name, 0)) for name in ISSUE_TYPE_NAMES},
        "decoys_total": int(sum(decoys.values())),
        "decoys_per_type": {name: int(decoys.get(name, 0)) for name in DECOY_TYPES},
        "difficulty_histogram": {name: int(difficulty.get(name, 0)) for name in DIFFICULTIES},
        "total_duration_seconds": round(float(sum(row["duration"] for row in rows)), 3),
        "bytes": int(total_bytes),
    }


def _split_bytes(out_dir: Path, split: str, rows: Sequence[dict[str, Any]]) -> int:
    return sum((out_dir / split / row["file_name"]).stat().st_size for row in rows)


def dev_package_rows(zip_path: Path) -> list[dict[str, Any]]:
    """Metadata-like rows of the dev package (read from its manifest) for the stats."""
    with zipfile.ZipFile(zip_path) as archive:
        manifest = VideoManifest.model_validate_json(archive.read(MANIFEST_FILENAME))
    return [
        {
            "issues": [i.model_dump(mode="json") for i in clip.issues],
            "decoys": [d.model_dump(mode="json") for d in clip.decoys],
            "difficulty": clip.difficulty,
            "duration": clip.duration,
        }
        for clip in manifest.clips
    ]


def issue_catalogue() -> dict[str, Any]:
    return {
        "suite_version": SUITE_VERSION,
        "time_convention": (
            "Frame i covers [i/fps, (i+1)/fps). A span over frames s..e inclusive has "
            "start_time = s/fps and end_time = (e+1)/fps; a point event (dropped_frames) at the cut "
            "before frame k has start_time == end_time == k/fps."
        ),
        "issue_types": [
            {
                "name": issue.name,
                "category": issue.category,
                "spatial": issue.spatial,
                "point_event": issue.point_event,
                "description": issue.description,
            }
            for issue in ISSUE_TYPES
        ],
        "decoy_types": [
            {"name": name, "description": DECOY_DESCRIPTIONS.get(name, "")} for name in DECOY_TYPES
        ],
    }


def _scoring_summary() -> str:
    try:
        from validator.modules.video_inconsistency.scoring import ScoringSettings

        settings = ScoringSettings()
        tious = getattr(settings, "tiou_thresholds", (0.3, 0.4, 0.5, 0.6, 0.7))
        weight_map = getattr(settings, "weight_map", 0.75)
        weight_loc = settings.weight_localization
        weight_clip = settings.weight_clip_accuracy
        bbox_iou = getattr(settings, "bbox_iou_threshold", 0.3)
        weights = dict(settings.difficulty_weights)
        cap = settings.max_predictions_per_clip
    except Exception:  # noqa: BLE001 - the card must still render if scoring changes
        tious, weight_map, weight_loc, weight_clip, bbox_iou, cap = (0.3, 0.4, 0.5, 0.6, 0.7), 0.75, 0.20, 0.05, 0.3, 25
        weights = {"easy": 1.0, "medium": 1.5, "hard": 2.0, "expert": 2.5}
    tiou_text = ", ".join(f"{v:g}" for v in tious)
    weight_text = ", ".join(f"{name} {value:g}" for name, value in weights.items())
    return (
        f"Detectors are scored with **mean average precision** (mAP): for every issue type, all of a "
        f"detector's predictions (ranked by their confidence, at most {cap} per clip) are matched to the "
        f"ground truth at temporal-IoU thresholds {tiou_text}, and the average precision is averaged over "
        f"thresholds and over the issue types. Spatial types (`inserted_object`, `blurred_region`) "
        f"additionally need a bounding-box IoU of at least {bbox_iou:g} to count as a match. "
        f"`score = {weight_map:g} * mean AP + {weight_loc:g} * localization + {weight_clip:g} * clip-level accuracy`; "
        f"clips are weighted by difficulty ({weight_text}). Confidences should be honest probabilities, "
        f"boundaries must be precise, and decoys (below) must not be reported."
    )


def _size_category(total_clips: int) -> str:
    for limit, label in ((1_000, "n<1K"), (10_000, "1K<n<10K"), (100_000, "10K<n<100K")):
        if total_clips < limit:
            return label
    return "100K<n<1M"


def render_dataset_card(stats: dict[str, Any], generation: dict[str, Any]) -> str:
    """The dataset README (with YAML front matter), built from the stats and generation params."""
    splits = stats["splits"]
    data_splits = [name for name in SPLITS if name in splits and splits[name]["clips"] > 0]
    total = sum(splits[name]["clips"] for name in data_splits)
    config_lines = "\n".join(
        f"  - split: {name}\n    path: {name}/**" for name in data_splits
    )
    tags = ["video", "synthetic", "video-forensics", "temporal-localization", "anomaly-detection"]
    front = (
        "---\n"
        "license: cc-by-4.0\n"
        "task_categories:\n- video-classification\n"
        "pretty_name: FLock Video Inconsistency\n"
        f"size_categories:\n- {_size_category(total)}\n"
        "tags:\n" + "".join(f"- {tag}\n" for tag in tags) +
        "configs:\n- config_name: default\n  data_files:\n" + config_lines + "\n"
        "---\n"
    )

    issue_rows = []
    for issue in ISSUE_TYPES:
        label = "point event" if issue.point_event else "span"
        if issue.spatial:
            label += " + bbox"
        counts = " / ".join(
            str(splits[name]["issues_per_type"][issue.name]) for name in data_splits
        )
        issue_rows.append(f"| `{issue.name}` | {issue.description} | {label} | {counts} |")
    decoy_rows = []
    for name in DECOY_TYPES:
        counts = " / ".join(str(splits[s]["decoys_per_type"][name]) for s in data_splits)
        decoy_rows.append(f"| `{name}` | {DECOY_DESCRIPTIONS.get(name, '')} | {counts} |")
    split_header = " / ".join(data_splits)
    difficulty_rows = []
    for name in DIFFICULTIES:
        counts = " / ".join(str(splits[s]["difficulty_histogram"][name]) for s in data_splits)
        difficulty_rows.append(f"| `{name}` | {DIFFICULTY_DESCRIPTIONS[name]} | {counts} |")
    size_rows = []
    for name in data_splits:
        info = splits[name]
        size_rows.append(
            f"| `{name}` | {info['clips']} | {info['clean_clips']} | {info['issues_total']} | "
            f"{info['decoys_total']} | {info['total_duration_seconds'] / 3600:.2f} h | {info['bytes'] / 1e9:.2f} GB |"
        )
    dev = splits.get("dev_package")
    dev_text = (
        f"`{DEV_PACKAGE_RELPATH}` is a validation package with {dev['clips']} clips built with seed "
        f"{generation['dev_package_seed']} by `package.build_validation_package`, in exactly the format "
        "the validator loads. Use it for local validation of a submission:\n\n"
        "```bash\npython environment_entrypoint.py video_inconsistency --local-validation \\\n"
        "    --hg-repo-id ./my_submission --validation-data-url ./video_inconsistency_dev_package.zip\n```\n\n"
        "It is disjoint from `train` and `validation` (separate seed namespace), so scores on it are honest."
        if dev else "No dev package is included in this build."
    )

    return f"""{front}
# FLock Video Inconsistency

Short procedurally generated videos (default 320x240 at 15 fps, {generation['min_duration']:g} to {generation['max_duration']:g} s), each a continuous
shot into which 0 to 4 known **inconsistencies** were injected, together with legitimate, unlabelled
**decoy** events that look like edits but are not. Every clip comes with frame-exact labels. The data
trains detectors for the FLock `video_inconsistency` task: given a video, list every inconsistency with
its type, time span, confidence and (for two types) a bounding box.

- Suite version: `{SUITE_VERSION}`; dataset build: `{DATASET_VERSION}`.
- License: CC-BY-4.0. All footage is synthetic{" (a fraction is edited from real footage)" if generation.get('footage_fraction') else ""}.
- Access: this dataset is private; trainers get access from the task organisers.

## Loading

```python
import json
from pathlib import Path
from huggingface_hub import snapshot_download

root = Path(snapshot_download("{DEFAULT_REPO_ID}", repo_type="dataset"))
rows = [json.loads(line) for line in (root / "train" / "metadata.jsonl").read_text().splitlines()]
first = rows[0]
video_path = root / "train" / first["file_name"]   # mp4 (H.264, yuv420p)
print(first["difficulty"], first["issues"], first["decoys"])
```

The trainer sample kit reads a split directly: `python train.py --data-dir <root>/train --out-dir runs/x`
(or `--data-dir <root>` to use `train/` for training and `validation/` for validation).

## Files

| Path | Content |
|------|---------|
| `train/metadata.jsonl`, `train/<clip_id>.mp4` | training split |
| `validation/metadata.jsonl`, `validation/<clip_id>.mp4` | validation split (labels included) |
| `dev_package/video_inconsistency_dev_package.zip` | package for local validation with the validator |
| `issue_types.json` | canonical issue and decoy catalogue |
| `stats.json` | counts per split |

Each metadata row: `file_name`, `clip_id`, `fps`, `num_frames`, `width`, `height`, `duration`, `difficulty`,
`source`, `crf` (H.264 quality the clip was encoded at), `issues` and `decoys`.
`issues` entries: `type`, `start_time`, `end_time`, `start_frame`, `end_frame`, `bbox` (spatial types only) and
`params` (how the edit was made). `decoys` entries: `type`, `start_time`, `end_time`.

### Time convention

Frame `i` covers `[i / fps, (i + 1) / fps)`. A span over frames `s..e` inclusive has `start_time = s / fps`
and `end_time = (e + 1) / fps` (`end_frame` is exclusive). A point event (`dropped_frames`) at the cut before
frame `k` has `start_time == end_time == k / fps` and `start_frame == end_frame == k`. Bounding boxes are
normalised `[x0, y0, x1, y1]` in `[0, 1]`; for a spatial issue the box is the union of the affected region
over its whole time span.

## The 10 issue types

Counts are {split_header}.

| Type | What to look for | Label | Count |
|------|------------------|-------|-------|
{chr(10).join(issue_rows)}

## Decoys (hard negatives)

Decoys are real events in the shot that a detector must **not** report. They are listed in `decoys`, are never
scored, and are never given to the detector at evaluation time. Use them as hard negatives.

| Decoy | Description | Count ({split_header}) |
|-------|-------------|-------|
{chr(10).join(decoy_rows)}

## Difficulty tiers

| Tier | Description | Clips ({split_header}) |
|------|-------------|-------|
{chr(10).join(difficulty_rows)}

Clips are also degraded after editing (sensor noise, optional blur or sharpening and down-up rescaling, then H.264 compression at a per-clip `crf`), so an edit must be
detectable after degradation. Every edit remains visible to a careful human looking at the frames.

## Split sizes

| Split | Clips | Clean clips | Issues | Decoys | Duration | Size |
|-------|-------|-------------|--------|--------|----------|------|
{chr(10).join(size_rows)}

## Generation

- Generator: `validator.modules.video_inconsistency.synthesis` (`generate_clip`), seed `{generation['seed']}`; each clip's
  seed is a hash of (split, seed, index, attempt), so `train` and `validation` are disjoint from each other and
  from the dev package (seed `{generation['dev_package_seed']}`).
- Resolution {generation['width']}x{generation['height']}, {generation['fps']:g} fps, {generation['min_duration']:g} to {generation['max_duration']:g} s per clip,
  fraction of clips edited from real footage: {generation['footage_fraction']:g}.
- Every mp4 was decoded back and verified to contain exactly `num_frames` frames.
- Rebuild: `python -m validator.modules.video_inconsistency.build_hf_dataset --out-dir <dir> --seed {generation['seed']}`.

## Local validation with the dev package

{dev_text}

## Scoring

{_scoring_summary()}

## The private evaluation set

The validators score submissions on a **private evaluation set generated with a different, secret seed**
(and, ideally, different content). It shares the generator and the label semantics with this dataset but no
clip. Do not expect a detector that memorises these clips to do well.
"""


# ---------------------------------------------------------------------------
# Building the whole dataset
# ---------------------------------------------------------------------------


def _dev_package_valid(path: Path, num_clips: int) -> bool:
    if not path.is_file():
        return False
    try:
        with zipfile.ZipFile(path) as archive:
            metadata = json.loads(archive.read("package.json"))
        return int(metadata.get("num_clips", -1)) == num_clips and metadata.get("suite_version") == SUITE_VERSION
    except (OSError, ValueError, KeyError, zipfile.BadZipFile):
        return False


def params_fingerprint(seed: int, config_kwargs: dict[str, Any]) -> str:
    """Hash of everything that determines the clips of a build (not counts or workers)."""
    payload = {
        "seed": seed,
        "config": {k: (sorted(v) if k == "footage_paths" else v) for k, v in sorted(config_kwargs.items())},
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _previous_generation(out_dir: Path) -> dict[str, Any]:
    try:
        return json.loads((out_dir / "stats.json").read_text(encoding="utf-8")).get("generation", {})
    except (OSError, ValueError):
        return {}


def _check_resume_params(out_dir: Path, fingerprint: str) -> None:
    """Refuse to resume into a folder that was built with different clip-defining parameters."""
    marker = out_dir / _BUILD_DIRNAME / "params.json"
    previous = None
    if marker.is_file():
        try:
            previous = json.loads(marker.read_text(encoding="utf-8")).get("fingerprint")
        except (OSError, ValueError):
            previous = None
    if previous is None:
        previous = _previous_generation(out_dir).get("fingerprint")
    if previous is not None and previous != fingerprint:
        raise ValueError(
            f"{out_dir} was built with different seed / size / duration / footage parameters; "
            "use a fresh --out-dir (or delete this one) instead of resuming"
        )
    _atomic_write(marker, json.dumps({"fingerprint": fingerprint}))


def build_dataset(
    out_dir: Path,
    *,
    train_clips: int,
    validation_clips: int,
    dev_package_clips: int,
    seed: int,
    dev_package_seed: int,
    workers: int,
    config_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Build all splits, the dev package and the metadata files; returns ``stats``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = params_fingerprint(seed, config_kwargs)
    previous = _previous_generation(out_dir)
    _check_resume_params(out_dir, fingerprint)
    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    for split, count in (("train", train_clips), ("validation", validation_clips)):
        if count > 0:
            rows_by_split[split] = build_split(out_dir, split, count, seed, config_kwargs, workers)

    splits_stats: dict[str, Any] = {
        split: compute_stats(rows, _split_bytes(out_dir, split, rows))
        for split, rows in rows_by_split.items()
    }
    if dev_package_clips > 0:
        dev_zip = out_dir / DEV_PACKAGE_RELPATH
        same_dev = previous.get("fingerprint") == fingerprint and previous.get("dev_package_seed") == dev_package_seed
        if not (same_dev and _dev_package_valid(dev_zip, dev_package_clips)):
            # The dev package is purely procedural (like the published one), whatever footage
            # the training splits use.
            dev_config = _config_from_kwargs({**config_kwargs, "footage_paths": (), "footage_fraction": 0.0})
            build_validation_package(dev_zip, num_clips=dev_package_clips, seed=dev_package_seed, config=dev_config)
        splits_stats["dev_package"] = compute_stats(dev_package_rows(dev_zip), dev_zip.stat().st_size)

    generation = {
        "fingerprint": fingerprint,
        "seed": seed,
        "dev_package_seed": dev_package_seed,
        "width": config_kwargs["width"],
        "height": config_kwargs["height"],
        "fps": config_kwargs["fps"],
        "min_duration": config_kwargs["min_duration"],
        "max_duration": config_kwargs["max_duration"],
        "footage_fraction": config_kwargs.get("footage_fraction", 0.0),
        "footage_files": len(config_kwargs.get("footage_paths", ())),
    }
    stats = {
        "dataset_version": DATASET_VERSION,
        "suite_version": SUITE_VERSION,
        "generation": generation,
        "splits": splits_stats,
    }
    _atomic_write(out_dir / "stats.json", json.dumps(stats, indent=2))
    _atomic_write(out_dir / "issue_types.json", json.dumps(issue_catalogue(), indent=2))
    _atomic_write(out_dir / "README.md", render_dataset_card(stats, generation))
    shutil.rmtree(out_dir / _BUILD_DIRNAME, ignore_errors=True)
    return stats


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------


def check_layout(out_dir: Path) -> None:
    required = ["README.md", "issue_types.json", "stats.json", "train/metadata.jsonl"]
    missing = [name for name in required if not (out_dir / name).is_file()]
    if missing:
        raise HubUploadError(f"{out_dir} is not a built dataset folder (missing: {', '.join(missing)})")


def push_to_hub(out_dir: Path, repo_id: str, *, private: bool = True) -> tuple[str, bool]:
    """Create the dataset repo if needed and upload ``out_dir``; returns ``(url, is_private)``.

    The token is taken from the ``HF_TOKEN`` environment variable only.
    """
    import huggingface_hub
    from huggingface_hub.utils import RepositoryNotFoundError

    check_layout(out_dir)
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise HubUploadError("set the HF_TOKEN environment variable (the token is never taken from arguments)")
    api = huggingface_hub.HfApi(token=token)

    try:
        existing = api.repo_info(repo_id=repo_id, repo_type="dataset")
    except RepositoryNotFoundError:
        existing = None
    if existing is not None and private and not bool(getattr(existing, "private", False)):
        raise HubUploadError(
            f"dataset repo {repo_id} already exists and is PUBLIC, but a private upload was requested; "
            "refusing. Make the repo private on the Hub, or pass --public if that is intended."
        )
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)

    upload = getattr(api, "upload_large_folder", None)
    if upload is not None:
        upload(repo_id=repo_id, repo_type="dataset", folder_path=str(out_dir), ignore_patterns=_UPLOAD_IGNORE)
    else:  # very old huggingface_hub: single-commit upload, not resumable
        logger.warning("huggingface_hub has no upload_large_folder; falling back to upload_folder")
        api.upload_folder(
            repo_id=repo_id, repo_type="dataset", folder_path=str(out_dir), ignore_patterns=_UPLOAD_IGNORE
        )

    try:
        info = api.repo_info(repo_id=repo_id, repo_type="dataset")
        is_private = bool(getattr(info, "private", private))
    except Exception:  # noqa: BLE001 - the report of visibility is best effort
        is_private = private
    return f"https://huggingface.co/datasets/{repo_id}", is_private


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    defaults = SynthesisConfig()
    parser = argparse.ArgumentParser(
        prog="build_hf_dataset",
        description="Build the video-inconsistency training dataset (Hugging Face videofolder layout) "
        "and optionally upload it as a dataset repo.",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--train-clips", type=int, default=8000)
    parser.add_argument("--validation-clips", type=int, default=1000)
    parser.add_argument("--dev-package-clips", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--dev-package-seed", type=int, default=7)
    parser.add_argument("--workers", type=int, default=max(1, min(8, mp.cpu_count())))
    parser.add_argument("--width", type=int, default=defaults.width)
    parser.add_argument("--height", type=int, default=defaults.height)
    parser.add_argument("--fps", type=float, default=defaults.fps)
    parser.add_argument("--min-duration", type=float, default=defaults.min_duration)
    parser.add_argument("--max-duration", type=float, default=defaults.max_duration)
    parser.add_argument("--footage-dir", help="directory of real videos to edit (optional)")
    parser.add_argument("--footage-fraction", type=float, default=None,
                        help="share of train/validation clips from --footage-dir (default 0.5 with a dir)")
    parser.add_argument("--skip-build", action="store_true", help="upload an existing --out-dir as is")
    parser.add_argument("--push-to-hub", metavar="REPO_ID", default=None,
                        help="upload to this dataset repo (token from the HF_TOKEN environment variable)")
    visibility = parser.add_mutually_exclusive_group()
    visibility.add_argument("--private", dest="public", action="store_false", help="private repo (default)")
    visibility.add_argument("--public", dest="public", action="store_true", help="public repo")
    parser.set_defaults(public=False)
    return parser.parse_args(argv)


def _config_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    footage: tuple[str, ...] = ()
    fraction = 0.0
    if args.footage_dir:
        footage = collect_footage(args.footage_dir)
        if not footage:
            raise SystemExit(f"no video files found under {args.footage_dir}")
        fraction = 0.5 if args.footage_fraction is None else args.footage_fraction
    elif args.footage_fraction:
        raise SystemExit("--footage-fraction needs --footage-dir")
    kwargs = {
        "width": args.width,
        "height": args.height,
        "fps": args.fps,
        "min_duration": args.min_duration,
        "max_duration": args.max_duration,
        "footage_paths": footage,
        "footage_fraction": fraction,
    }
    try:
        _config_from_kwargs(kwargs)  # validate early
    except ValueError as exc:
        raise SystemExit(f"invalid configuration: {exc}") from exc
    return kwargs


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    out_dir = Path(args.out_dir)
    if not args.skip_build:
        if args.train_clips <= 0:
            raise SystemExit("--train-clips must be positive")
        if min(args.validation_clips, args.dev_package_clips) < 0:
            raise SystemExit("clip counts must not be negative")
        stats = build_dataset(
            out_dir,
            train_clips=args.train_clips,
            validation_clips=args.validation_clips,
            dev_package_clips=args.dev_package_clips,
            seed=args.seed,
            dev_package_seed=args.dev_package_seed,
            workers=max(1, args.workers),
            config_kwargs=_config_kwargs(args),
        )
        for name, info in stats["splits"].items():
            print(f"{name}: {info['clips']} clips ({info['clean_clips']} clean), "
                  f"{info['issues_total']} issues, {info['decoys_total']} decoys, {info['bytes'] / 1e6:.1f} MB")
        print(f"dataset written to {out_dir}")
    if args.push_to_hub:
        try:
            url, is_private = push_to_hub(out_dir, args.push_to_hub, private=not args.public)
        except HubUploadError as exc:
            print(f"upload refused: {exc}", file=sys.stderr)
            return 2
        print(f"repo: {url}")
        print(f"visibility: {'private' if is_private else 'PUBLIC'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
