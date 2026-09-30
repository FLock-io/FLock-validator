"""Tests for the video_inconsistency trainer sample kit (features, decoder, localiser, adapter, training)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

SAMPLE_DIR = (
    Path(__file__).resolve().parents[1] / "validator" / "modules" / "video_inconsistency" / "trainer_sample"
)
if str(SAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(SAMPLE_DIR))

import vic_features  # noqa: E402
import vic_localize  # noqa: E402
import vic_model  # noqa: E402

from validator.modules.video_inconsistency.issue_types import ISSUE_TYPE_NAMES  # noqa: E402


def _moving_clip(num_frames: int = 30, height: int = 48, width: int = 64, seed: int = 0) -> np.ndarray:
    """A textured background panning to the right, so frames differ but stay coherent."""
    rng = np.random.default_rng(seed)
    base = rng.integers(60, 200, (height, width * 2, 3), dtype=np.uint8)
    return np.stack([base[:, t : t + width] for t in range(num_frames)]).astype(np.uint8)


def _smooth_clip(num_frames: int = 40, height: int = 96, width: int = 128) -> np.ndarray:
    """A smooth colour field drifting slowly: realistic frame-to-frame change, no random texture."""
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    frames = []
    for t in range(num_frames):
        phase = 0.05 * t
        channels = [
            128 + 60 * np.sin(xs / 17.0 + phase + k) * np.cos(ys / 23.0 - 0.5 * phase + 2 * k)
            for k in range(3)
        ]
        frames.append(np.stack(channels, axis=-1))
    return np.clip(np.stack(frames), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------------------
def test_issue_type_order_matches_validator() -> None:
    assert vic_model.ISSUE_TYPE_NAMES == ISSUE_TYPE_NAMES


@pytest.mark.parametrize("shape", [(30, 48, 64, 3), (2, 33, 47, 3), (5, 7, 9, 3), (1, 16, 16, 3)])
def test_features_shape_and_finite(shape: tuple[int, ...]) -> None:
    frames = np.random.default_rng(0).integers(0, 256, shape, dtype=np.uint8)
    feats = vic_features.extract_features(frames)
    assert feats.shape == (shape[0], vic_features.NUM_FEATURES)
    assert feats.dtype == np.float32
    assert np.isfinite(feats).all()


def test_features_constant_and_black_frames_are_finite() -> None:
    for value in (0, 128, 255):
        frames = np.full((20, 32, 48, 3), value, dtype=np.uint8)
        assert np.isfinite(vic_features.extract_features(frames)).all()


def test_freeze_is_visible_in_features() -> None:
    frames = _moving_clip(40)
    frames[15:25] = frames[14]
    feats = vic_features.extract_features(frames)
    dup = feats[:, vic_features.FEATURE_NAMES.index("dup_prev")]
    assert dup[15:25].min() > 0.9
    assert dup[:14].max() < 0.5


def test_normalisation_roundtrip() -> None:
    feats = [vic_features.extract_features(_moving_clip(20, seed=s)) for s in range(3)]
    stats = vic_features.compute_feature_stats(feats)
    normed = vic_features.normalize_features(feats[0], stats)
    assert normed.shape == feats[0].shape and np.isfinite(normed).all()
    assert np.abs(normed).max() <= 10.0


def test_feature_extraction_speed_is_reasonable() -> None:
    import time

    frames = np.random.default_rng(1).integers(0, 256, (150, 240, 320, 3), dtype=np.uint8)
    start = time.time()
    vic_features.extract_features(frames)
    assert time.time() - start < 6.0  # ~0.5 s on a laptop; generous for CI


# ---------------------------------------------------------------------------------------
# decoder
# ---------------------------------------------------------------------------------------
def _curve(length: int = 60) -> np.ndarray:
    return np.zeros((length, len(ISSUE_TYPE_NAMES)))


def test_decode_span_and_time_convention() -> None:
    probs = _curve()
    probs[10:20, ISSUE_TYPE_NAMES.index("zoom_jump")] = 0.9
    issues = vic_model.decode_intervals(probs, fps=10.0, duration=6.0)
    assert len(issues) == 1
    issue = issues[0]
    assert issue["type"] == "zoom_jump"
    assert issue["start_time"] == pytest.approx(1.0)
    assert issue["end_time"] == pytest.approx(2.0)  # frame 19 inclusive -> 20 / fps
    assert issue["confidence"] == pytest.approx(0.9, abs=1e-6)


def test_decode_merges_gap_of_one_frame_only() -> None:
    probs = _curve()
    column = ISSUE_TYPE_NAMES.index("mirrored_segment")
    probs[5:10, column] = 0.9
    probs[11:16, column] = 0.9  # gap of one frame -> merged
    probs[30:35, column] = 0.9
    probs[37:42, column] = 0.9  # gap of two frames -> separate
    issues = [i for i in vic_model.decode_intervals(probs, 10.0, 6.0) if i["type"] == "mirrored_segment"]
    spans = sorted((i["_start_frame"], i["_end_frame"]) for i in issues)
    assert spans == [(5, 16), (30, 35), (37, 42)]


def test_decode_drops_short_runs_but_keeps_short_flicker() -> None:
    probs = _curve()
    probs[10:12, ISSUE_TYPE_NAMES.index("color_grade_jump")] = 0.9  # 2 frames < min 3
    probs[20:21, ISSUE_TYPE_NAMES.index("exposure_flicker")] = 0.9  # 1 frame is allowed
    types = {i["type"] for i in vic_model.decode_intervals(probs, 10.0, 6.0)}
    assert types == {"exposure_flicker"}


def test_decode_dropped_frames_is_a_point_event() -> None:
    probs = _curve()
    column = ISSUE_TYPE_NAMES.index("dropped_frames")
    probs[29, column] = 0.9  # model fires on frames k-1 and k, with the cut at k = 30
    probs[30, column] = 0.9
    (issue,) = vic_model.decode_intervals(probs, 10.0, 6.0)
    assert issue["type"] == "dropped_frames"
    assert issue["start_time"] == issue["end_time"] == pytest.approx(3.0)


def test_decode_emits_candidates_below_half_without_clipping() -> None:
    probs = _curve()
    probs[10:20, 0] = 0.32  # below the default threshold 0.5 but above the candidate threshold
    (issue,) = vic_model.decode_intervals(probs, 10.0, 6.0)
    assert issue["confidence"] == pytest.approx(0.32, abs=1e-6)  # 0.5 * 0.32 / 0.5, not clipped to 0.5
    assert (issue["_start_frame"], issue["_end_frame"]) == (10, 20)
    # a calibrated threshold of 0.3 maps that same run to a confident detection
    (issue,) = vic_model.decode_intervals(probs, 10.0, 6.0, thresholds={"frozen_frames": 0.3})
    assert issue["confidence"] >= 0.5
    # below the candidate floor nothing is reported at all
    probs[10:20, 0] = 0.03
    assert vic_model.decode_intervals(probs, 10.0, 6.0) == []


def test_calibrated_confidence_is_monotone_and_maps_threshold_to_half() -> None:
    for threshold in (0.15, 0.4, 0.5, 0.8):
        values = [vic_model.calibrated_confidence(p, threshold) for p in np.linspace(0, 1, 41)]
        assert all(b >= a - 1e-12 for a, b in zip(values, values[1:]))
        assert vic_model.calibrated_confidence(threshold, threshold) == pytest.approx(0.5)
        assert vic_model.calibrated_confidence(1.0, threshold) == pytest.approx(1.0)
        assert min(values) >= vic_model.CONFIDENCE_FLOOR


def test_decode_candidates_do_not_overlap_confident_runs_and_are_trimmed() -> None:
    probs = _curve(100)
    column = ISSUE_TYPE_NAMES.index("zoom_jump")
    probs[20:30, column] = 0.9
    probs[30:36, column] = 0.3  # shoulder of the confident run: must not become a candidate
    probs[60:70, column] = 0.21  # wide weak plateau...
    probs[64:68, column] = 0.45  # ...with a peak: the candidate is trimmed to >= half the peak
    issues = vic_model.decode_intervals(probs, 10.0, 10.0)
    spans = sorted((i["_start_frame"], i["_end_frame"], round(i["confidence"], 2)) for i in issues)
    assert spans[0][:2] == (20, 30) and spans[0][2] >= 0.5
    assert len(spans) == 2
    assert spans[1][:2] == (64, 68) and spans[1][2] < 0.5


def test_decode_off_type_is_candidates_only() -> None:
    probs = _curve()
    probs[10:20, 0] = 0.9
    issues = vic_model.decode_intervals(probs, 10.0, 6.0, thresholds={"frozen_frames": 1.0})
    assert issues and all(i["confidence"] < 0.5 for i in issues)
    assert vic_model.decode_intervals(probs, 10.0, 6.0, thresholds={"frozen_frames": 0.5})[0]["confidence"] == pytest.approx(0.9)


def test_dilation_schedule() -> None:
    assert vic_model.dilations_for_layers(6) == (1, 2, 4, 8, 16, 32)
    assert vic_model.dilations_for_layers(8) == (1, 2, 4, 8, 16, 32, 1, 2)
    with pytest.raises(ValueError):
        vic_model.dilations_for_layers(0)


def test_decode_caps_to_top_25_by_confidence() -> None:
    probs = np.zeros((500, len(ISSUE_TYPE_NAMES)))
    for n in range(40):
        probs[n * 12 : n * 12 + 5, 0] = 0.6 + 0.009 * n
    issues = vic_model.decode_intervals(probs, 10.0, 50.0)
    assert len(issues) == 25 == vic_model.MAX_ISSUES
    assert min(i["confidence"] for i in issues) >= 0.6 + 0.009 * 15 - 1e-6  # the 15 weakest were cut


def test_decode_rejects_bad_shape() -> None:
    with pytest.raises(ValueError):
        vic_model.decode_intervals(np.zeros((10, 3)), 10.0, 1.0)


# ---------------------------------------------------------------------------------------
# localisation
# ---------------------------------------------------------------------------------------
def _assert_valid_box(box: list[float]) -> None:
    assert len(box) == 4 and all(0.0 <= v <= 1.0 for v in box)
    assert box[0] < box[2] and box[1] < box[3]


def test_localize_inserted_object_finds_the_patch() -> None:
    frames = _smooth_clip(40, 96, 128)
    frames[10:25, 20:44, 70:100] = 255
    truth = [70 / 128, 20 / 96, 100 / 128, 44 / 96]
    box = vic_localize.localize("inserted_object", frames, 10, 25)
    _assert_valid_box(box)
    assert vic_localize.box_iou(box, truth) > 0.5


def test_localize_blurred_region_finds_the_smooth_patch() -> None:
    rng = np.random.default_rng(3)
    frames = rng.integers(0, 256, (40, 96, 128, 3), dtype=np.uint8)
    frames[10:25, 16:64, 32:80] = 120  # texture-free patch, like a strong blur
    truth = [32 / 128, 16 / 96, 80 / 128, 64 / 96]
    box = vic_localize.localize("blurred_region", frames, 10, 25)
    _assert_valid_box(box)
    assert vic_localize.box_iou(box, truth) > 0.4


def test_localize_never_raises_and_returns_none_for_other_types() -> None:
    rng = np.random.default_rng(4)
    frames = rng.integers(0, 256, (12, 20, 20, 3), dtype=np.uint8)
    for issue_type in ("inserted_object", "blurred_region"):
        for start, end in [(0, 12), (0, 1), (11, 12), (5, 5), (-3, 400)]:
            _assert_valid_box(vic_localize.localize(issue_type, frames, start, end))
    _assert_valid_box(vic_localize.localize("inserted_object", np.zeros((1, 8, 8, 3), np.uint8), 0, 1))
    assert vic_localize.localize("zoom_jump", frames, 2, 5) is None


# ---------------------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------------------
def test_model_forward_shapes_and_mask() -> None:
    torch = pytest.importorskip("torch")
    model = vic_model.TemporalIssueNet(in_features=vic_features.NUM_FEATURES, hidden=16)
    x = torch.randn(2, 37, vic_features.NUM_FEATURES)
    mask = torch.ones(2, 37)
    mask[1, 20:] = 0
    out = model(x, mask)
    assert out.shape == (2, 37, len(ISSUE_TYPE_NAMES))
    assert model(x).shape == out.shape
    assert vic_model.model_hyperparameters(model)["hidden"] == 16


# ---------------------------------------------------------------------------------------
# adapter + training (need the synthesiser / video I/O)
# ---------------------------------------------------------------------------------------
def _synthesis():
    return pytest.importorskip("validator.modules.video_inconsistency.synthesis")


def _video_dict(clip) -> dict:
    frames = clip.frames
    return {
        "frames": frames, "frames_path": "", "video_path": "", "fps": float(clip.fps),
        "num_frames": int(frames.shape[0]), "width": int(frames.shape[2]), "height": int(frames.shape[1]),
        "duration": frames.shape[0] / float(clip.fps), "issue_types": list(ISSUE_TYPE_NAMES),
    }


def _short_config(synthesis):
    return synthesis.SynthesisConfig(min_duration=3.0, max_duration=4.0)


def test_heuristic_adapter_end_to_end(tmp_path: Path) -> None:
    synthesis = _synthesis()
    predictions = pytest.importorskip("validator.modules.video_inconsistency.predictions")
    import flock_video_adapter as adapter

    detector = adapter.load_detector(str(tmp_path), "cpu", "float32")  # empty dir -> heuristic
    assert isinstance(detector, adapter.HeuristicDetector)
    for seed in (5, 6, 7):
        clip = synthesis.generate_clip(seed, _short_config(synthesis))
        raw = detector.detect(_video_dict(clip))
        json.dumps(raw)  # plain Python types only
        parsed = predictions.parse_detector_output(raw, clip.frames.shape[0] / clip.fps)
        assert all(0.0 <= p.confidence <= 1.0 for p in parsed)
        for issue in raw["issues"]:
            assert not any(key.startswith("_") for key in issue)
            assert (issue.get("bbox") is not None) == (issue["type"] in vic_model.SPATIAL_TYPES)


def test_heuristic_adapter_finds_a_frozen_segment(tmp_path: Path) -> None:
    synthesis = _synthesis()
    import flock_video_adapter as adapter

    clip = synthesis.generate_clip(11, _short_config(synthesis), issue_types=["frozen_frames"], difficulty="easy")
    raw = adapter.load_detector(str(tmp_path), "cpu", "float32").detect(_video_dict(clip))
    truth = clip.issues[0]
    hits = [i for i in raw["issues"] if i["type"] == "frozen_frames"]
    assert hits, "the frozen segment should be found by the rule-based detector"
    overlap = min(hits[0]["end_time"], truth.end_time) - max(hits[0]["start_time"], truth.start_time)
    assert overlap > 0


def test_train_package_and_load_learned_detector(tmp_path: Path) -> None:
    synthesis = _synthesis()
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    pytest.importorskip("imageio_ffmpeg")
    predictions = pytest.importorskip("validator.modules.video_inconsistency.predictions")
    import flock_video_adapter as adapter
    import generate_data
    import package_submission
    import train

    data_dir, run_dir, sub_dir = tmp_path / "data", tmp_path / "run", tmp_path / "submission"
    assert generate_data.main(
        ["--out-dir", str(data_dir), "--num-clips", "6", "--seed", "3", "--workers", "1"]
    ) == 0
    labels = [json.loads(line) for line in (data_dir / "labels.jsonl").read_text().splitlines()]
    assert len(labels) == 6 and all((data_dir / r["clip_file"]).is_file() for r in labels)
    # dev-package seeds are small; training seeds must stay clear of them
    assert all(r["clip_seed"] >= generate_data.SEED_OFFSET for r in labels)

    assert train.main(
        ["--data-dir", str(data_dir), "--out-dir", str(run_dir), "--epochs", "2", "--batch-size", "3",
         "--val-fraction", "0.34", "--workers", "1", "--seed", "0"]
    ) == 0
    assert (run_dir / "weights.safetensors").is_file()
    config = json.loads((run_dir / "vic_config.json").read_text())
    assert config["issue_types"] == list(ISSUE_TYPE_NAMES)
    assert set(config["thresholds"]) == set(ISSUE_TYPE_NAMES)
    assert len(config["feature_stats"]["mean"]) == vic_features.NUM_FEATURES
    assert list((data_dir / "feature_cache").glob("*.npz")), "features should be cached"

    package_submission.build_submission(run_dir, sub_dir)
    assert {p.name for p in sub_dir.iterdir()} >= {
        "flock_video_adapter.py", "vic_features.py", "vic_model.py", "vic_localize.py",
        "weights.safetensors", "vic_config.json", "README.md",
    }
    detector = adapter.load_detector(str(sub_dir), "cpu", "bfloat16")
    assert isinstance(detector, adapter.LearnedDetector)
    clip = synthesis.generate_clip(21, _short_config(synthesis))
    raw = detector.detect(_video_dict(clip))
    json.dumps(raw)
    predictions.parse_detector_output(raw, clip.frames.shape[0] / clip.fps)

    # the packaged folder must work from a fresh interpreter with a scrubbed environment
    package_submission.smoke_test(sub_dir)


def test_build_targets_conventions() -> None:
    torch = pytest.importorskip("torch")  # train.py imports torch at module level
    del torch
    import train

    record = {
        "issues": [
            {"type": "zoom_jump", "start_frame": 5, "end_frame": 9},
            {"type": "dropped_frames", "start_frame": 20, "end_frame": 20},
        ]
    }
    targets = train.build_targets(record, 30)
    zoom = targets[:, ISSUE_TYPE_NAMES.index("zoom_jump")]
    drop = targets[:, ISSUE_TYPE_NAMES.index("dropped_frames")]
    assert np.flatnonzero(zoom).tolist() == [5, 6, 7, 8]  # [start, end)
    assert np.flatnonzero(drop).tolist() == [19, 20]  # k-1 and k


def test_build_frame_weights_upweights_decoys() -> None:
    pytest.importorskip("torch")
    import train

    record = {"fps": 10.0, "decoys": [{"type": "scene_cut", "start_time": 2.0, "end_time": 3.0}]}
    weights = train.build_frame_weights(record, 60, 3.0)
    assert weights[:17].max() == 1.0 and weights[33:].max() == 1.0
    assert weights[20:30].min() == 3.0
    assert train.build_frame_weights(record, 60, 1.0).max() == 1.0
    assert train.build_frame_weights({"fps": 10.0}, 60, 3.0).max() == 1.0  # v1-style rows have no decoys


def _fake_record(index: int) -> dict:
    return {
        "clip_file": f"videos/c{index}.mp4", "fps": 10.0, "num_frames": 40, "width": 64, "height": 48,
        "difficulty": "expert", "issues": [], "decoys": [],
    }


def test_read_labels_detects_both_layouts_and_dataset_root(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    import train

    own = tmp_path / "own"
    own.mkdir()
    (own / "labels.jsonl").write_text("\n".join(json.dumps(_fake_record(i)) for i in range(2)) + "\n")
    rows = train.read_labels(own)
    assert [r["clip_file"] for r in rows] == ["videos/c0.mp4", "videos/c1.mp4"]
    assert all(r["_root"] == str(own) and r["decoys"] == [] for r in rows)

    root = tmp_path / "hf"
    for split, count in (("train", 3), ("validation", 2)):
        (root / split).mkdir(parents=True)
        lines = []
        for i in range(count):
            row = _fake_record(i)
            row["file_name"] = f"clip{i}.mp4"
            del row["clip_file"]
            lines.append(json.dumps(row))
        (root / split / "metadata.jsonl").write_text("\n".join(lines) + "\n")
    hf_rows = train.read_labels(root / "train")
    assert [r["clip_file"] for r in hf_rows] == ["clip0.mp4", "clip1.mp4", "clip2.mp4"]
    assert train.resolve_splits(root, None) == (root / "train", root / "validation")
    assert train.resolve_splits(root / "train", None) == (root / "train", None)
    assert train.resolve_splits(own, None) == (own, None)
    with pytest.raises(FileNotFoundError):
        train.read_labels(tmp_path)


def test_train_reads_hf_split_layout_and_supports_layers(tmp_path: Path) -> None:
    _synthesis()
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    pytest.importorskip("imageio_ffmpeg")
    import generate_data
    import train

    data_dir = tmp_path / "data"
    assert generate_data.main(
        ["--out-dir", str(data_dir), "--num-clips", "6", "--seed", "4", "--workers", "1",
         "--width", "96", "--height", "64"]
    ) == 0
    rows = [json.loads(line) for line in (data_dir / "labels.jsonl").read_text().splitlines()]
    assert all("decoys" in r and "crf" in r for r in rows)

    # re-express the data as a Hugging Face split (metadata.jsonl + flat mp4 files)
    root = tmp_path / "hf"
    for split, part in (("train", rows[:4]), ("validation", rows[4:])):
        (root / split).mkdir(parents=True)
        with open(root / split / "metadata.jsonl", "w") as handle:
            for r in part:
                name = Path(r["clip_file"]).name
                (root / split / name).write_bytes((data_dir / r["clip_file"]).read_bytes())
                handle.write(json.dumps({**{k: v for k, v in r.items() if k != "clip_file"}, "file_name": name}) + "\n")

    run_dir = tmp_path / "run"
    assert train.main(
        ["--data-dir", str(root), "--out-dir", str(run_dir), "--epochs", "1", "--batch-size", "2",
         "--workers", "1", "--hidden", "8", "--layers", "8", "--calib-max-clips", "2"]
    ) == 0
    config = json.loads((run_dir / "vic_config.json").read_text())
    assert config["model"]["hidden"] == 8 and config["model"]["dilations"] == [1, 2, 4, 8, 16, 32, 1, 2]
    assert config["train"]["num_train_clips"] == 4 and config["train"]["num_val_clips"] == 2
    assert (root / "train" / "feature_cache").is_dir()
