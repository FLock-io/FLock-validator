from __future__ import annotations

import dataclasses
import json
import math
import stat
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from validator.modules.video_inconsistency import package as package_module
from validator.modules.video_inconsistency import synthesis as syn
from validator.modules.video_inconsistency.issue_types import (
    ISSUE_TYPE_NAMES,
    POINT_EVENT_ISSUE_TYPES,
    SPATIAL_ISSUE_TYPES,
)
from validator.modules.video_inconsistency.manifest import (
    DIFFICULTIES,
    ClipSpec,
    DecoyLabel,
    IssueLabel,
)
from validator.modules.video_inconsistency.package import (
    build_validation_package,
    resolve_validation_package,
)
from validator.modules.video_inconsistency.synthesis import (
    SynthesisConfig,
    generate_clip,
    render_procedural_scene,
)
from validator.modules.video_inconsistency.video_io import (
    decode_video,
    encode_video,
    probe_video,
)


FPS = 15.0
# 4 s exactly: 60 frames, small enough to keep the suite fast.
SMALL = SynthesisConfig(width=160, height=120, fps=FPS, min_duration=4.0, max_duration=4.0)
# Undegraded and decoy-free: lets tests compare the output with the raw source shot.
RAW = dataclasses.replace(SMALL, degradation=False, decoy_rate=0.0)
GAP_FRAMES = math.ceil(1.0 * FPS)
MARGIN_FRAMES = math.ceil(0.5 * FPS)


def _mean_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.astype(np.int16) - b.astype(np.int16))))


def _pan_seed(start: int) -> int:
    """First seed >= start whose scene is a normal-looking panning shot (motion for the guards)."""
    for seed in range(start, start + 200):
        spec = syn._sample_scene(syn._rng(seed, "scene"), SMALL.width, SMALL.height, FPS)
        if spec.camera_mode == "pan" and spec.style.brightness == 1.0 and spec.style.contrast == 1.0:
            return seed
    raise AssertionError("no panning scene found")


def _source_for(seed: int, config: SynthesisConfig, num_frames: int) -> np.ndarray:
    return render_procedural_scene(
        syn._rng(seed, "scene"), num_frames, config.width, config.height, config.fps
    )


# ---------------------------------------------------------------------------
# Determinism and scene rendering
# ---------------------------------------------------------------------------


def test_generate_clip_is_deterministic() -> None:
    first = generate_clip(11, SMALL, issue_types=["zoom_jump", "exposure_flicker"])
    second = generate_clip(11, SMALL, issue_types=["zoom_jump", "exposure_flicker"])
    other = generate_clip(12, SMALL, issue_types=["zoom_jump", "exposure_flicker"])
    assert np.array_equal(first.frames, second.frames)
    assert first.issues == second.issues
    assert first.difficulty == second.difficulty
    assert first.decoys == second.decoys and first.crf == second.crf
    assert not np.array_equal(first.frames, other.frames)
    # Sampled decoys and degradation are part of the deterministic output too.
    busy = dataclasses.replace(SMALL, decoy_rate=4.0)
    a, b = generate_clip(5, busy), generate_clip(5, busy)
    assert np.array_equal(a.frames, b.frames) and a.decoys == b.decoys and a.crf == b.crf


@pytest.mark.parametrize("seed", [3, 4, 5, 6])
def test_scene_prefix_is_stable_and_frames_never_repeat(seed: int) -> None:
    long = render_procedural_scene(np.random.default_rng(seed), 40, 160, 120, FPS)
    short = render_procedural_scene(np.random.default_rng(seed), 25, 160, 120, FPS)
    assert long.shape == (40, 120, 160, 3) and long.dtype == np.uint8
    assert np.array_equal(long[:25], short)
    # Sensor noise: consecutive real frames are never identical (even on a static tripod).
    assert all(not np.array_equal(long[i], long[i + 1]) for i in range(39))


def test_scene_realism_camera_modes_and_object_behaviour() -> None:
    specs = [syn._sample_scene(np.random.default_rng(i), 160, 120, FPS) for i in range(60)]
    assert {spec.camera_mode for spec in specs} == set(syn._CAMERA_MODES)
    by_mode = {mode: next(spec for spec in specs if spec.camera_mode == mode) for mode in syn._CAMERA_MODES}
    spread = {}
    for mode, spec in by_mode.items():
        cx, cy, zoom = syn._camera_arrays(spec, 90)
        spread[mode] = float(np.std(cx) + np.std(cy))
    assert spread["static"] == 0.0  # tripod: only the objects move
    assert spread["pan"] > 8.0 and spread["drift"] > 0.5
    handheld = by_mode["handheld"]
    cx, _cy, _z = syn._camera_arrays(handheld, 90)
    assert float(np.std(np.diff(cx))) > 0.3  # frame-to-frame jitter on top of the pan
    # Objects: mix of slow and fast movers, some that pause, some that wrap across the edges.
    objects = [obj for spec in specs for obj in spec.objects]
    speeds = np.array([float(np.hypot(*obj.vel)) for obj in objects])
    assert speeds.min() < 28.0 and speeds.max() > 90.0
    assert any(obj.pauses for obj in objects) and any(obj.behavior == "wrap" for obj in objects)
    styles = [spec.style for spec in specs]
    assert any(st.contrast < 0.7 for st in styles) and any(st.brightness < 0.8 for st in styles)
    assert any(st.symmetry > 0.4 for st in styles)
    # A paused object really stands still.
    obj = dataclasses.replace(objects[0], pauses=((0.4, 0.9),), vel=np.array([60.0, 0.0]), behavior="wrap")
    tracks, _clocks = syn._object_tracks([obj], 30, 240, 180, FPS)
    assert np.allclose(tracks[7:13, 0, 0], tracks[7, 0, 0]) and tracks[20, 0, 0] != tracks[13, 0, 0]


def test_default_config_clip_dimensions_and_length() -> None:
    config = SynthesisConfig(min_duration=6.0, max_duration=10.0)
    clip = generate_clip(5, config)
    assert clip.frames.shape[1:] == (240, 320, 3)
    assert 90 <= len(clip.frames) <= 150
    assert clip.source == "procedural"
    assert clip.frames.flags["C_CONTIGUOUS"]


# ---------------------------------------------------------------------------
# Every editor at every difficulty
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ISSUE_TYPE_NAMES)
@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_single_editor_label_is_valid(name: str, difficulty: str) -> None:
    seed = 1000 + ISSUE_TYPE_NAMES.index(name) * 10 + DIFFICULTIES.index(difficulty)
    clip = generate_clip(seed, SMALL, issue_types=[name], difficulty=difficulty)
    assert clip.difficulty == difficulty
    assert len(clip.frames) == 60  # output length is as planned even when frames are dropped
    assert len(clip.issues) == 1
    label = clip.issues[0]
    assert label.type == name
    assert 0 <= label.start_frame <= label.end_frame <= len(clip.frames)
    assert label.start_time == pytest.approx(label.start_frame / FPS)
    assert label.end_time == pytest.approx(label.end_frame / FPS)
    assert label.start_frame >= MARGIN_FRAMES
    assert len(clip.frames) - label.end_frame >= MARGIN_FRAMES
    if name in POINT_EVENT_ISSUE_TYPES:
        assert label.start_frame == label.end_frame
        assert label.start_time == label.end_time
        dropped = label.params["dropped_frames"]
        expected = {"expert": (1, 1), "hard": (1, 2)}.get(difficulty)
        assert (expected[0] <= dropped <= expected[1]) if expected else dropped >= 3
    else:
        assert label.end_frame > label.start_frame
    if name in SPATIAL_ISSUE_TYPES:
        assert label.bbox is not None
        x0, y0, x1, y1 = label.bbox
        assert 0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0
    else:
        assert label.bbox is None
    # Round trips through pydantic (manifest validators).
    assert IssueLabel.model_validate(label.model_dump()) == label


_TIER_FRAMES = {  # (issue, tier) -> (min frames, max frames) of the edit window
    ("frozen_frames", "hard"): (2, 3), ("frozen_frames", "expert"): (2, 2),
    ("spliced_footage", "hard"): (2, 4), ("spliced_footage", "expert"): (2, 3),
    ("reversed_segment", "hard"): (4, 8), ("reversed_segment", "expert"): (3, 6),  # 0.3-0.5 s / 0.2-0.35 s
    ("mirrored_segment", "hard"): (4, 10), ("mirrored_segment", "expert"): (4, 8),  # 0.3-0.6 s / 0.3-0.5 s
    ("exposure_flicker", "hard"): (1, 1), ("exposure_flicker", "expert"): (1, 1),
}


@pytest.mark.parametrize("difficulty", ["hard", "expert"])
@pytest.mark.parametrize("name", ISSUE_TYPE_NAMES)
def test_subtle_tier_parameters_match_the_spec(name: str, difficulty: str) -> None:
    seed = 3000 + ISSUE_TYPE_NAMES.index(name)
    config = dataclasses.replace(SMALL, min_duration=6.0, max_duration=6.0)
    clip = generate_clip(seed, config, issue_types=[name], difficulty=difficulty)
    assert clip.difficulty == difficulty and len(clip.issues) == 1
    label = clip.issues[0]
    frames = label.end_frame - label.start_frame
    if (name, difficulty) in _TIER_FRAMES:
        low, high = _TIER_FRAMES[(name, difficulty)]
        assert low <= frames <= high
    p = label.params
    expert = difficulty == "expert"
    if name == "color_grade_jump":
        if p["mode"] == "hue":
            assert (3.0 <= abs(p["hue_deg"]) <= 5.0) if expert else (4.0 <= abs(p["hue_deg"]) <= 8.0)
        else:
            gains = np.array([p["gain_r"], p["gain_g"], p["gain_b"]])
            assert float(gains.max() - gains.min()) <= (0.04 if expert else 0.06) + 1e-6
    elif name == "exposure_flicker":
        ranges = ((1.04, 1.07), (0.93, 0.96)) if expert else ((1.08, 1.12), (0.88, 0.92))
        assert any(lo <= p["factor"] <= hi for lo, hi in ranges)
    elif name == "zoom_jump":
        assert (1.02 <= p["scale"] <= 1.05) if expert else (1.03 <= p["scale"] <= 1.08)
    elif name == "inserted_object":
        fractions = (0.03, 0.05) if expert else (0.04, 0.06)
        assert max(8.0, fractions[0] * 160) - 1e-6 <= p["size_px"] <= max(8.0, fractions[1] * 160) + 1e-6
        assert p["drift"] == "follows_scene"  # subtle sprites move with the scene
    elif name == "blurred_region":
        x0, y0, x1, y1 = label.bbox
        assert (x1 - x0) * 160 >= 11.9  # 12 px minimum box on this tiny frame
        if p["mode"] == "gaussian":
            assert (1.0 <= p["sigma"] <= 1.5) if expert else (1.2 <= p["sigma"] <= 2.0)
        else:
            assert p["block"] == 2 if expert else 2 <= p["block"] <= 3
    elif name == "dropped_frames":
        assert p["dropped_frames"] == 1 if expert else 1 <= p["dropped_frames"] <= 2


def _round_sources(seed: int, config: SynthesisConfig, num_out: int, name: str, difficulty: str):
    """The source shot of every scene round the generator may have used for this clip."""
    extra = syn._max_dropped_frames(difficulty, config.fps) if name == "dropped_frames" else 0
    hints = syn._scene_hints([name], difficulty)  # motion-dependent edits get a moving scene
    for salt in range(syn._SOURCE_ROUNDS):
        yield syn._build_source(
            seed, config, num_out + extra, num_out, hints=hints, forced=None, salt=salt
        ).frames


@pytest.mark.parametrize("name", ISSUE_TYPE_NAMES)
def test_frames_outside_the_edit_are_untouched(name: str) -> None:
    seed = 2000 + ISSUE_TYPE_NAMES.index(name)
    clip = generate_clip(seed, RAW, issue_types=[name], difficulty="medium")
    assert clip.decoys == [] and clip.crf == 18
    label = clip.issues[0]
    dropped = int(label.params.get("dropped_frames", 0))
    before, after_out = label.start_frame, label.end_frame
    matches = 0
    for source in _round_sources(seed, RAW, len(clip.frames), name, "medium"):
        if not np.array_equal(clip.frames[:before], source[:before]):
            continue  # the generator drew a fresh scene when this one could not show the edit
        assert np.array_equal(clip.frames[after_out:], source[after_out + dropped : len(clip.frames) + dropped])
        if name == "dropped_frames":
            # The cut removes exactly `dropped` source frames.
            assert not np.array_equal(clip.frames[before], source[before])
        matches += 1
    assert matches == 1


def test_editor_semantics() -> None:
    rng = np.random.default_rng(0)
    seed = _pan_seed(4)
    src = _source_for(seed, RAW, 60)
    ctx = syn._EditContext(config=RAW, donor=syn._make_donor(seed, RAW))
    a, b = 20, 35

    def run(name: str, difficulty: str = "easy") -> syn._EditOutcome:
        for _ in range(20):
            outcome = syn.EDITORS[name](rng, src, a, b, difficulty, ctx)
            if outcome is not None:
                return outcome
        raise AssertionError(f"{name} never passed its detectability guard")

    frozen = run("frozen_frames")
    assert all(np.array_equal(frame, src[a]) for frame in frozen.frames)

    assert np.array_equal(run("reversed_segment").frames, src[a:b][::-1])
    assert np.array_equal(run("mirrored_segment").frames, src[a:b, :, ::-1])
    assert len(run("dropped_frames").frames) == 0

    splice = run("spliced_footage").frames
    assert splice.shape == src[a:b].shape and _mean_abs(splice[0], src[a - 1]) > 12.0

    flicker = run("exposure_flicker")
    n = len(flicker.frames)
    assert abs(float(flicker.frames.mean()) - float(src[a : a + n].mean())) >= 8.0

    grade = run("color_grade_jump")
    assert _mean_abs(grade.frames, src[a:b]) >= 4.0
    assert _mean_abs(run("zoom_jump").frames, src[a:b]) >= 8.0

    for name in ("inserted_object", "blurred_region"):
        outcome = run(name)
        height, width = src.shape[1:3]
        x0, y0, x1, y1 = outcome.bbox
        changed = np.abs(outcome.frames.astype(np.int16) - src[a:b].astype(np.int16)).max(axis=(0, 3))
        ys, xs = np.nonzero(changed)
        assert len(ys) > 0
        # Every changed pixel lies inside the labelled box (1 px tolerance).
        assert xs.min() >= math.floor(x0 * width) - 1 and xs.max() <= math.ceil(x1 * width)
        assert ys.min() >= math.floor(y0 * height) - 1 and ys.max() <= math.ceil(y1 * height)


def test_color_grade_strength_scales_with_difficulty() -> None:
    seed = _pan_seed(9)
    src = _source_for(seed, RAW, 60)
    ctx = syn._EditContext(config=RAW, donor=syn._make_donor(seed, RAW))
    rng = np.random.default_rng(1)
    strengths = {}
    for difficulty in DIFFICULTIES:
        deltas = []
        for _ in range(16):
            outcome = syn.EDITORS["color_grade_jump"](rng, src, 20, 35, difficulty, ctx)
            if outcome is not None:
                deltas.append(_mean_abs(outcome.frames, src[20:35]))
        assert deltas, f"no {difficulty} grade passed its guard"
        strengths[difficulty] = float(np.mean(deltas))
    assert strengths["easy"] > strengths["hard"] > strengths["expert"] > 0


# ---------------------------------------------------------------------------
# Clean clips and multi-issue layout
# ---------------------------------------------------------------------------


def test_clean_clips_have_no_issues() -> None:
    config = SynthesisConfig(
        width=160, height=120, min_duration=4.0, max_duration=5.0, clean_fraction=1.0
    )
    for seed in range(3):
        clip = generate_clip(seed, config)
        assert clip.issues == []
    # And the clean fraction is honoured statistically.
    mixed = SynthesisConfig(width=160, height=120, min_duration=4.0, max_duration=4.0, clean_fraction=0.5)
    clean = sum(1 for seed in range(30) if not generate_clip(seed, mixed).issues)
    assert 5 <= clean <= 25


def test_multi_issue_clips_respect_gaps_and_ordering() -> None:
    config = SynthesisConfig(
        width=160,
        height=120,
        min_duration=8.0,
        max_duration=9.0,
        clean_fraction=0.0,
        difficulty_weights=(("hard", 1.0),),
    )
    multi = 0
    for seed in range(8):
        clip = generate_clip(seed, config)
        total = len(clip.frames)
        issues = clip.issues
        assert issues, "hard clips with clean_fraction=0 always carry issues"
        assert 1 <= len(issues) <= 3
        assert len({issue.type for issue in issues}) == len(issues)  # sampled without replacement
        assert [i.start_frame for i in issues] == sorted(i.start_frame for i in issues)
        assert issues[0].start_frame >= MARGIN_FRAMES
        assert total - max(i.end_frame for i in issues) >= MARGIN_FRAMES
        for prev, nxt in zip(issues, issues[1:]):
            assert nxt.start_frame - prev.end_frame >= GAP_FRAMES
        multi += len(issues) > 1
    assert multi >= 3


def test_difficulty_controls_issue_count() -> None:
    easy = SynthesisConfig(
        width=160, height=120, min_duration=6.0, max_duration=6.0, clean_fraction=0.0,
        difficulty_weights=(("easy", 1.0),),
    )
    assert all(len(generate_clip(s, easy).issues) == 1 for s in range(4))


def test_expert_clips_carry_two_to_four_spaced_issues() -> None:
    config = SynthesisConfig(
        width=160, height=120, min_duration=8.0, max_duration=9.0, clean_fraction=0.0,
        difficulty_weights=(("expert", 1.0),),
    )
    counts = []
    for seed in range(8):
        clip = generate_clip(seed, config)
        assert clip.difficulty == "expert"
        issues = clip.issues
        counts.append(len(issues))
        assert len(issues) <= 4
        assert len({issue.type for issue in issues}) == len(issues)
        for prev, nxt in zip(issues, issues[1:]):
            assert nxt.start_frame - prev.end_frame >= GAP_FRAMES
        assert issues[0].start_frame >= MARGIN_FRAMES if issues else True
    assert max(counts) >= 3 and min(counts) >= 1
    assert sum(1 for c in counts if c >= 2) >= 5  # single-issue expert clips only as a fallback


def test_invalid_arguments_are_rejected() -> None:
    with pytest.raises(ValueError):
        generate_clip(0, SMALL, issue_types=["nope"])
    with pytest.raises(ValueError):
        generate_clip(0, SMALL, difficulty="impossible")
    with pytest.raises(ValueError):
        SynthesisConfig(width=161)
    with pytest.raises(ValueError):
        SynthesisConfig(issue_types=("nope",))
    with pytest.raises(ValueError):
        generate_clip(0, SMALL, decoys=["nope"])
    with pytest.raises(ValueError):
        SynthesisConfig(decoy_rate=-1.0)
    with pytest.raises(ValueError):
        SynthesisConfig(crf_range=(30, 20))
    with pytest.raises(ValueError):
        SynthesisConfig(difficulty_weights=(("nightmare", 1.0),))
    assert SynthesisConfig().difficulty_weights == (
        ("easy", 0.10), ("medium", 0.25), ("hard", 0.35), ("expert", 0.30),
    )


# ---------------------------------------------------------------------------
# video_io and footage
# ---------------------------------------------------------------------------


def test_video_io_round_trip(tmp_path: Path) -> None:
    frames = _source_for(21, RAW, 37)  # odd count on purpose
    path = encode_video(frames, tmp_path / "clip.mp4", FPS)
    decoded = decode_video(path)
    assert decoded.shape == frames.shape and decoded.dtype == np.uint8
    assert decoded.flags["C_CONTIGUOUS"]
    assert _mean_abs(decoded, frames) < 6.0
    info = probe_video(path)
    assert (info.num_frames, info.width, info.height) == (37, 160, 120)
    assert info.fps == pytest.approx(FPS)
    assert decode_video(path, max_frames=5).shape[0] == 5


def test_video_io_error_handling(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        decode_video(tmp_path / "missing.mp4")
    garbage = tmp_path / "garbage.mp4"
    garbage.write_bytes(b"not a video at all" * 100)
    with pytest.raises(ValueError):
        decode_video(garbage)
    with pytest.raises(ValueError):
        probe_video(garbage)
    with pytest.raises(ValueError):
        encode_video(np.zeros((2, 5, 6, 3), dtype=np.uint8), tmp_path / "odd.mp4", 15.0)


def test_encoder_is_byte_reproducible(tmp_path: Path) -> None:
    frames = _source_for(22, RAW, 20)
    first = encode_video(frames, tmp_path / "a.mp4", FPS).read_bytes()
    second = encode_video(frames, tmp_path / "b.mp4", FPS).read_bytes()
    assert first == second
    assert encode_video(frames, tmp_path / "c.mp4", FPS, crf=18).read_bytes() == first


def test_encode_crf_controls_quality(tmp_path: Path) -> None:
    frames = _source_for(23, RAW, 30)
    sizes = {}
    errors = {}
    for crf in (10, 18, 28, 40):
        path = encode_video(frames, tmp_path / f"crf{crf}.mp4", FPS, crf=crf)
        sizes[crf] = path.stat().st_size
        errors[crf] = _mean_abs(decode_video(path), frames)
    assert sizes[10] > sizes[18] > sizes[28] > sizes[40]
    assert errors[10] < errors[40]
    for bad in (-1, 52, 18.5, True):
        with pytest.raises(ValueError):
            encode_video(frames, tmp_path / "bad.mp4", FPS, crf=bad)


def test_real_footage_source_and_fallback(tmp_path: Path) -> None:
    footage = tmp_path / "real.mp4"
    encode_video(_source_for(31, SynthesisConfig(width=240, height=136), 90), footage, FPS)
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"\x00" * 64)

    config = SynthesisConfig(
        width=160, height=120, min_duration=3.0, max_duration=3.0,
        footage_paths=(str(footage),), footage_fraction=1.0,
    )
    clip = generate_clip(3, config, issue_types=["mirrored_segment"])
    assert clip.source == "footage" and clip.frames.shape == (45, 120, 160, 3)
    assert len(clip.issues) == 1
    again = generate_clip(3, config, issue_types=["mirrored_segment"])
    assert np.array_equal(clip.frames, again.frames)

    fallback = SynthesisConfig(
        width=160, height=120, min_duration=3.0, max_duration=3.0,
        footage_paths=(str(broken),), footage_fraction=1.0,
    )
    assert generate_clip(3, fallback).source == "procedural"
    # Footage too short for the clip also falls back.
    tiny = tmp_path / "tiny.mp4"
    encode_video(_source_for(32, SynthesisConfig(width=240, height=136), 10), tiny, FPS)
    short = SynthesisConfig(
        width=160, height=120, min_duration=3.0, max_duration=3.0,
        footage_paths=(str(tiny),), footage_fraction=1.0,
    )
    assert generate_clip(3, short).source == "procedural"


# ---------------------------------------------------------------------------
# Packages
# ---------------------------------------------------------------------------

PKG_CONFIG = SynthesisConfig(width=160, height=120, min_duration=4.0, max_duration=5.0, decoy_rate=3.0)


@pytest.fixture(scope="module")
def built_package(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("video_pkg")
    return build_validation_package(directory / "pkg.zip", num_clips=4, seed=77, config=PKG_CONFIG)


def test_build_and_resolve_package(built_package: Path, tmp_path: Path) -> None:
    resolved = resolve_validation_package(str(built_package), tmp_path / "cache")
    manifest = resolved.manifest
    assert len(manifest.clips) == 4
    assert len({clip.clip_id for clip in manifest.clips}) == 4
    for clip in manifest.clips:
        assert len(clip.clip_id) == 12
        path = resolved.clip_video_path(clip)
        assert path.is_file()
        assert probe_video(path).num_frames == clip.num_frames
        assert clip.video_path == f"videos/{clip.clip_id}.mp4"
    diagnostics = resolved.diagnostics
    for key in (
        "video_package_source", "video_package_sha256", "num_clips", "suite_version", "issue_type_counts",
        "decoy_type_counts", "difficulty_counts",
    ):
        assert key in diagnostics
    assert diagnostics["num_clips"] == "4"
    assert isinstance(json.loads(diagnostics["issue_type_counts"]), dict)
    assert sum(json.loads(diagnostics["difficulty_counts"]).values()) == 4
    assert sum(json.loads(diagnostics["decoy_type_counts"]).values()) == sum(
        len(clip.decoys) for clip in manifest.clips
    )
    with zipfile.ZipFile(built_package) as archive:
        names = archive.namelist()
        assert names == sorted(names)
        metadata = json.loads(archive.read("package.json"))
    assert metadata["package_version"] == "video_inconsistency_package_v1"
    assert metadata["num_clips"] == 4


def test_package_round_trips_decoys_and_records_each_clips_crf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    used: list[int] = []
    real_encode = package_module.encode_video

    def spy(frames, path, fps, *, crf=18):  # noqa: ANN001
        used.append(crf)
        return real_encode(frames, path, fps, crf=crf)

    monkeypatch.setattr(package_module, "encode_video", spy)
    config = dataclasses.replace(PKG_CONFIG, decoy_rate=6.0)
    zip_path = build_validation_package(tmp_path / "pkg.zip", num_clips=5, seed=123, config=config)
    resolved = resolve_validation_package(str(zip_path), tmp_path / "cache")
    expected = [generate_clip(package_module._clip_seed(123, index, 0), config) for index in range(5)]
    manifest_crfs = [clip.crf for clip in resolved.manifest.clips]
    assert manifest_crfs == used == [clip.crf for clip in expected]  # recorded == actually encoded
    assert len(set(manifest_crfs)) > 1 and all(18 <= crf <= 28 for crf in manifest_crfs)
    manifest_decoys = [clip.decoys for clip in resolved.manifest.clips]
    assert manifest_decoys == [clip.decoys for clip in expected]
    assert sum(len(d) for d in manifest_decoys) >= 5
    assert all(isinstance(d, DecoyLabel) for decoys in manifest_decoys for d in decoys)
    assert [clip.difficulty for clip in resolved.manifest.clips] == [clip.difficulty for clip in expected]


def test_package_build_is_byte_reproducible(built_package: Path, tmp_path: Path) -> None:
    again = build_validation_package(tmp_path / "again.zip", num_clips=4, seed=77, config=PKG_CONFIG)
    assert again.read_bytes() == built_package.read_bytes()
    other = build_validation_package(tmp_path / "other.zip", num_clips=2, seed=78, config=PKG_CONFIG)
    assert other.read_bytes() != built_package.read_bytes()


def test_clip_video_path_rejects_escape(built_package: Path, tmp_path: Path) -> None:
    resolved = resolve_validation_package(str(built_package), tmp_path / "cache")
    clip = resolved.manifest.clips[0]
    evil = ClipSpec(**{**clip.model_dump(), "video_path": "../../etc/passwd"})
    with pytest.raises(ValueError):
        resolved.clip_video_path(evil)


def _rewrite_zip(source: Path, target: Path, *, drop=(), replace=None, extra=()) -> Path:
    replace = replace or {}
    with zipfile.ZipFile(source) as src, zipfile.ZipFile(target, "w") as out:
        for info in src.infolist():
            if info.filename in drop:
                continue
            out.writestr(info.filename, replace.get(info.filename, src.read(info.filename)))
        for info, data in extra:
            out.writestr(info, data)
    return target


@pytest.mark.parametrize("member", ["../evil.txt", "/abs/evil.txt", "a/../../evil.txt", "C:evil.txt"])
def test_zip_slip_and_absolute_entries_are_rejected(built_package: Path, tmp_path: Path, member: str) -> None:
    bad = _rewrite_zip(built_package, tmp_path / "bad.zip", extra=[(member, b"x")])
    with pytest.raises(ValueError):
        resolve_validation_package(str(bad), tmp_path / "cache")
    assert not (tmp_path / "evil.txt").exists()


def test_symlink_entries_are_rejected(built_package: Path, tmp_path: Path) -> None:
    link = zipfile.ZipInfo("videos/link.mp4")
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    bad = _rewrite_zip(built_package, tmp_path / "link.zip", extra=[(link, b"/etc/passwd")])
    with pytest.raises(ValueError, match="regular file"):
        resolve_validation_package(str(bad), tmp_path / "cache")


def test_duplicate_entries_and_non_zip_are_rejected(built_package: Path, tmp_path: Path) -> None:
    dup = tmp_path / "dup.zip"
    with zipfile.ZipFile(dup, "w") as out:
        out.writestr("package.json", "{}")
        with pytest.warns(UserWarning):
            out.writestr("package.json", "{}")
    with pytest.raises(ValueError):
        resolve_validation_package(str(dup), tmp_path / "cache")
    junk = tmp_path / "junk.zip"
    junk.write_bytes(b"this is not a zip")
    with pytest.raises(ValueError):
        resolve_validation_package(str(junk), tmp_path / "cache")


def test_too_many_members_rejected(built_package: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(package_module, "MAX_ARCHIVE_MEMBERS", 3)
    with pytest.raises(ValueError, match="too many"):
        resolve_validation_package(str(built_package), tmp_path / "cache")


def test_uncompressed_size_cap(built_package: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(package_module, "MAX_UNCOMPRESSED_BYTES", 1000)
    with pytest.raises(ValueError, match="exceeds"):
        resolve_validation_package(str(built_package), tmp_path / "cache")


def test_missing_video_file_is_rejected(built_package: Path, tmp_path: Path) -> None:
    with zipfile.ZipFile(built_package) as archive:
        victim = next(name for name in archive.namelist() if name.endswith(".mp4"))
    bad = _rewrite_zip(built_package, tmp_path / "novideo.zip", drop={victim})
    with pytest.raises(FileNotFoundError):
        resolve_validation_package(str(bad), tmp_path / "cache")


def test_bad_metadata_and_manifest_are_rejected(built_package: Path, tmp_path: Path) -> None:
    wrong_version = _rewrite_zip(
        built_package, tmp_path / "v.zip",
        replace={"package.json": json.dumps({"package_version": "nope"}).encode()},
    )
    with pytest.raises(ValueError, match="package_version"):
        resolve_validation_package(str(wrong_version), tmp_path / "cache")

    no_meta = _rewrite_zip(built_package, tmp_path / "m.zip", drop={"package.json"})
    with pytest.raises(FileNotFoundError):
        resolve_validation_package(str(no_meta), tmp_path / "cache")

    bad_manifest = _rewrite_zip(
        built_package, tmp_path / "bm.zip", replace={"manifest.json": b'{"clips": []}'}
    )
    with pytest.raises(ValueError):
        resolve_validation_package(str(bad_manifest), tmp_path / "cache")

    escaping = _rewrite_zip(
        built_package, tmp_path / "esc.zip",
        replace={"package.json": json.dumps(
            {"package_version": "video_inconsistency_package_v1", "manifest_path": "../manifest.json"}
        ).encode()},
    )
    with pytest.raises(ValueError):
        resolve_validation_package(str(escaping), tmp_path / "cache")


def test_missing_local_path_and_bad_scheme(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        resolve_validation_package(str(tmp_path / "nope.zip"), tmp_path / "cache")
    with pytest.raises(ValueError):
        resolve_validation_package("ftp://example.com/pkg.zip", tmp_path / "cache")


class _FakeResponse:
    def __init__(self, data: bytes, headers: dict[str, str] | None = None) -> None:
        self._data = data
        self.headers = headers or {}

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, chunk_size: int):
        for start in range(0, len(self._data), chunk_size):
            yield self._data[start : start + chunk_size]


def test_http_download_and_size_cap(
    built_package: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = built_package.read_bytes()
    calls: list[str] = []

    def fake_get(url: str, **kwargs: object) -> _FakeResponse:
        calls.append(url)
        assert kwargs.get("stream") is True and kwargs.get("timeout")
        return _FakeResponse(data)

    monkeypatch.setattr(package_module.requests, "get", fake_get)
    url = "https://example.com/pkg.zip"
    resolved = resolve_validation_package(url, tmp_path / "cache")
    assert calls == [url]
    assert resolved.diagnostics["video_package_source"] == url
    assert len(resolved.manifest.clips) == 4
    assert not list((tmp_path / "cache").glob("*.part"))

    monkeypatch.setattr(package_module, "MAX_DOWNLOAD_BYTES", 1000)
    with pytest.raises(ValueError, match="size cap"):
        resolve_validation_package("https://example.com/other.zip", tmp_path / "cache2")
    assert not list((tmp_path / "cache2").glob("*.part"))


def test_build_package_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from validator.modules.video_inconsistency.build_package import main

    output = tmp_path / "cli.zip"
    code = main(
        [
            "--output", str(output), "--num-clips", "2", "--seed", "5",
            "--width", "160", "--height", "120", "--min-duration", "4", "--max-duration", "4",
            "--issue-types", "frozen_frames,zoom_jump", "--clean-fraction", "0",
        ]
    )
    assert code == 0 and output.is_file()
    out = capsys.readouterr().out
    assert "sha256:" in out and "frozen_frames" in out
    assert "difficulty:" in out and "expert:" in out and "decoys:" in out and "scene_cut:" in out


# ---------------------------------------------------------------------------
# Decoys: legitimate, unlabelled events
# ---------------------------------------------------------------------------

DECOY_CONFIG = SynthesisConfig(width=160, height=120, fps=FPS, min_duration=6.0, max_duration=6.0)
_DECOY_SECONDS = {  # decoy -> (min, max) seconds of its labelled interval (None = unbounded)
    "exposure_drift": (1.0, None),
    "white_balance_drift": (1.5, None),
    "smooth_zoom": (1.5, None),
    "camera_stops": (1.0, None),
    "object_stops": (0.5, None),
    "illumination_flicker": (2.0, None),
    "auto_exposure_step": (5 / FPS, 10 / FPS),
    "auto_white_balance_step": (5 / FPS, 10 / FPS),
    "fast_zoom": (0.4, 0.8),
    "camera_direction_change": (0.3, 0.6),
    "camera_speed_change": (0.2, 0.4),
}


@pytest.mark.parametrize("decoy", syn.DECOY_TYPES)
def test_every_decoy_can_be_forced_without_creating_an_issue(decoy: str) -> None:
    clean_only = dataclasses.replace(DECOY_CONFIG, clean_fraction=1.0)
    for seed in range(3):
        clip = generate_clip(seed, clean_only, decoys=[decoy])
        assert clip.issues == []  # a clean clip with a decoy is a pure hard negative
        assert [d.type for d in clip.decoys] == [decoy]
        label = clip.decoys[0]
        assert isinstance(label, DecoyLabel)
        assert 0.0 <= label.start_time <= label.end_time <= len(clip.frames) / FPS + 1e-9
        if decoy == "scene_cut":
            assert label.start_time == label.end_time
            assert 1.0 - 1e-9 <= label.start_time <= len(clip.frames) / FPS - 1.0 + 1e-9
        elif decoy in ("object_enters", "object_exits"):
            assert 2 / FPS - 1e-9 <= label.end_time - label.start_time <= 26 / FPS
        else:
            low, high = _DECOY_SECONDS[decoy]
            length = label.end_time - label.start_time
            assert length >= low - 1e-9 and (high is None or length <= high + 1.5 / FPS)


def test_decoy_effects_are_visible_in_the_frames() -> None:
    config = dataclasses.replace(DECOY_CONFIG, clean_fraction=1.0, degradation=False)
    for seed in range(4):
        clip = generate_clip(seed, config, decoys=["scene_cut"])
        index = int(round(clip.decoys[0].start_time * FPS))
        steps = [_mean_abs(clip.frames[i], clip.frames[i + 1]) for i in range(len(clip.frames) - 1)]
        assert steps[index - 1] > 10.0 and steps[index - 1] == max(steps)  # a hard cut: a different shot
        drift = generate_clip(seed, config, decoys=["exposure_drift"])
        window = drift.decoys[0]
        a, b = int(round(window.start_time * FPS)), int(round(window.end_time * FPS))
        means = [float(f.mean()) for f in drift.frames]
        assert max(means[a : b + 1]) - min(means[a : b + 1]) > 0.03 * means[a]
        # The ramp is smooth: no frame-to-frame brightness jump anywhere inside it.
        assert max(abs(means[i + 1] - means[i]) for i in range(a, b - 1)) < 0.06 * means[a]


def test_forcing_decoys_keeps_edits_and_none_disables_them() -> None:
    clip = generate_clip(9, DECOY_CONFIG, issue_types=["zoom_jump", "color_grade_jump"], difficulty="hard",
                         decoys=["exposure_drift", "smooth_zoom"])
    assert {d.type for d in clip.decoys} == {"exposure_drift", "smooth_zoom"}
    assert {i.type for i in clip.issues} == {"zoom_jump", "color_grade_jump"}
    assert generate_clip(9, DECOY_CONFIG, decoys=[]).decoys == []
    silent = dataclasses.replace(DECOY_CONFIG, decoy_rate=0.0)
    assert all(generate_clip(s, silent).decoys == [] for s in range(6))
    busy = dataclasses.replace(DECOY_CONFIG, decoy_rate=4.0)
    counts = [len(generate_clip(s, busy).decoys) for s in range(10)]
    assert np.mean(counts) > 2.0
    # Clean and edited clips both get decoys at the same rate.
    default = dataclasses.replace(DECOY_CONFIG, decoy_rate=2.0)
    clean = [len(generate_clip(s, dataclasses.replace(default, clean_fraction=1.0)).decoys) for s in range(12)]
    edited = [len(generate_clip(s, dataclasses.replace(default, clean_fraction=0.0)).decoys) for s in range(12)]
    assert np.mean(clean) > 0.8 and np.mean(edited) > 0.8


def test_scene_cut_never_inside_or_near_an_edit_window() -> None:
    config = SynthesisConfig(width=160, height=120, fps=FPS, min_duration=8.0, max_duration=9.0, clean_fraction=0.0,
                             difficulty_weights=(("hard", 1.0),))
    clearance = math.ceil(0.5 * FPS)
    checked = 0
    for seed in range(14):
        clip = generate_clip(seed, config, decoys=["scene_cut"])
        cuts = [d for d in clip.decoys if d.type == "scene_cut"]
        assert len(cuts) == 1 and clip.issues
        cut = int(round(cuts[0].start_time * FPS))
        for issue in clip.issues:
            assert cut <= issue.start_frame - clearance or cut >= issue.end_frame + clearance, (seed, issue.type)
            checked += 1
    assert checked >= 14


def test_splice_returns_to_the_original_shot_even_with_a_scene_cut() -> None:
    config = SynthesisConfig(width=160, height=120, fps=FPS, min_duration=8.0, max_duration=8.0, clean_fraction=0.0,
                             degradation=False)
    for seed in range(6):
        clip = generate_clip(seed, config, issue_types=["spliced_footage"], difficulty="medium", decoys=["scene_cut"])
        label = clip.issues[0]
        cut = int(round(clip.decoys[0].start_time * FPS))
        a, b = label.start_frame, label.end_frame
        steps = [_mean_abs(clip.frames[i], clip.frames[i + 1]) for i in range(len(clip.frames) - 1)]
        # The three biggest jumps are into the splice, out of it, and the legitimate cut.
        biggest = {int(i) + 1 for i in np.argsort(steps)[-3:]}
        assert biggest == {a, b, cut}


def test_camera_stop_eases_the_camera_without_a_jump() -> None:
    spec = next(
        sp for sp in (syn._sample_scene(np.random.default_rng(i), 160, 120, FPS) for i in range(80))
        if sp.camera_mode == "pan"
    )
    cx, cy, _zoom = syn._camera_arrays(spec, 90, [("camera_stops", 30, 60, 1.0)])
    fx, fy, _ = syn._camera_arrays(spec, 90)
    assert np.allclose(cx[34:56], cx[34]) and np.allclose(cy[34:56], cy[34])  # at rest
    assert np.abs(np.diff(fx[34:56])).max() > 0.2  # whereas the unedited camera keeps moving
    assert np.abs(np.diff(cx)).max() < 4.0  # no jump anywhere: a legitimate ease


# ---------------------------------------------------------------------------
# Degradation
# ---------------------------------------------------------------------------


def test_degradation_removes_bit_identical_frozen_frames_only_when_enabled() -> None:
    raw = dataclasses.replace(SMALL, degradation=False, decoy_rate=0.0, min_duration=6.0, max_duration=6.0)
    noisy = dataclasses.replace(raw, degradation=True)
    for seed in range(5):
        for config, expect_identical in ((raw, True), (noisy, False)):
            clip = generate_clip(seed, config, issue_types=["frozen_frames"], difficulty="medium")
            label = clip.issues[0]
            pairs = [
                np.array_equal(clip.frames[i], clip.frames[i + 1])
                for i in range(label.start_frame, label.end_frame - 1)
            ]
            assert pairs and (all(pairs) if expect_identical else not any(pairs))


def test_degradation_parameters_and_crf() -> None:
    off = dataclasses.replace(SMALL, degradation=False, crf_range=(30, 40))
    assert generate_clip(1, off).crf == 18
    on = dataclasses.replace(SMALL, crf_range=(22, 26))
    crfs, sigmas = set(), []
    for seed in range(40):
        deg, crf = syn._sample_degradation(seed, on)
        assert deg is not None and 22 <= crf <= 26 and 1.0 <= deg.sigma <= 4.0
        assert deg.scale == 1.0 or 0.6 <= deg.scale <= 1.0
        crfs.add(crf)
        sigmas.append(deg.sigma)
    assert crfs == {22, 23, 24, 25, 26} and max(sigmas) - min(sigmas) > 1.5
    # Measured noise on a flat frame matches sigma, and every frame gets independent noise.
    deg = syn._Degradation(sigma=3.0, seed=1)
    flat = np.full((6, 64, 64, 3), 100, dtype=np.uint8)
    out = syn._degrade(flat, deg)
    assert abs(float(out.astype(float).std()) - 3.0) < 0.3
    assert not np.array_equal(out[0], out[1])
    assert syn._degrade(flat, None) is flat
    # Rescale acts on structure, not just noise.
    textured = _source_for(7, RAW, 4)
    soft = syn._degrade_det(textured, syn._Degradation(sigma=0.0, scale=0.6))
    assert _mean_abs(soft, textured) > 0.5 and soft.shape == textured.shape


def _pooled(frame: np.ndarray, k: int = 4) -> np.ndarray:
    h, w = frame.shape[0] // k * k, frame.shape[1] // k * k
    return frame[:h, :w].astype(np.float32).reshape(h // k, k, w // k, k, 3).mean(axis=(1, 3))


def _pooled_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(_pooled(a) - _pooled(b)).mean())


def test_guards_still_hold_on_the_degraded_frames_of_expert_clips() -> None:
    config = SynthesisConfig(width=160, height=120, fps=FPS, min_duration=6.0, max_duration=6.0)
    checked: Counter = Counter()
    for seed in range(60):
        for name in ("exposure_flicker", "color_grade_jump", "frozen_frames", "dropped_frames"):
            if checked[name] >= 4:
                continue
            clip = generate_clip(seed * 7 + ISSUE_TYPE_NAMES.index(name), config, issue_types=[name],
                                 difficulty="expert", decoys=[])
            frames, label = clip.frames, clip.issues[0]
            means = np.array([f.reshape(-1, 3).mean(axis=0) for f in frames])
            if name == "exposure_flicker":
                i = label.start_frame
                neighbours = 0.5 * (means[i - 1].mean() + means[i + 1].mean())
                assert abs(means[i].mean() - neighbours) >= 2.2  # guard: >= 3 levels before noise
            elif name == "color_grade_jump":
                step_in = np.abs(means[label.start_frame] - means[label.start_frame - 1]).max()
                step_out = np.abs(means[label.end_frame] - means[label.end_frame - 1]).max()
                assert max(step_in, step_out) >= 1.0
            elif name == "frozen_frames":
                s, e = label.start_frame, label.end_frame
                held = max(_pooled_diff(frames[i], frames[i + 1]) for i in range(s, e - 1))
                jump = _pooled_diff(frames[e - 1], frames[e])
                assert jump > held + 0.15  # motion stops (noise floor only), then resumes with a jump
            else:
                k = label.start_frame
                jump = _pooled_diff(frames[k - 1], frames[k])
                normal = np.median([_pooled_diff(frames[i], frames[i + 1]) for i in range(k - 6, k - 1)])
                assert jump > 1.1 * normal
            checked[name] += 1
    assert min(checked.values()) >= 4


# ---------------------------------------------------------------------------
# Scene internals: similar donors, sprite motion
# ---------------------------------------------------------------------------


def test_expert_splice_donor_is_a_similar_shot() -> None:
    seed = _pan_seed(20)
    base = syn._sample_scene(syn._rng(seed, "scene"), 160, 120, FPS)
    donor_fn = syn._make_donor(seed, RAW, [(0, base)])
    similar = syn._sample_scene(np.random.default_rng(3), 160, 120, FPS, like=base)
    assert similar.palette_seed == base.palette_seed and similar.style == base.style
    assert similar.camera_mode == base.camera_mode and similar.layout_seed != base.layout_seed
    rng = np.random.default_rng(0)
    near = donor_fn(rng, 3, 10, True)
    assert near.shape == (3, 120, 160, 3)
    source = _source_for(seed, RAW, 30)
    assert _mean_abs(near[0], source[9]) > 6.0  # yet clearly a different picture


def test_global_shift_estimation_tracks_scene_motion() -> None:
    base = _source_for(_pan_seed(30), RAW, 1)[0]
    shifted = [np.roll(base, (dy, dx), axis=(0, 1)) for dx, dy in [(0, 0), (4, 2), (8, 4), (12, 6)]]
    est = syn._global_shifts(np.stack(shifted))
    assert np.allclose(est[-1], [12, 6], atol=1.5)  # sign convention: content moved right / down
    still = syn._global_shifts(np.stack([base] * 4))
    assert np.abs(still).max() < 1e-3


def test_expert_inserted_object_follows_the_camera() -> None:
    config = SynthesisConfig(width=160, height=120, fps=FPS, min_duration=6.0, max_duration=6.0, degradation=False)
    seed = _pan_seed(40)
    follows = 0
    for offset in range(6):
        clip = generate_clip(seed + offset, config, issue_types=["inserted_object"], difficulty="expert", decoys=[])
        label = clip.issues[0]
        follows += label.params["drift"] == "follows_scene"
        x0, y0, x1, y1 = label.bbox
        assert (x1 - x0) < 0.5 and (y1 - y0) < 0.5
    assert follows >= 4


def test_issue_types_are_chosen_before_rendering_and_motion_types_get_moving_scenes() -> None:
    hints = syn._scene_hints(["frozen_frames", "zoom_jump"], "hard")
    assert hints.camera_modes == ("pan", "handheld") and hints.fast_objects
    plain = syn._scene_hints(["zoom_jump", "spliced_footage"], "hard")
    assert plain.camera_modes is None and not plain.fast_objects
    for seed in range(20):
        spec = syn._sample_scene(np.random.default_rng(seed), 160, 120, FPS, camera_modes=hints.camera_modes,
                                 fast_objects=True)
        assert spec.camera_mode in ("pan", "handheld")
        assert all(not obj.pauses and float(np.hypot(*obj.vel)) >= 30.0 for obj in spec.objects)


def test_issue_type_mix_is_balanced_across_types() -> None:
    config = SynthesisConfig(width=96, height=72, fps=FPS, min_duration=5.0, max_duration=7.0,
                             clean_fraction=0.0, decoy_rate=0.0)
    counts: Counter = Counter()
    for seed in range(120):
        counts.update(issue.type for issue in generate_clip(seed, config).issues)
    mean = sum(counts.values()) / len(ISSUE_TYPE_NAMES)
    assert set(counts) == set(ISSUE_TYPE_NAMES)
    for name in ISSUE_TYPE_NAMES:  # small sample, so a looser band than the 400-clip target
        assert 0.6 * mean <= counts[name] <= 1.4 * mean, (name, dict(counts))


def test_scene_look_does_not_depend_on_the_planned_issues() -> None:
    motion = syn._scene_hints(["frozen_frames"], "hard")
    for seed in range(30):
        free = syn._sample_scene(np.random.default_rng(seed), 160, 120, FPS)
        tied = syn._sample_scene(np.random.default_rng(seed), 160, 120, FPS, camera_modes=motion.camera_modes,
                                 fast_objects=motion.fast_objects)
        assert free.style == tied.style  # no "dim clip => no freeze" shortcut


def test_new_decoys_have_the_documented_effects() -> None:
    config = dataclasses.replace(DECOY_CONFIG, clean_fraction=1.0, degradation=False)
    for seed in range(4):
        means = lambda clip: np.array([float(f.mean()) for f in clip.frames])  # noqa: E731
        # auto exposure: a persistent, eased step of 5-15 % that stays after the transition.
        clip = generate_clip(seed, config, decoys=["auto_exposure_step"])
        d = clip.decoys[0]
        a, b = int(round(d.start_time * FPS)), int(round(d.end_time * FPS))
        m = means(clip)
        assert abs(np.mean(m[b + 1 : b + 4]) / np.mean(m[max(a - 3, 0) : a]) - 1.0) > 0.03 or b + 4 > len(m)
        # illumination flicker: brightness oscillates around a constant level without any jump.
        clip = generate_clip(seed, config, decoys=["illumination_flicker"])
        d = clip.decoys[0]
        a, b = int(round(d.start_time * FPS)), int(round(d.end_time * FPS))
        m = means(clip)
        assert (b - a) / FPS >= 2.0 - 1e-9 and max(abs(np.diff(m[a:b]))) < 0.06 * m[a]
        # white-balance step: channel means move in opposite directions and stay there.
        clip = generate_clip(seed, config, decoys=["auto_white_balance_step"])
        d = clip.decoys[0]
        a, b = int(round(d.start_time * FPS)), int(round(d.end_time * FPS))
        chan = np.array([f.reshape(-1, 3).mean(axis=0) for f in clip.frames])
        ratio_before = chan[max(a - 2, 0) : a + 1, 0].mean() / chan[max(a - 2, 0) : a + 1, 2].mean()
        ratio_after = chan[-3:, 0].mean() / chan[-3:, 2].mean()
        assert abs(ratio_after / ratio_before - 1.0) > 0.04
        # fast zoom does not come back: the picture is still zoomed in at the end of the clip.
        clip = generate_clip(seed, config, decoys=["fast_zoom"])
        raw = generate_clip(seed, config, decoys=[])
        assert _mean_abs(clip.frames[-1], raw.frames[-1]) > 4.0


def test_camera_direction_and_speed_events_shape_the_pan() -> None:
    spec = next(
        sp for sp in (syn._sample_scene(np.random.default_rng(i), 160, 120, FPS) for i in range(80))
        if sp.camera_mode == "pan"
    )
    n = 90
    long_x, _ly, _lz = syn._camera_arrays(spec, 600)  # the unedited path, long enough to look up
    base_x = long_x[:n]
    lookup = lambda index: np.interp(index, np.arange(600), long_x)  # noqa: E731

    turned_x, _y, _z = syn._camera_arrays(spec, n, [("camera_direction_change", 20, 29, 1.0)])
    assert np.array_equal(turned_x[:20], base_x[:20])
    frames = np.arange(36, 80)
    # After the reversal the camera retraces its path backwards: x(t) == base(c - t) for some c.
    errors = [float(np.abs(turned_x[frames] - lookup(c - frames)).max()) for c in np.arange(40, 140, 0.25)]
    assert min(errors) < 0.5
    assert np.abs(np.diff(turned_x)).max() < 5.0 and np.abs(np.diff(turned_x, 2)).max() < 2.0  # eased

    faster_x, _y2, _z2 = syn._camera_arrays(spec, n, [("camera_speed_change", 20, 26, 2.0)])
    assert np.array_equal(faster_x[:20], base_x[:20])
    errors = [float(np.abs(faster_x[frames] - lookup(c + 2.0 * frames)).max()) for c in np.arange(-100, 100, 0.25)]
    assert min(errors) < 0.5  # the same path, played twice as fast
    assert np.abs(np.diff(faster_x, 2)).max() < 3.0


def test_handheld_camera_has_a_small_scale_jitter() -> None:
    spec = next(
        sp for sp in (syn._sample_scene(np.random.default_rng(i), 160, 120, FPS) for i in range(80))
        if sp.camera_mode == "handheld"
    )
    _x, _y, zoom = syn._camera_arrays(spec, 120)
    calm = dataclasses.replace(spec, cam={**spec.cam, "jitter": 0.0})
    _cx, _cy, base_zoom = syn._camera_arrays(calm, 120)
    ratio = zoom / base_zoom
    assert ratio.max() <= 1.0101 and ratio.min() >= 0.9899 and ratio.std() > 0.001


# (decoy, edit it resembles, clearance in seconds): the edit window keeps its distance.
_PLACEMENT_RULES = [
    ("illumination_flicker", "exposure_flicker", 0.5),
    ("auto_exposure_step", "exposure_flicker", 0.5),
    ("auto_exposure_step", "color_grade_jump", 0.3),
    ("auto_white_balance_step", "color_grade_jump", 0.3),
    ("fast_zoom", "zoom_jump", 0.3),
    ("camera_direction_change", "reversed_segment", 0.5),
    ("camera_speed_change", "dropped_frames", 0.5),
]


@pytest.mark.parametrize("decoy, edit, seconds", _PLACEMENT_RULES)
def test_decoys_never_make_a_labelled_edit_ambiguous(decoy: str, edit: str, seconds: float) -> None:
    config = SynthesisConfig(width=160, height=120, fps=FPS, min_duration=8.0, max_duration=8.0, clean_fraction=0.0)
    clearance = math.ceil(seconds * FPS) - 1  # one frame of tolerance for label rounding
    for seed in range(6):
        clip = generate_clip(seed, config, issue_types=[edit], difficulty="hard", decoys=[decoy])
        d = next(x for x in clip.decoys if x.type == decoy)
        issue = clip.issues[0]
        d_start, d_end = int(round(d.start_time * FPS)), int(round(d.end_time * FPS))
        assert d_start - clearance >= issue.end_frame or issue.start_frame >= d_end + clearance, (seed, decoy, edit)
