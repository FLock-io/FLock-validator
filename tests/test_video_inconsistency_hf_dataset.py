"""Tests for the Hugging Face dataset builder (tiny builds; the network is never touched)."""

from __future__ import annotations

import hashlib
import json
import os
import zipfile
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("imageio_ffmpeg")

from validator.modules.video_inconsistency import build_hf_dataset as hf  # noqa: E402
from validator.modules.video_inconsistency.issue_types import ISSUE_TYPE_NAMES  # noqa: E402
from validator.modules.video_inconsistency.manifest import DIFFICULTIES, DecoyLabel, IssueLabel  # noqa: E402
from validator.modules.video_inconsistency.package import resolve_validation_package  # noqa: E402
from validator.modules.video_inconsistency.video_io import probe_video  # noqa: E402

SMALL = ["--width", "64", "--height", "48", "--fps", "10", "--min-duration", "2", "--max-duration", "3"]


def _build(out_dir: Path, train: int, validation: int, dev: int, workers: int = 1, seed: int = 11) -> None:
    argv = [
        "--out-dir", str(out_dir), "--train-clips", str(train), "--validation-clips", str(validation),
        "--dev-package-clips", str(dev), "--workers", str(workers), "--seed", str(seed), *SMALL,
    ]
    assert hf.main(argv) == 0


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out_dir = tmp_path_factory.mktemp("hf") / "dataset"
    _build(out_dir, train=6, validation=3, dev=3)
    return out_dir


# ---------------------------------------------------------------------------------------
# layout and schema
# ---------------------------------------------------------------------------------------
def test_layout(dataset: Path) -> None:
    for name in ("README.md", "issue_types.json", "stats.json", "train/metadata.jsonl",
                 "validation/metadata.jsonl", hf.DEV_PACKAGE_RELPATH):
        assert (dataset / name).is_file(), name
    assert len(list((dataset / "train").glob("*.mp4"))) == 6
    assert len(list((dataset / "validation").glob("*.mp4"))) == 3
    assert not (dataset / ".build").exists(), "resume sidecars must not survive a finished build"
    catalogue = json.loads((dataset / "issue_types.json").read_text())
    assert [i["name"] for i in catalogue["issue_types"]] == list(ISSUE_TYPE_NAMES)
    assert {"spatial", "point_event", "description", "category"} <= set(catalogue["issue_types"][0])
    assert [d["name"] for d in catalogue["decoy_types"]] == list(hf.DECOY_TYPES)


def test_metadata_schema_roundtrips(dataset: Path) -> None:
    keys = {"file_name", "clip_id", "fps", "num_frames", "width", "height", "duration", "difficulty",
            "source", "crf", "issues", "decoys"}
    for split, count in (("train", 6), ("validation", 3)):
        rows = _rows(dataset / split / "metadata.jsonl")
        assert len(rows) == count
        for row in rows:
            assert set(row) == keys
            assert row["file_name"] == f"{row['clip_id']}.mp4" and len(row["clip_id"]) == 12
            assert row["difficulty"] in DIFFICULTIES
            assert row["duration"] == pytest.approx(row["num_frames"] / row["fps"])
            assert isinstance(row["crf"], int)
            for issue in row["issues"]:
                assert IssueLabel.model_validate(issue).model_dump(mode="json") == issue
            for decoy in row["decoys"]:
                assert DecoyLabel.model_validate(decoy).model_dump(mode="json") == decoy
            hf.validate_row(row)
            info = probe_video(dataset / split / row["file_name"])
            assert info.num_frames == row["num_frames"]
            assert (info.width, info.height) == (row["width"], row["height"]) == (64, 48)


def test_splits_are_disjoint_and_ids_carry_no_labels(dataset: Path) -> None:
    train = {r["clip_id"] for r in _rows(dataset / "train" / "metadata.jsonl")}
    validation = {r["clip_id"] for r in _rows(dataset / "validation" / "metadata.jsonl")}
    assert train.isdisjoint(validation)
    dev = resolve_validation_package(str(dataset / hf.DEV_PACKAGE_RELPATH), str(dataset.parent / "cache"))
    dev_ids = {c.clip_id for c in dev.manifest.clips}
    assert len(dev_ids) == 3 and dev_ids.isdisjoint(train | validation)
    # seeds: the same index in train and validation, and a different --seed, all give different seeds
    seeds = {hf.clip_seed_for(s, 11, i, 0) for s in hf.SPLITS for i in range(50)}
    assert len(seeds) == 100
    assert hf.clip_seed_for("train", 12, 0, 0) != hf.clip_seed_for("train", 11, 0, 0)
    # the dev package's own seed namespace (package._clip_seed) cannot collide with ours
    package = pytest.importorskip("validator.modules.video_inconsistency.package")
    if hasattr(package, "_clip_seed"):
        dev_seeds = {package._clip_seed(7, i, 0) for i in range(50)}
        assert dev_seeds.isdisjoint(seeds)
    # ids are hashes, not something like train_0001 or an issue-type name
    assert all(all(ch in "0123456789abcdef" for ch in clip_id) for clip_id in train | validation)


def test_stats_are_consistent_with_metadata(dataset: Path) -> None:
    stats = json.loads((dataset / "stats.json").read_text())
    assert set(stats["splits"]) == {"train", "validation", "dev_package"}
    for split in ("train", "validation"):
        rows = _rows(dataset / split / "metadata.jsonl")
        info = stats["splits"][split]
        assert info["clips"] == len(rows)
        assert info["clean_clips"] == sum(1 for r in rows if not r["issues"])
        assert info["issues_total"] == sum(len(r["issues"]) for r in rows) == sum(info["issues_per_type"].values())
        assert set(info["issues_per_type"]) == set(ISSUE_TYPE_NAMES)
        assert info["decoys_total"] == sum(len(r["decoys"]) for r in rows) == sum(info["decoys_per_type"].values())
        assert set(info["decoys_per_type"]) == set(hf.DECOY_TYPES)
        assert sum(info["difficulty_histogram"].values()) == len(rows)
        assert set(info["difficulty_histogram"]) == set(DIFFICULTIES)
        assert info["total_duration_seconds"] == pytest.approx(sum(r["duration"] for r in rows), abs=1e-2)
        assert info["bytes"] == sum((dataset / split / r["file_name"]).stat().st_size for r in rows)
    assert stats["splits"]["dev_package"]["clips"] == 3
    assert stats["generation"]["dev_package_seed"] == 7


def test_dataset_card(dataset: Path) -> None:
    card = (dataset / "README.md").read_text()
    assert card.startswith("---\nlicense: cc-by-4.0\n")
    front = card.split("---\n")[1]
    assert "video-classification" in front and "split: train" in front and "split: validation" in front
    assert "train/**" in front
    for name in ISSUE_TYPE_NAMES:
        assert f"`{name}`" in card
    for name in hf.DECOY_TYPES:
        assert f"`{name}`" in card
    for name in DIFFICULTIES:
        assert f"`{name}`" in card
    assert "different, secret seed" in card
    assert "snapshot_download" in card and "train" in card and "metadata.jsonl" in card
    assert "end_time = (e + 1) / fps" in card


def test_dev_package_is_a_valid_package(dataset: Path) -> None:
    with zipfile.ZipFile(dataset / hf.DEV_PACKAGE_RELPATH) as archive:
        meta = json.loads(archive.read("package.json"))
    assert meta["num_clips"] == 3


# ---------------------------------------------------------------------------------------
# determinism, resuming
# ---------------------------------------------------------------------------------------
def test_deterministic_across_worker_counts(tmp_path: Path) -> None:
    one, two = tmp_path / "one", tmp_path / "two"
    _build(one, train=4, validation=2, dev=0, workers=1)
    _build(two, train=4, validation=2, dev=0, workers=2)
    for split in ("train", "validation"):
        assert (one / split / "metadata.jsonl").read_bytes() == (two / split / "metadata.jsonl").read_bytes()
        names = sorted(p.name for p in (one / split).glob("*.mp4"))
        assert names == sorted(p.name for p in (two / split).glob("*.mp4")) and names
        for name in names:
            assert _digest(one / split / name) == _digest(two / split / name)
    assert not (one / "dev_package").exists()
    assert "dev_package" not in json.loads((one / "stats.json").read_text())["splits"]


def test_resume_only_rebuilds_missing_clips(tmp_path: Path) -> None:
    out = tmp_path / "ds"
    _build(out, train=4, validation=0, dev=0)
    assert not (out / "validation").exists()
    rows = _rows(out / "train" / "metadata.jsonl")
    before = {r["clip_id"]: (out / "train" / r["file_name"]).stat().st_mtime_ns for r in rows}
    original_bytes = (out / "train" / "metadata.jsonl").read_bytes()

    victim = rows[1]["clip_id"]
    (out / "train" / f"{victim}.mp4").unlink()
    (out / "train" / "metadata.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows if r["clip_id"] != victim))
    stale = out / "train" / "stale_clip.mp4"
    stale.write_bytes(b"not a real clip")

    _build(out, train=4, validation=0, dev=0)
    after = {r["clip_id"]: (out / "train" / r["file_name"]).stat().st_mtime_ns for r in _rows(out / "train" / "metadata.jsonl")}
    assert (out / "train" / "metadata.jsonl").read_bytes() == original_bytes  # identical after the rebuild
    assert {k for k in after if after[k] != before[k]} == {victim}  # only the missing clip was rebuilt
    assert not stale.exists(), "mp4 files that are not in the metadata must be removed"

    # extending the split keeps the existing clips (same seed, more indices)
    _build(out, train=6, validation=0, dev=0)
    assert len(_rows(out / "train" / "metadata.jsonl")) == 6
    assert (out / "train" / f"{victim}.mp4").stat().st_mtime_ns == after[victim]


def test_resume_refuses_different_parameters(tmp_path: Path) -> None:
    out = tmp_path / "ds"
    _build(out, train=2, validation=0, dev=0, seed=1)
    with pytest.raises(ValueError, match="different"):
        _build(out, train=2, validation=0, dev=0, seed=2)


# ---------------------------------------------------------------------------------------
# upload (fake HfApi; never the network)
# ---------------------------------------------------------------------------------------
class FakeHfApi:
    instances: list["FakeHfApi"] = []
    existing: dict[str, bool] = {}  # repo_id -> private?

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        assert not args
        self.token = kwargs.get("token")
        self.calls: list[tuple[str, dict[str, Any]]] = []
        FakeHfApi.instances.append(self)

    def repo_info(self, repo_id: str, repo_type: str = "model", **_: Any) -> Any:
        from huggingface_hub.utils import RepositoryNotFoundError

        self.calls.append(("repo_info", {"repo_id": repo_id, "repo_type": repo_type}))
        if repo_id not in FakeHfApi.existing:
            from unittest import mock

            try:
                raise RepositoryNotFoundError("not found", response=mock.MagicMock())
            except TypeError:  # older huggingface_hub signature
                raise RepositoryNotFoundError("not found")
        return type("Info", (), {"private": FakeHfApi.existing[repo_id]})()

    def create_repo(self, repo_id: str, **kwargs: Any) -> None:
        self.calls.append(("create_repo", {"repo_id": repo_id, **kwargs}))
        FakeHfApi.existing.setdefault(repo_id, bool(kwargs.get("private")))

    def upload_large_folder(self, **kwargs: Any) -> None:
        self.calls.append(("upload_large_folder", kwargs))


@pytest.fixture
def fake_hub(monkeypatch: pytest.MonkeyPatch) -> type[FakeHfApi]:
    import huggingface_hub

    FakeHfApi.instances = []
    FakeHfApi.existing = {}
    monkeypatch.setattr(huggingface_hub, "HfApi", FakeHfApi)
    monkeypatch.setenv("HF_TOKEN", "hf_test_token_value")
    return FakeHfApi


def _call(api: FakeHfApi, name: str) -> dict[str, Any]:
    return next(kwargs for call, kwargs in api.calls if call == name)


def test_upload_is_private_by_default_and_uses_env_token(
    dataset: Path, fake_hub: type[FakeHfApi], capsys: pytest.CaptureFixture[str]
) -> None:
    assert hf.main(["--out-dir", str(dataset), "--skip-build", "--push-to-hub", "org/ds"]) == 0
    (api,) = fake_hub.instances
    assert api.token == "hf_test_token_value"
    create = _call(api, "create_repo")
    assert create["private"] is True and create["repo_type"] == "dataset" and create["exist_ok"] is True
    upload = _call(api, "upload_large_folder")
    assert upload["repo_type"] == "dataset" and upload["repo_id"] == "org/ds"
    assert upload["folder_path"] == str(dataset)
    assert any(".build" in pattern for pattern in upload["ignore_patterns"])
    out = capsys.readouterr().out
    assert "https://huggingface.co/datasets/org/ds" in out and "visibility: private" in out
    assert "hf_test_token_value" not in out


def test_public_flag_creates_public_repo(dataset: Path, fake_hub: type[FakeHfApi], capsys: pytest.CaptureFixture[str]) -> None:
    assert hf.main(["--out-dir", str(dataset), "--skip-build", "--push-to-hub", "org/pub", "--public"]) == 0
    assert _call(fake_hub.instances[0], "create_repo")["private"] is False
    assert "visibility: PUBLIC" in capsys.readouterr().out


def test_refuses_public_repo_when_private_requested(
    dataset: Path, fake_hub: type[FakeHfApi], capsys: pytest.CaptureFixture[str]
) -> None:
    fake_hub.existing["org/open"] = False
    assert hf.main(["--out-dir", str(dataset), "--skip-build", "--push-to-hub", "org/open"]) == 2
    (api,) = fake_hub.instances
    assert not [c for c, _ in api.calls if c in ("create_repo", "upload_large_folder")]
    assert "PUBLIC" in capsys.readouterr().err
    # an existing private repo is fine; --private is explicit but equivalent to the default
    fake_hub.existing["org/closed"] = True
    assert hf.main(["--out-dir", str(dataset), "--skip-build", "--push-to-hub", "org/closed", "--private"]) == 0


def test_token_only_from_environment(
    dataset: Path, fake_hub: type[FakeHfApi], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("HF_TOKEN")
    assert hf.main(["--out-dir", str(dataset), "--skip-build", "--push-to-hub", "org/ds"]) == 2
    assert not fake_hub.instances, "no API client may be created without HF_TOKEN"
    assert "HF_TOKEN" in capsys.readouterr().err
    with pytest.raises(SystemExit):  # there is no token argument
        hf.main(["--out-dir", str(dataset), "--skip-build", "--push-to-hub", "org/ds", "--token", "x"])
    with pytest.raises(SystemExit):  # private and public are mutually exclusive
        hf.main(["--out-dir", str(dataset), "--skip-build", "--private", "--public"])


def test_upload_refuses_a_folder_that_is_not_a_dataset(tmp_path: Path, fake_hub: type[FakeHfApi]) -> None:
    (tmp_path / "junk.txt").write_text("x")
    assert hf.main(["--out-dir", str(tmp_path), "--skip-build", "--push-to-hub", "org/ds"]) == 2
    assert not fake_hub.instances


def test_build_without_push_never_touches_hub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import huggingface_hub

    def boom(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("the Hub must not be contacted without --push-to-hub")

    monkeypatch.setattr(huggingface_hub, "HfApi", boom)
    monkeypatch.setenv("HF_TOKEN", "hf_should_not_be_used")
    _build(tmp_path / "ds", train=2, validation=0, dev=0)
    assert os.path.isfile(tmp_path / "ds" / "train" / "metadata.jsonl")


def test_trainer_sample_reads_the_built_dataset(dataset: Path) -> None:
    pytest.importorskip("torch")
    import sys

    sample = Path(hf.__file__).resolve().parent / "trainer_sample"
    if str(sample) not in sys.path:
        sys.path.insert(0, str(sample))
    import train

    assert train.resolve_splits(dataset, None) == (dataset / "train", dataset / "validation")
    records = train.read_labels(dataset / "train")
    assert len(records) == 6
    for record in records:
        assert (Path(record["_root"]) / record["clip_file"]).is_file()
        assert "decoys" in record
        train.to_clip_spec(0, record)  # rows convert to the validator's ClipSpec (issues + decoys)
