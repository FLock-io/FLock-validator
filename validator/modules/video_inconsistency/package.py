"""Build and resolve validation packages for the video inconsistency task.

Zip layout::

    package.json    {"package_version", "suite_version", "manifest_path", "num_clips", "generator"}
    manifest.json   VideoManifest (clip.video_path like "videos/<clip_id>.mp4")
    videos/<clip_id>.mp4

Package or manifest problems are *operator-side* faults (a bad URL, a corrupt or
malicious archive, a stale package): they raise ``ValueError`` /
``FileNotFoundError`` and never ``VideoSubmissionError``, so a broken package
cannot zero a trainer's score.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

import requests
from loguru import logger

from validator.modules.video_inconsistency.issue_types import SUITE_VERSION
from validator.modules.video_inconsistency.manifest import ClipSpec, VideoManifest
from validator.modules.video_inconsistency.synthesis import SynthesisConfig, generate_clip
from validator.modules.video_inconsistency.video_io import encode_video, probe_video


PACKAGE_METADATA_FILENAME = "package.json"
MANIFEST_FILENAME = "manifest.json"
PACKAGE_VERSION = "video_inconsistency_package_v1"
GENERATOR_VERSION = "synthesis_v2"
DEFAULT_PACKAGE_CACHE_DIR = ".cache/video_inconsistency/package_cache"

# Limits protecting the validator host from hostile or broken archives.
MAX_DOWNLOAD_BYTES = 8 * 1024**3
MAX_UNCOMPRESSED_BYTES = 16 * 1024**3
MAX_ARCHIVE_MEMBERS = 20_000
_DOWNLOAD_TIMEOUT = (10.0, 60.0)  # (connect, read) seconds
_CLIP_ENCODE_ATTEMPTS = 5

# Fixed ZIP timestamp and permissions so the archive bytes depend only on content.
_ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)
_ZIP_FILE_MODE = (stat.S_IFREG | 0o644) << 16


@dataclass(frozen=True)
class ResolvedVideoPackage:
    manifest: VideoManifest
    root: Path  # extracted package root
    diagnostics: dict[str, str] = field(default_factory=dict)

    def clip_video_path(self, clip: ClipSpec) -> Path:
        """Absolute path of a clip's video, refusing anything outside the package root."""
        root = self.root.resolve()
        path = (root / clip.video_path).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"clip video path escapes the package root: {clip.video_path}") from exc
        return path


# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------


def _clip_id(seed: int, index: int) -> str:
    """12 hex chars from sha256(seed, index): carries no label information."""
    return hashlib.sha256(f"video_inconsistency_clip:{seed}:{index}".encode("utf-8")).hexdigest()[:12]


def _clip_seed(seed: int, index: int, attempt: int) -> int:
    digest = hashlib.sha256(f"video_inconsistency_seed:{seed}:{index}:{attempt}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _build_clip(
    seed: int, index: int, config: SynthesisConfig, videos_dir: Path
) -> ClipSpec:
    clip_id = _clip_id(seed, index)
    relative = f"videos/{clip_id}.mp4"
    for attempt in range(_CLIP_ENCODE_ATTEMPTS):
        clip = generate_clip(_clip_seed(seed, index, attempt), config)
        path = encode_video(clip.frames, videos_dir / f"{clip_id}.mp4", clip.fps, crf=clip.crf)
        # Verify the codec round trip kept every frame: label frame indices are
        # only meaningful if the decoded video has exactly the planned length.
        decoded = probe_video(path).num_frames
        if decoded == len(clip.frames):
            return ClipSpec(
                clip_id=clip_id,
                video_path=relative,
                fps=clip.fps,
                num_frames=len(clip.frames),
                width=config.width,
                height=config.height,
                difficulty=clip.difficulty,  # type: ignore[arg-type]
                source=clip.source,
                issues=clip.issues,
                decoys=clip.decoys,
                crf=clip.crf,
            )
        logger.warning(
            "clip {} attempt {}: decoded {} frames, expected {}; retrying with a new seed",
            index, attempt, decoded, len(clip.frames),
        )
    raise RuntimeError(f"could not produce a frame-exact encode for clip index {index}")


def _zip_bytes(archive: zipfile.ZipFile, name: str, data: bytes) -> None:
    info = zipfile.ZipInfo(name, date_time=_ZIP_EPOCH)
    info.compress_type = zipfile.ZIP_STORED  # mp4 is already compressed; json is tiny
    info.external_attr = _ZIP_FILE_MODE
    info.create_system = 3  # unix, independent of the build host
    archive.writestr(info, data)


def build_validation_package(
    output_zip: str | Path,
    *,
    num_clips: int,
    seed: int,
    config: SynthesisConfig = SynthesisConfig(),
) -> Path:
    """Generate ``num_clips`` labelled clips and write the package zip (byte-reproducible).

    Each clip is encoded at its own ``crf`` (part of the clip's degradation) and that crf is
    recorded in the manifest, as are the decoys (telemetry / hard-negative information).
    """
    if num_clips <= 0:
        raise ValueError("num_clips must be positive")
    output = Path(output_zip)
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="video_pkg_") as tmp:
        videos_dir = Path(tmp) / "videos"
        videos_dir.mkdir()
        specs = [_build_clip(seed, index, config, videos_dir) for index in range(num_clips)]
        manifest = VideoManifest(suite_version=SUITE_VERSION, clips=specs)
        metadata = {
            "package_version": PACKAGE_VERSION,
            "suite_version": SUITE_VERSION,
            "manifest_path": MANIFEST_FILENAME,
            "num_clips": num_clips,
            "generator": GENERATOR_VERSION,
        }
        entries: dict[str, bytes] = {
            MANIFEST_FILENAME: manifest.model_dump_json(indent=2).encode("utf-8"),
            PACKAGE_METADATA_FILENAME: json.dumps(metadata, indent=2, sort_keys=True).encode("utf-8"),
        }
        for spec in specs:
            entries[spec.video_path] = (Path(tmp) / spec.video_path).read_bytes()

        partial = output.with_name(output.name + ".partial")
        with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_STORED) as archive:
            for name in sorted(entries):
                _zip_bytes(archive, name, entries[name])
        os.replace(partial, output)
    logger.info("wrote video validation package {} ({} clips)", output, num_clips)
    return output


# ---------------------------------------------------------------------------
# Resolving
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, cache_dir: Path) -> Path:
    """Stream ``url`` into the cache with a timeout and a hard size cap."""
    key = hashlib.sha256(url.encode("utf-8")).hexdigest()
    target = cache_dir / f"download_{key}.zip"
    partial = target.with_name(target.name + ".part")
    written = 0
    with requests.get(url, stream=True, timeout=_DOWNLOAD_TIMEOUT) as response:
        response.raise_for_status()
        declared = response.headers.get("Content-Length")
        if declared and declared.isdigit() and int(declared) > MAX_DOWNLOAD_BYTES:
            raise ValueError(f"validation package is too large ({declared} bytes)")
        try:
            with partial.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    written += len(chunk)
                    if written > MAX_DOWNLOAD_BYTES:
                        raise ValueError("validation package exceeds the download size cap")
                    handle.write(chunk)
        except BaseException:
            partial.unlink(missing_ok=True)
            raise
    os.replace(partial, target)
    return target


def _materialize_zip(url_or_path: str, cache_dir: Path) -> Path:
    local = Path(url_or_path).expanduser()
    if local.exists():
        if not local.is_file():
            raise ValueError(f"validation package path is not a file: {url_or_path}")
        return local
    scheme = urlparse(url_or_path).scheme.lower()
    if scheme in ("http", "https"):
        return _download(url_or_path, cache_dir)
    if scheme:
        raise ValueError(f"unsupported validation package URL scheme: {scheme!r}")
    raise FileNotFoundError(f"validation package not found: {url_or_path}")


def _check_member_name(name: str) -> None:
    """Reject anything that could land outside the extraction root (zip-slip)."""
    if not name or "\\" in name or "\x00" in name:
        raise ValueError(f"unsafe zip entry name: {name!r}")
    if name.startswith("/") or re.match(r"^[A-Za-z]:", name):
        raise ValueError(f"absolute zip entry name: {name!r}")
    if ".." in PurePosixPath(name).parts:
        raise ValueError(f"zip entry escapes target directory: {name!r}")


def _safe_extract_zip(zip_path: Path, output_dir: Path) -> None:
    try:
        archive = zipfile.ZipFile(zip_path)
    except zipfile.BadZipFile as exc:
        raise ValueError(f"validation package is not a valid zip: {exc}") from exc
    with archive:
        members = archive.infolist()
        if len(members) > MAX_ARCHIVE_MEMBERS:
            raise ValueError(f"zip has too many entries ({len(members)})")
        if sum(member.file_size for member in members) > MAX_UNCOMPRESSED_BYTES:
            raise ValueError("zip uncompressed size exceeds the cap")
        seen: set[str] = set()
        root = output_dir.resolve()
        for member in members:
            _check_member_name(member.filename)
            if member.filename in seen:
                raise ValueError(f"duplicate zip entry: {member.filename!r}")
            seen.add(member.filename)
            mode = member.external_attr >> 16
            file_type = stat.S_IFMT(mode)
            # Symlinks (and devices, fifos...) could redirect later writes or reads
            # out of the cache directory; only regular files and directories are allowed.
            if file_type not in (0, stat.S_IFREG, stat.S_IFDIR):
                raise ValueError(f"zip entry is not a regular file: {member.filename!r}")
            target = (root / member.filename).resolve()
            try:
                target.relative_to(root)
            except ValueError as exc:
                raise ValueError(f"zip entry escapes target directory: {member.filename!r}") from exc
        written_total = 0
        for member in members:
            target = (root / member.filename).resolve()
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            # Count real bytes rather than trusting the header's declared size.
            with archive.open(member) as source, target.open("wb") as sink:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    written_total += len(chunk)
                    if written_total > MAX_UNCOMPRESSED_BYTES:
                        raise ValueError("zip uncompressed size exceeds the cap")
                    sink.write(chunk)


def _read_package_metadata(root: Path) -> dict:
    path = root / PACKAGE_METADATA_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"validation package is missing {PACKAGE_METADATA_FILENAME}")
    try:
        metadata = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{PACKAGE_METADATA_FILENAME} is not valid JSON: {exc}") from exc
    if not isinstance(metadata, dict):
        raise ValueError(f"{PACKAGE_METADATA_FILENAME} must contain a JSON object")
    if metadata.get("package_version") != PACKAGE_VERSION:
        raise ValueError(
            f"unsupported package_version {metadata.get('package_version')!r}; expected {PACKAGE_VERSION!r}"
        )
    return metadata


def _inside_root(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"packaged path escapes the package root: {relative}") from exc
    return path


def resolve_validation_package(url_or_path: str, cache_dir: str | Path) -> ResolvedVideoPackage:
    """Fetch (if remote), safely extract and validate a validation package."""
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    zip_path = _materialize_zip(str(url_or_path), cache)
    sha256 = _sha256_file(zip_path)

    extract_dir = cache / f"zip_{sha256[:16]}"
    staging = Path(tempfile.mkdtemp(prefix=f"zip_{sha256[:16]}.tmp-", dir=cache))
    try:
        _safe_extract_zip(zip_path, staging)
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        os.chmod(staging, 0o755)  # mkdtemp creates 0700
        os.replace(staging, extract_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    metadata = _read_package_metadata(extract_dir)
    manifest_path = _inside_root(extract_dir, str(metadata.get("manifest_path") or MANIFEST_FILENAME))
    if not manifest_path.is_file():
        raise FileNotFoundError(f"validation package manifest is missing: {manifest_path.name}")
    manifest = VideoManifest.model_validate_json(manifest_path.read_text())
    # suite_version is deliberately NOT checked here: the validation module compares
    # it against its configured suite and raises RecoverableException, so an
    # operator-side package/config mismatch re-queues the assignment instead of
    # being retried and marked failed.
    declared = metadata.get("num_clips")
    if declared is not None and declared != len(manifest.clips):
        raise ValueError(f"package.json declares {declared} clips but the manifest has {len(manifest.clips)}")

    package = ResolvedVideoPackage(manifest=manifest, root=extract_dir, diagnostics={})
    for clip in manifest.clips:
        video = package.clip_video_path(clip)
        if not video.is_file():
            raise FileNotFoundError(f"clip {clip.clip_id}: video file missing: {clip.video_path}")

    counts = Counter(issue.type for clip in manifest.clips for issue in clip.issues)
    decoy_counts = Counter(decoy.type for clip in manifest.clips for decoy in clip.decoys)
    difficulty_counts = Counter(clip.difficulty for clip in manifest.clips)
    diagnostics = {
        "video_package_source": str(url_or_path),
        "video_package_sha256": sha256,
        "num_clips": str(len(manifest.clips)),
        "suite_version": manifest.suite_version,
        "clean_clips": str(sum(1 for clip in manifest.clips if not clip.issues)),
        "issue_type_counts": json.dumps(dict(sorted(counts.items())), sort_keys=True),
        "decoy_type_counts": json.dumps(dict(sorted(decoy_counts.items())), sort_keys=True),
        "difficulty_counts": json.dumps(dict(sorted(difficulty_counts.items())), sort_keys=True),
    }
    return ResolvedVideoPackage(manifest=manifest, root=extract_dir, diagnostics=diagnostics)
