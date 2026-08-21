"""Download and validate the frozen RenAge inference bundle."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path


ASSET_VERSION = "1.0.0"
ARCHIVE_NAME = f"renage-ensemble-v{ASSET_VERSION}.tar.gz"
ASSET_URL = (
    f"https://github.com/GuoqinRen/RenAge/releases/download/v{ASSET_VERSION}/"
    f"{ARCHIVE_NAME}"
)
ARCHIVE_SHA256 = "a2fb0f104b454ba6f43ac3a31cac0254be9a64f490327c966df7ef2adfb4495c"
REQUIRED_FILES = ("renage_ensemble.pt", "feature_ids.txt", "reference_values.npy", "manifest.json")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def default_cache_root() -> Path:
    if platform.system() == "Darwin":
        return Path.home() / "Library" / "Caches" / "RenAge"
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "renage"


def validate_asset_dir(asset_dir: Path) -> dict[str, object]:
    asset_dir = asset_dir.expanduser().resolve()
    missing = [name for name in REQUIRED_FILES if not (asset_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete RenAge asset directory; missing: {', '.join(missing)}")

    manifest = json.loads((asset_dir / "manifest.json").read_text())
    if manifest.get("asset_version") != ASSET_VERSION:
        raise ValueError(
            f"Asset version {manifest.get('asset_version')!r} does not match software version {ASSET_VERSION!r}"
        )
    expected = manifest.get("files", {})
    for name in REQUIRED_FILES[:-1]:
        entry = expected.get(name)
        if not isinstance(entry, dict) or "sha256" not in entry:
            raise ValueError(f"Missing checksum entry for {name}")
        observed = sha256_file(asset_dir / name)
        if observed != entry["sha256"]:
            raise ValueError(f"Checksum mismatch for {name}")
    return manifest


def _safe_extract(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle.getmembers():
            if member.issym() or member.islnk():
                raise ValueError(f"Links are not allowed in the asset archive: {member.name}")
            member_path = (destination / member.name).resolve()
            try:
                member_path.relative_to(destination)
            except ValueError as exc:
                raise ValueError(f"Unsafe path in asset archive: {member.name}") from exc
        handle.extractall(destination)


def download_assets(destination: Path | None = None, force: bool = False) -> Path:
    destination = (destination or default_cache_root() / f"v{ASSET_VERSION}").expanduser().resolve()
    if destination in {Path(destination.anchor), Path.home().resolve()}:
        raise ValueError(f"Refusing to use an unsafe asset destination: {destination}")
    if destination.exists():
        if not force:
            validate_asset_dir(destination)
            return destination
        known_names = {path.name for path in destination.iterdir()}
        if not set(REQUIRED_FILES).issubset(known_names):
            raise ValueError("Refusing to replace a directory that is not a recognizable RenAge asset directory")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="renage-download-", dir=destination.parent) as temporary:
        temporary_path = Path(temporary)
        archive_path = temporary_path / ARCHIVE_NAME
        urllib.request.urlretrieve(ASSET_URL, archive_path)
        observed = sha256_file(archive_path)
        if observed != ARCHIVE_SHA256:
            raise ValueError(f"Downloaded archive checksum mismatch: expected {ARCHIVE_SHA256}, observed {observed}")
        extracted = temporary_path / "extracted"
        extracted.mkdir()
        _safe_extract(archive_path, extracted)
        candidates = [path for path in extracted.iterdir() if path.is_dir()]
        source = candidates[0] if len(candidates) == 1 else extracted
        validate_asset_dir(source)
        staged = temporary_path / "staged"
        shutil.copytree(source, staged)
        if destination.exists():
            shutil.rmtree(destination)
        staged.replace(destination)
    return destination


def resolve_assets(explicit: str | Path | None = None, allow_download: bool = True) -> Path:
    configured = explicit or os.environ.get("RENAGE_ASSET_DIR")
    destination = (
        Path(configured).expanduser().resolve()
        if configured
        else default_cache_root() / f"v{ASSET_VERSION}"
    )
    try:
        validate_asset_dir(destination)
        return destination
    except (FileNotFoundError, ValueError):
        if not allow_download:
            raise
    return download_assets(destination)
