"""Load the published bundle for serving, trusting nothing about it.

This is the same contract the Go loader enforces (serving/internal/bundle), now
on the side that actually searches the vectors. The publisher ran its own
checks, but a bundle can be truncated by a bad copy or edited by hand after it
was written, and a retrieval service that serves vectors paired with the wrong
ads gives confident nonsense rather than an error. So every claim the manifest
makes is checked against the bytes, cheap checks first, and any mismatch stops
the service from starting.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ingestion.publish import (
    FORMAT_VERSION,
    LATEST_POINTER,
    PublishError,
    verify_vectors,
)

MANIFEST_FILE = "manifest.json"


class BundleError(RuntimeError):
    """The bundle cannot be served."""


class ManifestError(BundleError):
    pass


class ChecksumError(BundleError):
    pass


class VectorsError(BundleError):
    pass


class AdsError(BundleError):
    pass


@dataclass(frozen=True)
class Bundle:
    version: str
    path: Path
    manifest: dict
    vectors: np.ndarray  # (count, dim) float32, read-only
    ad_ids: tuple[str, ...]  # ad_ids[row]; the rest of each ad belongs to Go

    @property
    def dim(self) -> int:
        return self.vectors.shape[1]

    @property
    def count(self) -> int:
        return self.vectors.shape[0]

    @property
    def model(self) -> dict:
        return self.manifest["model"]


def is_plain_name(name: str) -> bool:
    """A name that, joined onto a directory, stays inside it."""
    return name not in ("", ".", "..") and "/" not in name and "\\" not in name


def resolve_latest(root: Path) -> Path:
    try:
        version = (root / LATEST_POINTER).read_text(encoding="utf-8").strip()
    except OSError as err:
        raise ManifestError(f"cannot read {root / LATEST_POINTER}: {err}") from err
    if not is_plain_name(version):
        raise ManifestError(f"LATEST names {version!r}, want a plain version directory")
    return root / version


def load(root: Path | str) -> Bundle:
    """Follow root/LATEST to the current bundle and load it."""
    return load_dir(resolve_latest(Path(root)))


def load_dir(path: Path | str) -> Bundle:
    path = Path(path)
    manifest = read_manifest(path / MANIFEST_FILE)
    vectors = read_vectors(path / manifest["vectors"]["file"], manifest["vectors"])
    ad_ids = read_ad_ids(path / manifest["ads"]["file"], manifest["ads"])
    return Bundle(version=path.name, path=path, manifest=manifest, vectors=vectors, ad_ids=ad_ids)


def read_manifest(path: Path) -> dict:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as err:
        raise ManifestError(f"cannot read {path}: {err}") from err
    try:
        validate_manifest(manifest)
    except (KeyError, TypeError) as err:
        raise ManifestError(f"{path} is missing or mistypes a field: {err!r}") from err
    return manifest


def validate_manifest(m: dict) -> None:
    """Everything the manifest can be checked for on its own, before any large
    file is read: there's no point hashing 68MB described by a manifest that
    contradicts itself."""
    if m["format_version"] != FORMAT_VERSION:
        raise ManifestError(
            f"format_version {m['format_version']}, this loader only reads {FORMAT_VERSION}"
        )

    v, model, ads = m["vectors"], m["model"], m["ads"]
    if (v["dtype"], v["byte_order"], v["layout"]) != ("float32", "little", "row_major"):
        raise ManifestError(
            f"vectors are {v['dtype']}/{v['byte_order']}/{v['layout']}, "
            "want float32/little/row_major"
        )

    # Search scores with an inner product. That is only cosine similarity if
    # every vector is unit length, so the manifest must promise both.
    if model["normalized"] is not True or model["similarity"] != "dot":
        raise ManifestError(
            f"model similarity {model['similarity']!r} normalized={model['normalized']}, "
            "want dot over unit vectors"
        )

    if v["count"] <= 0 or v["dim"] <= 0:
        raise ManifestError(f"vectors are {v['count']} x {v['dim']}")
    if v["dim"] != model["dim"]:
        raise ManifestError(f"vectors.dim {v['dim']} but model.dim {model['dim']}")
    if v["count"] != ads["count"]:
        raise ManifestError(
            f"{v['count']} vectors but {ads['count']} ads -- rows and ads must correspond"
        )
    if v["bytes"] != v["count"] * v["dim"] * 4:
        raise ManifestError(
            f"vectors.bytes {v['bytes']}, but {v['count']} x {v['dim']} float32 is "
            f"{v['count'] * v['dim'] * 4}"
        )

    for name in (v["file"], ads["file"]):
        if not is_plain_name(name):
            raise ManifestError(f"file name {name!r} is not a plain file name")
    if not v["sha256"] or not ads["sha256"]:
        raise ManifestError("missing checksum")


def read_vectors(path: Path, spec: dict) -> np.ndarray:
    try:
        size = path.stat().st_size
    except OSError as err:
        raise VectorsError(f"cannot stat {path}: {err}") from err
    # Catches truncation without reading a byte.
    if size != spec["bytes"]:
        raise VectorsError(f"{path} is {size} bytes, manifest says {spec['bytes']}")

    data = path.read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if actual != spec["sha256"]:
        raise ChecksumError(
            f"{path} sha256 is {actual[:12]}, manifest says {spec['sha256'][:12]}"
        )

    # "<f4" is little-endian float32 whatever this machine's byte order is.
    # frombuffer over immutable bytes gives a read-only array: nothing
    # downstream can modify the vectors by accident.
    vectors = np.frombuffer(data, dtype="<f4").reshape(spec["count"], spec["dim"])
    try:
        verify_vectors(vectors)
    except PublishError as err:
        raise VectorsError(str(err)) from err
    return vectors


def read_ad_ids(path: Path, spec: dict) -> tuple[str, ...]:
    digest = hashlib.sha256()
    ad_ids: list[str] = []
    seen: set[str] = set()
    try:
        handle = path.open("rb")
    except OSError as err:
        raise AdsError(f"cannot open {path}: {err}") from err
    with handle:
        for line in handle:
            digest.update(line)
            index = len(ad_ids)
            try:
                record = json.loads(line)
                row, ad_id = record["row"], record["ad_id"]
            except (ValueError, KeyError, TypeError) as err:
                raise AdsError(f"{path} line {index}: unreadable ad ({err!r})") from err
            if row != index:
                raise AdsError(f"{path} line {index} has row {row} -- vector rows would misalign")
            if ad_id in seen:
                raise AdsError(f"{path} line {index}: duplicate ad_id {ad_id!r}")
            seen.add(ad_id)
            ad_ids.append(ad_id)

    actual = digest.hexdigest()
    if actual != spec["sha256"]:
        raise ChecksumError(f"{path} sha256 is {actual[:12]}, manifest says {spec['sha256'][:12]}")
    if len(ad_ids) != spec["count"]:
        raise AdsError(f"{path} has {len(ad_ids)} ads, manifest says {spec['count']}")
    return tuple(ad_ids)
