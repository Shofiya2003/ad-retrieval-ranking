"""The retrieval service's bundle loader: the happy path and every refusal.

Mirrors the Go loader's corruption table, so neither side is looser than the
other about what counts as a servable bundle.
"""

import hashlib
import json

import numpy as np
import pytest

from retrieval import bundle as B


def bundle_dir(root):
    return root / (root / "LATEST").read_text().strip()


def edit_manifest(root, change):
    path = bundle_dir(root) / "manifest.json"
    manifest = json.loads(path.read_text())
    change(manifest)
    path.write_text(json.dumps(manifest))


def rewrite_vectors(root, vectors):
    """Replace the vectors AND update the checksum, so only the content check
    stands between the bad data and the service."""
    data = np.ascontiguousarray(vectors, dtype="<f4").tobytes()
    (bundle_dir(root) / "vectors.f32").write_bytes(data)
    edit_manifest(root, lambda m: m["vectors"].update(sha256=hashlib.sha256(data).hexdigest()))


def rewrite_ads(root, change_lines):
    path = bundle_dir(root) / "ads.jsonl"
    lines = change_lines(path.read_text().splitlines())
    data = ("\n".join(lines) + "\n").encode()
    path.write_bytes(data)
    edit_manifest(root, lambda m: m["ads"].update(sha256=hashlib.sha256(data).hexdigest()))


def original_vectors(root):
    raw = (bundle_dir(root) / "vectors.f32").read_bytes()
    return np.frombuffer(raw, dtype="<f4").reshape(12, 8).copy()


# --- the happy path -------------------------------------------------------


def test_load_follows_latest_and_returns_aligned_data(published_root, tmp_path):
    loaded = B.load(published_root)

    assert loaded.version == (published_root / "LATEST").read_text().strip()
    assert (loaded.count, loaded.dim) == (12, 8)
    assert loaded.ad_ids == tuple(f"ad_{i:05d}" for i in range(12))
    expected = np.load(tmp_path / "embedded" / "embeddings.npy")
    assert np.array_equal(loaded.vectors, expected)
    assert loaded.model["name"] == "fake/test-embedder"


def test_vectors_are_read_only(published_root):
    loaded = B.load(published_root)
    with pytest.raises(ValueError):
        loaded.vectors[0, 0] = 1.0


# --- refusals -------------------------------------------------------------


@pytest.mark.parametrize("latest", ["", "   \n", "../elsewhere", "a/b", "."])
def test_bad_latest_pointer(published_root, latest):
    (published_root / "LATEST").write_text(latest)
    with pytest.raises(B.ManifestError):
        B.load(published_root)


def test_missing_latest_pointer(published_root):
    (published_root / "LATEST").unlink()
    with pytest.raises(B.ManifestError):
        B.load(published_root)


@pytest.mark.parametrize(
    "change",
    [
        pytest.param(lambda m: m.update(format_version=2), id="unknown-format-version"),
        pytest.param(lambda m: m["vectors"].update(dtype="float64"), id="dtype"),
        pytest.param(lambda m: m["vectors"].update(byte_order="big"), id="byte-order"),
        pytest.param(lambda m: m["model"].update(normalized=False), id="not-normalized"),
        pytest.param(lambda m: m["model"].update(similarity="l2"), id="not-dot"),
        pytest.param(lambda m: m["model"].update(dim=16), id="dim-disagrees"),
        pytest.param(lambda m: m["ads"].update(count=11), id="counts-disagree"),
        pytest.param(lambda m: m["vectors"].update(bytes=4), id="bytes-arithmetic"),
        pytest.param(lambda m: m["vectors"].update(file="../vectors.f32"), id="path-escape"),
        pytest.param(lambda m: m["ads"].update(sha256=""), id="missing-checksum"),
        pytest.param(lambda m: m.pop("model"), id="missing-section"),
    ],
)
def test_manifest_contract(published_root, change):
    edit_manifest(published_root, change)
    with pytest.raises(B.ManifestError):
        B.load(published_root)


def test_unreadable_manifest(published_root):
    (bundle_dir(published_root) / "manifest.json").write_text("{not json")
    with pytest.raises(B.ManifestError):
        B.load(published_root)


def test_truncated_vectors(published_root):
    path = bundle_dir(published_root) / "vectors.f32"
    path.write_bytes(path.read_bytes()[:-4])
    with pytest.raises(B.VectorsError):
        B.load(published_root)


def test_flipped_byte_in_vectors(published_root):
    path = bundle_dir(published_root) / "vectors.f32"
    data = bytearray(path.read_bytes())
    data[17] ^= 0x01
    path.write_bytes(bytes(data))
    with pytest.raises(B.ChecksumError):
        B.load(published_root)


def test_nan_row_with_a_matching_checksum(published_root):
    vectors = original_vectors(published_root)
    vectors[3, 0] = np.nan
    rewrite_vectors(published_root, vectors)
    with pytest.raises(B.VectorsError):
        B.load(published_root)


def test_zero_row_with_a_matching_checksum(published_root):
    """An all-zero row scores 0 against every query: it can never be retrieved."""
    vectors = original_vectors(published_root)
    vectors[5] = 0.0
    rewrite_vectors(published_root, vectors)
    with pytest.raises(B.VectorsError):
        B.load(published_root)


def test_edited_ads_without_a_new_checksum(published_root):
    path = bundle_dir(published_root) / "ads.jsonl"
    path.write_text(path.read_text().replace("Premium", "Discount", 1))
    with pytest.raises(B.ChecksumError):
        B.load(published_root)


def test_ad_row_out_of_line(published_root):
    def swap_first_two(lines):
        return [lines[1], lines[0], *lines[2:]]

    rewrite_ads(published_root, swap_first_two)
    with pytest.raises(B.AdsError, match="misalign"):
        B.load(published_root)


def test_duplicate_ad_id(published_root):
    def duplicate(lines):
        second = json.loads(lines[1])
        second["ad_id"] = json.loads(lines[0])["ad_id"]
        return [lines[0], json.dumps(second), *lines[2:]]

    rewrite_ads(published_root, duplicate)
    with pytest.raises(B.AdsError, match="duplicate"):
        B.load(published_root)


def test_fewer_ads_than_the_manifest_claims(published_root):
    rewrite_ads(published_root, lambda lines: lines[:-1])
    with pytest.raises(B.AdsError, match="11 ads"):
        B.load(published_root)


def test_unparseable_ad_line(published_root):
    rewrite_ads(published_root, lambda lines: [lines[0], "{broken", *lines[2:]])
    with pytest.raises(B.AdsError):
        B.load(published_root)
