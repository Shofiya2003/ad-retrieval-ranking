"""Tests for the publish stage: the bundle layout and every refusal."""

import json
import struct

import numpy as np
import pytest

from ingestion import embed, publish
from tests.conftest import make_ad


@pytest.fixture
def embedded(clean_file, tmp_path, fake_embedder):
    """A clean file that has been embedded, ready to publish."""
    out = tmp_path / "embedded"
    embed.run(clean_file, out, fake_embedder)
    return out


# --- the happy path -------------------------------------------------------


def test_bundle_contains_the_three_files(clean_file, embedded, tmp_path):
    manifest = publish.run(clean_file, embedded, tmp_path / "published")
    bundle = tmp_path / "published" / manifest["version"]

    assert (bundle / "vectors.f32").is_file()
    assert (bundle / "ads.jsonl").is_file()
    assert (bundle / "manifest.json").is_file()


def test_vectors_file_is_exactly_count_x_dim_x_four_bytes(clean_file, embedded, tmp_path):
    """Go will size its read off this arithmetic, so it has to hold exactly."""
    manifest = publish.run(clean_file, embedded, tmp_path / "published")
    size = (tmp_path / "published" / manifest["version"] / "vectors.f32").stat().st_size
    assert size == 12 * 8 * 4
    assert size == manifest["vectors"]["bytes"]


def test_raw_bytes_decode_the_way_go_will_read_them(clean_file, embedded, tmp_path):
    """Decode row 0 with plain struct -- no numpy -- exactly as Go's
    encoding/binary will, and check it matches what the embedder produced."""
    manifest = publish.run(clean_file, embedded, tmp_path / "published")
    bundle = tmp_path / "published" / manifest["version"]

    dim = manifest["vectors"]["dim"]
    raw = (bundle / "vectors.f32").read_bytes()[: dim * 4]
    decoded = struct.unpack(f"<{dim}f", raw)  # "<" = little-endian, as the manifest promises

    expected = np.load(embedded / "embeddings.npy")[0]
    assert np.allclose(decoded, expected, atol=1e-6)
    assert abs(np.linalg.norm(decoded) - 1.0) < 1e-5


def test_ads_jsonl_line_i_matches_vector_row_i(clean_file, embedded, tmp_path):
    manifest = publish.run(clean_file, embedded, tmp_path / "published")
    bundle = tmp_path / "published" / manifest["version"]

    lines = (bundle / "ads.jsonl").read_text().splitlines()
    assert len(lines) == 12
    for i, line in enumerate(lines):
        record = json.loads(line)
        assert record["row"] == i          # explicit, so alignment is checkable
        assert record["ad_id"] == f"ad_{i:05d}"


def test_manifest_checksums_match_the_files_on_disk(clean_file, embedded, tmp_path):
    from ingestion.fileutil import sha256_of_file

    manifest = publish.run(clean_file, embedded, tmp_path / "published")
    bundle = tmp_path / "published" / manifest["version"]

    assert manifest["vectors"]["sha256"] == sha256_of_file(bundle / "vectors.f32")
    assert manifest["ads"]["sha256"] == sha256_of_file(bundle / "ads.jsonl")


def test_manifest_carries_model_identity_for_the_phase_4_sidecar(clean_file, embedded, tmp_path):
    """The sidecar will read the model from here instead of hard-coding it, so
    query vectors and ad vectors cannot come from different models."""
    manifest = publish.run(clean_file, embedded, tmp_path / "published")
    assert manifest["model"]["name"] == "fake/test-embedder"
    assert manifest["model"]["revision"] == "0" * 40
    assert manifest["model"]["normalized"] is True
    assert manifest["model"]["similarity"] == "dot"


def test_latest_pointer_resolves_to_the_new_bundle(clean_file, embedded, tmp_path):
    root = tmp_path / "published"
    manifest = publish.run(clean_file, embedded, root)

    assert (root / "LATEST").read_text().strip() == manifest["version"]
    assert publish.read_latest(root) == root / manifest["version"]


def test_no_staging_directory_is_left_behind(clean_file, embedded, tmp_path):
    root = tmp_path / "published"
    publish.run(clean_file, embedded, root)
    assert not list(root.glob(".tmp-*"))


# --- refusals -------------------------------------------------------------


def test_publish_refuses_when_the_clean_file_changed_after_embedding(clean_file, embedded, tmp_path):
    """The quiet disaster this prevents: re-run validate.py with a different
    threshold, skip re-embedding, and every vector now belongs to a different
    ad. Nothing crashes; search just returns confident nonsense."""
    clean_file.write_text(
        "\n".join(json.dumps(make_ad(i)) for i in range(11)) + "\n", encoding="utf-8"
    )

    with pytest.raises(publish.PublishError, match="has changed since it was embedded"):
        publish.run(clean_file, embedded, tmp_path / "published")


def test_publish_refuses_on_a_row_count_mismatch(clean_file, embedded, tmp_path):
    vectors = np.load(embedded / "embeddings.npy")
    np.save(embedded / "embeddings.npy", vectors[:5])

    with pytest.raises(publish.PublishError, match="rows and ads must correspond"):
        publish.run(clean_file, embedded, tmp_path / "published")


def test_publish_refuses_on_nan(clean_file, embedded, tmp_path):
    """A NaN compares false against everything, so the row silently becomes
    invisible to search."""
    vectors = np.load(embedded / "embeddings.npy")
    vectors[3][0] = np.nan
    np.save(embedded / "embeddings.npy", vectors)

    with pytest.raises(publish.PublishError, match="NaN or infinity"):
        publish.run(clean_file, embedded, tmp_path / "published")


def test_publish_refuses_an_all_zero_vector(clean_file, embedded, tmp_path):
    """Dot product zero against every query: it can never be retrieved, and
    nothing would ever tell you."""
    vectors = np.load(embedded / "embeddings.npy")
    vectors[2] = 0.0
    np.save(embedded / "embeddings.npy", vectors)

    with pytest.raises(publish.PublishError, match="not unit length"):
        publish.run(clean_file, embedded, tmp_path / "published")


def test_a_refused_publish_leaves_the_previous_bundle_serving(clean_file, embedded, tmp_path):
    """The point of writing LATEST last: a failure part-way through must not
    disturb whatever the Go service is currently reading."""
    root = tmp_path / "published"
    good = publish.run(clean_file, embedded, root)

    vectors = np.load(embedded / "embeddings.npy")
    vectors[0][0] = np.nan
    np.save(embedded / "embeddings.npy", vectors)

    with pytest.raises(publish.PublishError):
        publish.run(clean_file, embedded, root)

    assert (root / "LATEST").read_text().strip() == good["version"]
    assert not list(root.glob(".tmp-*"))


def test_missing_inputs_are_reported_distinctly(tmp_path):
    code = publish.main(
        ["--clean", str(tmp_path / "nope.jsonl"), "--embedded", str(tmp_path), "--quiet"]
    )
    assert code == publish.EXIT_INPUT_ERROR


def test_cli_returns_the_gate_exit_code_on_refusal(clean_file, embedded, tmp_path):
    clean_file.write_text(json.dumps(make_ad(0)) + "\n", encoding="utf-8")
    code = publish.main(
        ["--clean", str(clean_file), "--embedded", str(embedded),
         "--out-root", str(tmp_path / "published"), "--quiet"]
    )
    assert code == publish.EXIT_GATE_FAILED
