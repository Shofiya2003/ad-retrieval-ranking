"""Tests for the embedding stage, using FakeEmbedder (no model download)."""

import json

import numpy as np
import pytest

from ingestion import embed
from tests.conftest import make_ad


# --- what text actually gets embedded ------------------------------------


def test_headline_and_description_are_joined():
    assert embed.compose_text(make_ad(1)) == (
        "Premium running shoes model 1. Tested over 500 miles."
    )


def test_missing_description_leaves_just_the_headline():
    assert embed.compose_text(make_ad(1, description="")) == "Premium running shoes model 1"


def test_no_double_punctuation_when_the_headline_already_ends_in_a_stop():
    text = embed.compose_text(make_ad(1, headline="Save big today!"))
    assert text == "Save big today! Tested over 500 miles."


def test_category_and_bid_are_not_embedded():
    """They are separate ranking signals in Phase 6. Folding category into the
    vector would count it twice and blur retrieval against ranking."""
    text = embed.compose_text(make_ad(1, category="footwear", bid_cpm_usd=9.99))
    assert "footwear" not in text
    assert "9.99" not in text


# --- the token gate -------------------------------------------------------


def test_over_limit_ad_fails_the_run(tmp_path, fake_embedder):
    """Phase 1 limits characters; the model truncates by tokens. Without this
    gate an over-long ad is silently cut short and nothing reports it."""
    path = tmp_path / "ads.jsonl"
    long_ad = make_ad(0, description=" ".join(["word"] * 40))  # 40+ tokens vs a limit of 16
    path.write_text(json.dumps(long_ad) + "\n", encoding="utf-8")

    with pytest.raises(embed.TruncationError) as exc:
        embed.run(path, tmp_path / "out", fake_embedder)

    assert exc.value.offenders[0][0] == "ad_00000"
    assert "ad_00000" in str(exc.value)


def test_token_gate_runs_before_embedding(tmp_path, fake_embedder):
    """Failing after running the model over 44k ads would waste the expensive
    part of the pipeline, so the check has to come first."""
    path = tmp_path / "ads.jsonl"
    path.write_text(json.dumps(make_ad(0, description=" ".join(["word"] * 40))) + "\n")

    with pytest.raises(embed.TruncationError):
        embed.run(path, tmp_path / "out", fake_embedder)

    assert fake_embedder.encode_calls == []  # the model was never invoked


def test_allow_truncation_overrides_the_gate(tmp_path, fake_embedder):
    path = tmp_path / "ads.jsonl"
    path.write_text(json.dumps(make_ad(0, description=" ".join(["word"] * 40))) + "\n")

    manifest = embed.run(path, tmp_path / "out", fake_embedder, allow_truncation=True)
    assert manifest["tokens"]["truncated"] == 1


def test_ads_within_the_limit_report_zero_truncations(clean_file, tmp_path, fake_embedder):
    manifest = embed.run(clean_file, tmp_path / "out", fake_embedder)
    assert manifest["tokens"]["truncated"] == 0
    assert manifest["tokens"]["max"] <= fake_embedder.max_tokens


# --- outputs --------------------------------------------------------------


def test_embeddings_are_written_as_float32_with_one_row_per_ad(clean_file, tmp_path, fake_embedder):
    out = tmp_path / "out"
    embed.run(clean_file, out, fake_embedder)

    vectors = np.load(out / "embeddings.npy")
    assert vectors.shape == (12, fake_embedder.dim)
    assert vectors.dtype == np.float32


def test_vectors_are_unit_length(clean_file, tmp_path, fake_embedder):
    """Normalisation is what lets Phase 4 use a dot product as cosine similarity."""
    out = tmp_path / "out"
    embed.run(clean_file, out, fake_embedder)

    norms = np.linalg.norm(np.load(out / "embeddings.npy"), axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)


def test_manifest_records_the_alignment_tripwires(clean_file, tmp_path, fake_embedder):
    from ingestion.fileutil import sha256_of_file

    manifest = embed.run(clean_file, tmp_path / "out", fake_embedder)

    assert manifest["count"] == 12
    assert manifest["source_sha256"] == sha256_of_file(clean_file)
    assert manifest["first_ad_id"] == "ad_00000"
    assert manifest["last_ad_id"] == "ad_00011"
    assert manifest["model"] == "fake/test-embedder"
    assert manifest["dim"] == fake_embedder.dim
    assert manifest["normalized"] is True


def test_manifest_is_written_to_disk(clean_file, tmp_path, fake_embedder):
    out = tmp_path / "out"
    embed.run(clean_file, out, fake_embedder)
    on_disk = json.loads((out / "embed_manifest.json").read_text())
    assert on_disk["count"] == 12


def test_batch_size_is_passed_through(clean_file, tmp_path, fake_embedder):
    embed.run(clean_file, tmp_path / "out", fake_embedder, batch_size=4)
    assert fake_embedder.encode_calls == [4]


def test_limit_only_embeds_the_first_n(clean_file, tmp_path, fake_embedder):
    manifest = embed.run(clean_file, tmp_path / "out", fake_embedder, limit=5)
    assert manifest["count"] == 5
    assert manifest["last_ad_id"] == "ad_00004"


def test_missing_input_is_reported_distinctly(tmp_path):
    assert embed.main(["--input", str(tmp_path / "nope.jsonl"), "--quiet"]) == embed.EXIT_INPUT_ERROR
