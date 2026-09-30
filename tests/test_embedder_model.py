"""Tests that load the REAL model. Deselected by default; run with:

    uv run pytest -m slow

Everything here is about the model itself rather than pipeline logic, which is
why it is the only place worth paying a 90MB download and a torch import.
"""

import numpy as np
import pytest

from ingestion import embedder

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def model():
    # Imported inside the fixture, not at module level: pytest imports every
    # test file during collection, so a top-level import here would cost the
    # FAST suite ~4s of torch startup for tests it then deselects.
    from ingestion.st_embedder import SentenceTransformerEmbedder

    return SentenceTransformerEmbedder(device="cpu")


def test_model_identity_matches_what_the_manifest_will_claim(model):
    assert model.name == embedder.MODEL_NAME
    assert model.name == "sentence-transformers/all-MiniLM-L6-v2"
    assert model.revision == embedder.MODEL_REVISION
    assert model.revision == "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


def test_dimensions_and_token_limit_are_what_phase_4_assumes(model):
    assert model.dim == 384
    assert model.max_tokens == 256


def test_vectors_are_unit_length(model):
    vectors = model.encode(["running shoes", "tax software"], batch_size=8)
    assert vectors.shape == (2, 384)
    assert vectors.dtype == np.float32
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


def test_token_count_includes_the_special_tokens(model):
    """The [CLS] and [SEP] the model adds consume two of the 256 slots, so the
    gate in embed.py has to count them."""
    counts = model.count_tokens(["running shoes"])
    assert counts[0] == 4  # 2 words + [CLS] + [SEP]


def test_similar_text_scores_higher_than_unrelated_text(model):
    """The actual claim the whole phase rests on: distance in this space
    tracks meaning, which is what lets Phase 4 retrieve without keywords."""
    query = model.encode(["trail running shoes"], batch_size=8)[0]
    footwear = model.encode(
        ["Premium trail runners from Strider. Grip that holds on wet rock."], batch_size=8
    )[0]
    finance = model.encode(
        ["Northbank high-yield savings account. No monthly fee."], batch_size=8
    )[0]

    assert float(query @ footwear) > float(query @ finance)


def test_retrieval_works_without_shared_keywords(model):
    """"lower my tax bill" shares no words with the tax ad, and shares "my"
    with nothing. Keyword matching would find nothing here."""
    query = model.encode(["how do I lower my tax bill"], batch_size=8)[0]
    relevant = model.encode(
        ["Tallyhouse tax filing service. One flat rate, no monthly fee."], batch_size=8
    )[0]
    irrelevant = model.encode(
        ["Barkwell orthopedic dog bed. Machine washable cover."], batch_size=8
    )[0]

    assert float(query @ relevant) > float(query @ irrelevant)
