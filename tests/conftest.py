"""Shared test fixtures.

The important one is FakeEmbedder. Everything that can genuinely go wrong in
Phase 2 -- the token gate, row alignment, checksums, atomic publishing -- is
pipeline logic, not neural network behaviour. Testing it against a real model
would mean downloading 90MB and loading torch to assert things that have
nothing to do with either. So the fast suite runs against a stand-in, and only
tests marked `slow` touch the real model.
"""

import hashlib
import json

import numpy as np
import pytest


class FakeEmbedder:
    """Deterministic stand-in for the real model.

    Small dimensions keep test fixtures readable. Vectors are derived from a
    hash of the text, so they are stable across runs and different texts get
    different vectors -- enough to test alignment and integrity, which is all
    the pipeline logic cares about.
    """

    def __init__(self, dim: int = 8, max_tokens: int = 16) -> None:
        self.name = "fake/test-embedder"
        self.revision = "0" * 40
        self.dim = dim
        self.max_tokens = max_tokens
        self.device = "cpu"
        self.encode_calls: list[int] = []

    def count_tokens(self, texts: list[str]) -> list[int]:
        # Words plus two, standing in for the [CLS]/[SEP] the real tokenizer adds.
        return [len(text.split()) + 2 for text in texts]

    def encode(self, texts: list[str], batch_size: int) -> np.ndarray:
        self.encode_calls.append(batch_size)
        rows = []
        for text in texts:
            seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")
            vector = np.random.default_rng(seed).standard_normal(self.dim)
            rows.append(vector / np.linalg.norm(vector))  # unit length, like the real thing
        return np.ascontiguousarray(np.array(rows), dtype=np.float32)


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()


def make_ad(row: int, **overrides) -> dict:
    ad = {
        "ad_id": f"ad_{row:05d}",
        "headline": f"Premium running shoes model {row}",
        "description": "Tested over 500 miles.",
        "category": "footwear",
        "advertiser_id": "adv_0007",
        "bid_cpm_usd": 4.25,
        "created_at": "2026-03-14T09:21:00Z",
    }
    ad.update(overrides)
    return ad


@pytest.fixture
def clean_file(tmp_path):
    """A small stand-in for data/clean/ads_clean.jsonl."""
    path = tmp_path / "ads_clean.jsonl"
    path.write_text(
        "\n".join(json.dumps(make_ad(i)) for i in range(12)) + "\n", encoding="utf-8"
    )
    return path
