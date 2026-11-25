"""Exact and HNSW search: correctness, determinism and thread safety.

Recall at different HNSW settings is measured separately, against the real
bundle; these tests pin the behaviour that must hold at ANY setting.
"""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from retrieval import bundle as B
from retrieval.index import ExactIndex, HNSWConfig, HNSWIndex

N, DIM = 2000, 16
SMALL = HNSWConfig(m=8, ef_construction=40, ef_search=16)


def unit_vectors(n, dim, seed):
    v = np.random.default_rng(seed).standard_normal((n, dim)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


@pytest.fixture(scope="module")
def vectors():
    return unit_vectors(N, DIM, seed=7)


@pytest.fixture(scope="module")
def queries():
    return unit_vectors(50, DIM, seed=8)


@pytest.fixture(scope="module")
def exact(vectors):
    return ExactIndex(vectors)


@pytest.fixture(scope="module")
def hnsw(vectors):
    return HNSWIndex(vectors, SMALL)


def test_exact_matches_a_full_sort(exact, vectors, queries):
    scores, rows = exact.search(queries, 10)
    expected = np.argsort(-(queries @ vectors.T), axis=1)[:, :10]
    assert np.array_equal(rows, expected)
    assert np.all(np.diff(scores, axis=1) <= 0), "best first"


def test_hnsw_scores_are_dot_products(hnsw, vectors, queries):
    scores, rows = hnsw.search(queries, 10)
    expected = np.einsum("qd,qkd->qk", queries, vectors[rows])
    assert np.allclose(scores, expected, atol=1e-5)
    assert np.all(np.diff(scores, axis=1) <= 1e-6), "best first"


def test_hnsw_with_ef_covering_everything_equals_exact(vectors, exact, queries):
    """With a beam as wide as the corpus nothing is approximated, so any
    difference is a broken graph or search, not a recall trade-off."""
    wide = HNSWIndex(vectors, HNSWConfig(m=8, ef_construction=40, ef_search=N))
    _, approx = wide.search(queries, 10)
    _, truth = exact.search(queries, 10)
    assert np.array_equal(approx, truth)


def test_same_config_builds_the_same_graph(vectors, tmp_path):
    for name in ("a", "b"):
        HNSWIndex(vectors, SMALL)._index.save_index(str(tmp_path / name))
    assert (tmp_path / "a").read_bytes() == (tmp_path / "b").read_bytes()


def test_concurrent_searches_match_sequential(hnsw, queries):
    def run(i):
        row = i % len(queries)
        return hnsw.search(queries[row : row + 1], 10)[1][0]

    sequential = [run(i) for i in range(800)]
    with ThreadPoolExecutor(max_workers=16) as pool:
        concurrent = list(pool.map(run, range(800)))
    for want, got in zip(sequential, concurrent):
        assert np.array_equal(want, got)


@pytest.mark.parametrize("index_name", ["exact", "hnsw"])
def test_k_beyond_the_corpus_returns_every_row(vectors, index_name):
    small = ExactIndex(vectors[:5]) if index_name == "exact" else HNSWIndex(vectors[:5], SMALL)
    scores, rows = small.search(vectors[:1], 8)
    assert sorted(rows[0]) == [0, 1, 2, 3, 4]
    assert scores.shape == (1, 5)


@pytest.mark.parametrize("index_name", ["exact", "hnsw"])
def test_wrong_query_dimension_is_an_error(request, index_name):
    index = request.getfixturevalue(index_name)
    with pytest.raises(ValueError):
        index.search(np.ones((1, DIM + 1), dtype=np.float32), 5)


@pytest.mark.parametrize("config", [dict(m=1), dict(ef_construction=0), dict(ef_search=0)])
def test_invalid_config(config):
    with pytest.raises(ValueError):
        HNSWConfig(**config)


# --- the real bundle --------------------------------------------------------

PUBLISHED = Path("data/published")
GROUND_TRUTH = Path("data/eval/ground_truth.json")


def test_exact_search_agrees_with_the_python_ground_truth():
    """Exact search must reproduce probe_embeddings.py's brute force, since it
    is the reference HNSW recall is measured against."""
    if not (PUBLISHED / "LATEST").exists() or not GROUND_TRUTH.exists():
        pytest.skip("needs the published bundle and `make ground-truth`")
    loaded = B.load(PUBLISHED)
    truth = json.loads(GROUND_TRUTH.read_text())
    assert truth["bundle_version"] == loaded.version, "ground truth is stale: re-run make ground-truth"

    exact = ExactIndex(loaded.vectors)
    for query in truth["queries"]:
        scores, rows = exact.search(np.array([query["vector"]], dtype=np.float32), truth["k"])
        expected = query["top_k"]
        assert [loaded.ad_ids[r] for r in rows[0]] == [hit["ad_id"] for hit in expected]
        assert np.allclose(scores[0], [hit["score"] for hit in expected], atol=1e-5)
