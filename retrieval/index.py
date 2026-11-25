"""Nearest-neighbour search over the bundle's vectors, exact and approximate.

Both indexes score by inner product. The vectors are unit length (the bundle
loader checked), so inner product is cosine similarity.

`ExactIndex` compares the query with every ad. It is the ground truth that the
approximate index's recall is measured against, and the fallback when the
approximate index is suspected of being wrong.

`HNSWIndex` is hnswlib's HNSW graph. Its knobs trade different things:
  * m               links per node: more means better recall and more memory
  * ef_construction beam width while building: better graph, slower build
  * ef_search       beam width per query: better recall, slower query
m and ef_construction are baked into the graph. ef_search is a setting on the
index, fixed at startup: hnswlib has no per-query ef, and changing it while
queries run would race with them.

Why hnswlib rather than faiss: the faiss and torch wheels each bundle their own
OpenMP runtime, and with both in one process the first parallel region crashes
(macOS, faiss-cpu 1.16 + torch 2.14). hnswlib uses plain threads, no OpenMP.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import hnswlib
import numpy as np


class Index(Protocol):
    kind: str

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (scores, rows), each (len(queries), min(k, count)), best first."""


@dataclass(frozen=True)
class HNSWConfig:
    m: int = 16
    ef_construction: int = 200
    ef_search: int = 64
    seed: int = 42

    def __post_init__(self) -> None:
        if self.m < 2 or self.ef_construction < 1 or self.ef_search < 1:
            raise ValueError(f"invalid HNSW config {self}")


def _as_queries(queries: np.ndarray, dim: int) -> np.ndarray:
    queries = np.ascontiguousarray(queries, dtype=np.float32)
    if queries.ndim != 2 or queries.shape[1] != dim:
        raise ValueError(f"queries have shape {queries.shape}, want (n, {dim})")
    return queries


class ExactIndex:
    kind = "exact"

    def __init__(self, vectors: np.ndarray) -> None:
        self._vectors = vectors
        self.count, self.dim = vectors.shape

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        k = min(k, self.count)
        scores = _as_queries(queries, self.dim) @ self._vectors.T
        # argpartition finds the top k in O(N); only those k get sorted.
        top = np.argpartition(-scores, k - 1, axis=1)[:, :k]
        top_scores = np.take_along_axis(scores, top, axis=1)
        order = np.argsort(-top_scores, axis=1, kind="stable")
        return np.take_along_axis(top_scores, order, axis=1), np.take_along_axis(top, order, axis=1)


class HNSWIndex:
    kind = "hnsw"

    def __init__(self, vectors: np.ndarray, config: HNSWConfig = HNSWConfig()) -> None:
        self.count, self.dim = vectors.shape
        self.config = config
        self._index = hnswlib.Index(space="ip", dim=self.dim)
        self._index.init_index(
            max_elements=self.count,
            ef_construction=config.ef_construction,
            M=config.m,
            random_seed=config.seed,
        )
        # One thread: parallel insertion order is nondeterministic, and the
        # same bundle + config should always build the same graph so recall
        # numbers reproduce.
        self._index.add_items(vectors, np.arange(self.count), num_threads=1)
        self._index.set_ef(config.ef_search)

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        k = min(k, self.count)
        # num_threads=1: concurrency comes from the gRPC pool. hnswlib's
        # default would fan every single query out over all cores.
        rows, distances = self._index.knn_query(
            _as_queries(queries, self.dim), k=k, num_threads=1
        )
        # hnswlib's "ip" space reports 1 - dot product as a distance.
        return 1.0 - distances, rows.astype(np.int64)
