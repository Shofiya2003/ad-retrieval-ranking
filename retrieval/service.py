"""The gRPC face of retrieval: validate, clean, embed, search, map to ad ids.

The servicer is built from already-loaded parts (embedder, bundle, indexes),
so the fast tests can hand it the fake embedder and a tiny bundle and never
import torch.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import grpc
from grpc_health.v1 import health, health_pb2, health_pb2_grpc

from adretrieval.v1 import retrieval_pb2, retrieval_pb2_grpc
from ingestion.clean import clean_text
from ingestion.embedder import Embedder
from retrieval.bundle import Bundle
from retrieval.index import ExactIndex, HNSWIndex

SERVICE_NAME = retrieval_pb2.DESCRIPTOR.services_by_name["RetrievalService"].full_name
MAX_K = 1000


class ModelMismatchError(RuntimeError):
    """The query model is not the model that embedded the ads."""


class RetrievalServicer(retrieval_pb2_grpc.RetrievalServiceServicer):
    def __init__(
        self, embedder: Embedder, bundle: Bundle, hnsw: HNSWIndex, exact: ExactIndex
    ) -> None:
        # Query vectors from a different model than the ad vectors give
        # plausible-looking scores that mean nothing, so this fails closed.
        expected = (bundle.model["name"], bundle.model["revision"], bundle.dim)
        actual = (embedder.name, embedder.revision, embedder.dim)
        if actual != expected:
            raise ModelMismatchError(f"embedder is {actual}, bundle was embedded with {expected}")
        self._embedder = embedder
        self._bundle = bundle
        self._hnsw = hnsw
        self._exact = exact

    def Retrieve(self, request, context):
        if not 1 <= request.k <= MAX_K:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"k must be 1..{MAX_K}, got {request.k}")

        # Same normalisation the ads went through before they were embedded.
        text, _ = clean_text(request.query)
        if not text:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "query is empty")
        tokens = self._embedder.count_tokens([text])[0]
        if tokens > self._embedder.max_tokens:
            # Rejected rather than silently truncated, as in the ingestion token gate.
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"query is {tokens} tokens, the model's limit is {self._embedder.max_tokens}",
            )

        started = time.perf_counter()
        vector = self._embedder.encode([text], batch_size=1)
        embedded = time.perf_counter()
        index = self._exact if request.exact else self._hnsw
        scores, rows = index.search(vector, request.k)
        searched = time.perf_counter()

        ad_ids = self._bundle.ad_ids
        hits = [
            retrieval_pb2.Hit(ad_id=ad_ids[row], row=int(row), score=float(score))
            for score, row in zip(scores[0], rows[0])
        ]
        return retrieval_pb2.RetrieveResponse(
            hits=hits,
            bundle_version=self._bundle.version,
            embed_us=int((embedded - started) * 1e6),
            search_us=int((searched - embedded) * 1e6),
        )

    def GetInfo(self, request, context):
        config = self._hnsw.config
        return retrieval_pb2.GetInfoResponse(
            bundle_version=self._bundle.version,
            model_name=self._embedder.name,
            model_revision=self._embedder.revision,
            dim=self._bundle.dim,
            ad_count=self._bundle.count,
            index_kind=self._hnsw.kind,
            hnsw_m=config.m,
            hnsw_ef_construction=config.ef_construction,
            hnsw_ef_search_default=config.ef_search,
        )


def create_server(
    servicer: RetrievalServicer, address: str, threads: int
) -> tuple[grpc.Server, health.HealthServicer, int]:
    """Bind the retrieval and standard health services. Returns the bound port.

    Requests run on a thread pool. That gives real parallelism because hnswlib
    and torch release the GIL while they compute.

    SO_REUSEPORT is on by default in gRPC, which lets a second server bind the
    same port and silently split traffic with the first -- possibly one still
    serving an older bundle. Off, a second bind fails loudly.
    """
    server = grpc.server(
        ThreadPoolExecutor(max_workers=threads), options=[("grpc.so_reuseport", 0)]
    )
    retrieval_pb2_grpc.add_RetrievalServiceServicer_to_server(servicer, server)

    health_servicer = health.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    for name in ("", SERVICE_NAME):
        health_servicer.set(name, health_pb2.HealthCheckResponse.SERVING)

    port = server.add_insecure_port(address)
    return server, health_servicer, port
