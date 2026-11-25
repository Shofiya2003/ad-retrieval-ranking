"""The retrieval service with the REAL model and the real bundle. Deselected by
default; run with `uv run pytest -m slow`.

The question here is whether the online path embeds a query exactly as the
offline path did. If it didn't, every score would be quietly off.
"""

import json
from pathlib import Path

import grpc
import numpy as np
import pytest

from adretrieval.v1 import retrieval_pb2 as pb
from adretrieval.v1 import retrieval_pb2_grpc
from ingestion.embedder import load_embedder
from retrieval import bundle as B
from retrieval.index import ExactIndex, HNSWIndex
from retrieval.service import RetrievalServicer, create_server

pytestmark = pytest.mark.slow

PUBLISHED = Path("data/published")
GROUND_TRUTH = Path("data/eval/ground_truth.json")


@pytest.fixture(scope="module")
def setup():
    if not (PUBLISHED / "LATEST").exists() or not GROUND_TRUTH.exists():
        pytest.skip("needs the published bundle and `make ground-truth`")
    loaded = B.load(PUBLISHED)
    truth = json.loads(GROUND_TRUTH.read_text())
    assert truth["bundle_version"] == loaded.version, "ground truth is stale: re-run make ground-truth"
    embedder = load_embedder(
        device="cpu", model_name=loaded.model["name"], revision=loaded.model["revision"]
    )
    return loaded, truth, embedder


@pytest.fixture(scope="module")
def stub(setup):
    loaded, _, embedder = setup
    servicer = RetrievalServicer(
        embedder, loaded, HNSWIndex(loaded.vectors), ExactIndex(loaded.vectors)
    )
    server, _, port = create_server(servicer, "127.0.0.1:0", threads=4)
    server.start()
    channel = grpc.insecure_channel(f"127.0.0.1:{port}")
    yield retrieval_pb2_grpc.RetrievalServiceStub(channel)
    channel.close()
    server.stop(grace=None)


def test_query_vectors_match_the_offline_ground_truth(setup):
    """Tolerance rather than ==: the ground truth may have been embedded on
    MPS, and MPS and CPU differ in the last bits."""
    _, truth, embedder = setup
    for query in truth["queries"]:
        online = embedder.encode([query["query"]], batch_size=1)[0]
        assert np.abs(online - np.array(query["vector"], dtype=np.float32)).max() < 1e-4


def test_exact_rpc_returns_the_ground_truth_top_k(setup, stub):
    _, truth, _ = setup
    for query in truth["queries"]:
        response = stub.Retrieve(pb.RetrieveRequest(query=query["query"], k=truth["k"], exact=True))
        got = {hit.ad_id for hit in response.hits}
        want = {hit["ad_id"] for hit in query["top_k"]}
        # Device differences can reorder near-ties at the cut-off, never more.
        assert len(got & want) >= truth["k"] - 1, query["query"]
