"""The gRPC service end to end, over a real local server, with the fake embedder.

These go through the network stack on purpose: status codes, the health
protocol and concurrent calls are properties of the server, not of the
servicer class on its own.
"""

from concurrent.futures import ThreadPoolExecutor

import grpc
import pytest
from grpc_health.v1 import health_pb2, health_pb2_grpc

from adretrieval.v1 import retrieval_pb2 as pb
from adretrieval.v1 import retrieval_pb2_grpc
from ingestion.embed import compose_text
from retrieval import bundle as B
from retrieval.index import ExactIndex, HNSWConfig, HNSWIndex
from retrieval.service import (
    MAX_K,
    SERVICE_NAME,
    ModelMismatchError,
    RetrievalServicer,
    create_server,
)
from tests.conftest import FakeEmbedder, make_ad

CONFIG = HNSWConfig(m=4, ef_construction=20, ef_search=16)


@pytest.fixture
def loaded(published_root):
    return B.load(published_root)


@pytest.fixture
def server(loaded, fake_embedder):
    servicer = RetrievalServicer(
        fake_embedder, loaded, HNSWIndex(loaded.vectors, CONFIG), ExactIndex(loaded.vectors)
    )
    srv, health_servicer, port = create_server(servicer, "127.0.0.1:0", threads=8)
    srv.start()
    channel = grpc.insecure_channel(f"127.0.0.1:{port}")
    yield retrieval_pb2_grpc.RetrievalServiceStub(channel), channel, health_servicer
    channel.close()
    srv.stop(grace=None)


@pytest.fixture
def stub(server):
    return server[0]


def ad_text(row):
    return compose_text(make_ad(row))


def test_an_ads_own_text_retrieves_that_ad_first(stub, loaded):
    response = stub.Retrieve(pb.RetrieveRequest(query=ad_text(3), k=5))

    assert response.hits[0].ad_id == "ad_00003"
    assert response.hits[0].score == pytest.approx(1.0, abs=1e-5)
    assert response.bundle_version == loaded.version
    assert len(response.hits) == 5


def test_hits_are_best_first_and_ids_match_rows(stub, loaded):
    response = stub.Retrieve(pb.RetrieveRequest(query="running shoes", k=12))
    scores = [hit.score for hit in response.hits]
    assert scores == sorted(scores, reverse=True)
    for hit in response.hits:
        assert hit.ad_id == loaded.ad_ids[hit.row]


def test_k_beyond_the_corpus_returns_every_ad(stub):
    response = stub.Retrieve(pb.RetrieveRequest(query="running shoes", k=50))
    assert sorted(hit.row for hit in response.hits) == list(range(12))


def test_exact_and_hnsw_agree_on_a_tiny_corpus(stub):
    request = dict(query="running shoes", k=12)
    hnsw = stub.Retrieve(pb.RetrieveRequest(**request))
    exact = stub.Retrieve(pb.RetrieveRequest(**request, exact=True))
    assert [h.ad_id for h in hnsw.hits] == [h.ad_id for h in exact.hits]


def test_query_is_cleaned_like_the_ads_were(stub):
    clean = stub.Retrieve(pb.RetrieveRequest(query=ad_text(2), k=3))
    messy = stub.Retrieve(pb.RetrieveRequest(query=f"  <b>{ad_text(2)}</b>​ ", k=3))
    assert [h.ad_id for h in messy.hits] == [h.ad_id for h in clean.hits]


def test_timings_are_reported(stub):
    response = stub.Retrieve(pb.RetrieveRequest(query="running shoes", k=3))
    assert response.embed_us > 0
    assert response.search_us > 0


@pytest.mark.parametrize(
    "request_fields",
    [
        pytest.param(dict(query="", k=5), id="empty"),
        pytest.param(dict(query="   \t ", k=5), id="whitespace"),
        pytest.param(dict(query="<b></b>", k=5), id="only-markup"),
        pytest.param(dict(query="word " * 20, k=5), id="over-token-limit"),
        pytest.param(dict(query="shoes", k=0), id="k-zero"),
        pytest.param(dict(query="shoes", k=MAX_K + 1), id="k-too-large"),
    ],
)
def test_bad_requests_are_invalid_argument(stub, request_fields):
    with pytest.raises(grpc.RpcError) as err:
        stub.Retrieve(pb.RetrieveRequest(**request_fields))
    assert err.value.code() == grpc.StatusCode.INVALID_ARGUMENT


def test_get_info_describes_what_is_served(stub, loaded):
    info = stub.GetInfo(pb.GetInfoRequest())
    assert info.bundle_version == loaded.version
    assert (info.model_name, info.model_revision) == ("fake/test-embedder", "0" * 40)
    assert (info.dim, info.ad_count) == (8, 12)
    assert (info.index_kind, info.hnsw_m, info.hnsw_ef_construction, info.hnsw_ef_search_default) == (
        "hnsw", 4, 20, 16,
    )


def test_health_flips_to_not_serving_on_shutdown(server):
    _, channel, health_servicer = server
    health = health_pb2_grpc.HealthStub(channel)
    for name in ("", SERVICE_NAME):
        status = health.Check(health_pb2.HealthCheckRequest(service=name)).status
        assert status == health_pb2.HealthCheckResponse.SERVING

    health_servicer.enter_graceful_shutdown()
    status = health.Check(health_pb2.HealthCheckRequest(service=SERVICE_NAME)).status
    assert status == health_pb2.HealthCheckResponse.NOT_SERVING


def test_concurrent_calls_return_what_sequential_calls_do(stub):
    queries = [ad_text(i % 12) for i in range(200)]

    def call(query):
        return [h.ad_id for h in stub.Retrieve(pb.RetrieveRequest(query=query, k=5)).hits]

    sequential = [call(q) for q in queries]
    with ThreadPoolExecutor(max_workers=16) as pool:
        assert list(pool.map(call, queries)) == sequential


@pytest.mark.parametrize(
    "field, value", [("name", "some/other-model"), ("revision", "f" * 40), ("dim", 16)]
)
def test_refuses_an_embedder_that_did_not_embed_the_ads(loaded, field, value):
    embedder = FakeEmbedder()
    setattr(embedder, field, value)
    with pytest.raises(ModelMismatchError):
        RetrievalServicer(
            embedder, loaded, HNSWIndex(loaded.vectors, CONFIG), ExactIndex(loaded.vectors)
        )


def test_a_second_server_cannot_share_the_port(loaded, fake_embedder):
    """With SO_REUSEPORT (gRPC's default) both would bind and split traffic."""
    servicer = RetrievalServicer(
        fake_embedder, loaded, HNSWIndex(loaded.vectors, CONFIG), ExactIndex(loaded.vectors)
    )
    first, _, port = create_server(servicer, "127.0.0.1:0", threads=1)
    first.start()
    try:
        with pytest.raises(RuntimeError):
            create_server(servicer, f"127.0.0.1:{port}", threads=1)
    finally:
        first.stop(grace=None)
