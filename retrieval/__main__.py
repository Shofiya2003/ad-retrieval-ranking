"""Serve retrieval for the latest bundle over gRPC.

    uv run python -m retrieval --root data/published --port 50051

Startup is ordered so the port is only bound once every check has passed and
the first request will be as fast as the rest:

    verify bundle -> load the model the manifest names -> check it matches
    -> build indexes -> warm up -> bind and serve

A client that can connect can therefore be served. On SIGTERM/SIGINT the
health status flips to NOT_SERVING first, so load balancers stop routing new
requests, then in-flight requests get a grace period to finish.
"""

from __future__ import annotations

import argparse
import signal
import sys
import threading
import time
from pathlib import Path

import torch

from ingestion.embedder import load_embedder
from retrieval import bundle as bundle_mod
from retrieval.index import ExactIndex, HNSWConfig, HNSWIndex
from retrieval.service import ModelMismatchError, RetrievalServicer, create_server

EXIT_OK = 0
EXIT_VERIFY_FAILED = 1
EXIT_INPUT_ERROR = 2

SHUTDOWN_GRACE_S = 5.0


def build_parser() -> argparse.ArgumentParser:
    defaults = HNSWConfig()
    parser = argparse.ArgumentParser(prog="python -m retrieval", description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path("data/published"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--threads", type=int, default=8, help="gRPC worker threads")
    parser.add_argument("--device", default="cpu", help="cpu | mps | cuda | auto")
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=1,
        help="intra-op threads per request; parallelism comes from --threads",
    )
    parser.add_argument("--m", type=int, default=defaults.m)
    parser.add_argument("--ef-construction", type=int, default=defaults.ef_construction)
    parser.add_argument("--ef-search", type=int, default=defaults.ef_search)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.root.is_dir():
        print(f"error: bundle root {args.root} does not exist", file=sys.stderr)
        return EXIT_INPUT_ERROR

    try:
        started = time.perf_counter()
        bundle = bundle_mod.load(args.root)
        loaded = time.perf_counter()

        model = bundle.model
        embedder = load_embedder(
            device=args.device, model_name=model["name"], revision=model["revision"]
        )
        torch.set_num_threads(args.torch_threads)
        model_loaded = time.perf_counter()

        config = HNSWConfig(
            m=args.m, ef_construction=args.ef_construction, ef_search=args.ef_search, seed=args.seed
        )
        hnsw = HNSWIndex(bundle.vectors, config)
        exact = ExactIndex(bundle.vectors)
        built = time.perf_counter()

        servicer = RetrievalServicer(embedder, bundle, hnsw, exact)
    except (bundle_mod.BundleError, ModelMismatchError) as err:
        print(f"refusing to serve: {err}", file=sys.stderr)
        return EXIT_VERIFY_FAILED

    # The first encode allocates and initialises; that cost shouldn't land on
    # a real query.
    warm = embedder.encode(["warm up"], batch_size=1)
    hnsw.search(warm, 10)
    exact.search(warm, 10)

    try:
        server, health_servicer, port = create_server(
            servicer, f"{args.host}:{args.port}", args.threads
        )
    except RuntimeError as err:
        print(f"error: {err}", file=sys.stderr)
        return EXIT_INPUT_ERROR
    server.start()

    print(f"bundle    {bundle.version}   {bundle.count:,} x {bundle.dim}   verified in {loaded - started:.2f}s")
    print(f"model     {model['name']} @ {model['revision'][:12]}   {embedder.device}   loaded in {model_loaded - loaded:.2f}s")
    print(
        f"index     hnsw M={config.m} efC={config.ef_construction} ef={config.ef_search}"
        f"   built in {built - model_loaded:.2f}s"
    )
    print(f"serving   {args.host}:{port}   {args.threads} threads", flush=True)

    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    stop.wait()

    health_servicer.enter_graceful_shutdown()
    server.stop(grace=SHUTDOWN_GRACE_S).wait()
    print("stopped", flush=True)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
