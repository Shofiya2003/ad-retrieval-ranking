"""The embedding model, behind a small interface.

`embed.py` depends on the Embedder protocol below rather than on
sentence-transformers directly, for two practical reasons:

  1. Everything that can genuinely go wrong in this stage -- the token gate,
     row alignment, checksums, publishing -- is pipeline logic, not neural
     network behaviour. The fast test suite plugs in a deterministic fake and
     tests all of it without a model.
  2. Importing sentence_transformers costs ~4 seconds (measured: 3.28s for
     sentence_transformers plus 0.73s for torch). This module stays free of
     that cost so that importing it -- which embed.py, publish.py and every
     test does -- stays instant. The whole fast suite runs in 0.11s.

The heavy stack lives in `ingestion/st_embedder.py` and is imported in exactly
one place: `load_embedder()` at the bottom of this file.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

# Pinned to an exact revision, not just a name.
#
# This is deliberately a constant in version control rather than an environment
# variable. It is not a secret and not deployment-specific -- it is a
# CORRECTNESS constraint. An env var could differ between the machine that
# embeds ads and the machine that embeds queries, which is exactly the silent
# drift the pin exists to prevent. In git, changing it shows up in a diff.
#
# It is still overridable per run (`--model` / `--revision`) for experiments,
# and whatever was actually used is recorded in the published manifest.
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


@runtime_checkable
class Embedder(Protocol):
    """What embed.py actually needs from a model."""

    name: str
    revision: str
    dim: int
    max_tokens: int

    def count_tokens(self, texts: list[str]) -> list[int]:
        """Token length of each text INCLUDING the model's special tokens."""

    def encode(self, texts: list[str], batch_size: int) -> np.ndarray:
        """Return a (len(texts), dim) float32 array of unit-length vectors."""


def load_embedder(
    device: str = "auto",
    model_name: str = MODEL_NAME,
    revision: str = MODEL_REVISION,
) -> Embedder:
    """Construct the real model.

    The only deferred import in the package. PEP 8 wants imports at the top of
    a module, and the rest of the codebase follows that; this is the documented
    exception for an expensive import. Putting it at module level would add ~4s
    to every test run, every `--help`, and every import of embed.py.

    Keeping it to a single site means there is one place to look, rather than
    the same import scattered through several `main()` functions.
    """
    from ingestion.st_embedder import SentenceTransformerEmbedder

    return SentenceTransformerEmbedder(
        device=device, model_name=model_name, revision=revision
    )
