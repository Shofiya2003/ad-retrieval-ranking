"""The real sentence-transformers model.

Isolated in its own module so that importing `ingestion.embedder` -- which
every stage and every test does -- does not drag in torch. Imports here are at
the top of the file as normal; the laziness lives in which module you import,
not in a function-level import.
"""

from __future__ import annotations

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from ingestion.embedder import MODEL_NAME, MODEL_REVISION


def resolve_device(requested: str = "auto") -> str:
    """Pick the compute device.

    On this Apple M4, "mps" routes the matrix maths through the GPU via Metal.
    Note that mps and cpu do not produce bit-identical floats -- the operations
    are reassociated differently -- so vectors are compared with a tolerance,
    never with ==.
    """
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class SentenceTransformerEmbedder:
    """Implements the Embedder protocol using the pinned MiniLM model."""

    def __init__(
        self,
        device: str = "auto",
        model_name: str = MODEL_NAME,
        revision: str = MODEL_REVISION,
    ) -> None:
        self.name = model_name
        self.revision = revision
        self.device = resolve_device(device)
        self._model = SentenceTransformer(model_name, revision=revision, device=self.device)
        self.dim = self._model.get_embedding_dimension()
        self.max_tokens = self._model.max_seq_length

    def count_tokens(self, texts: list[str]) -> list[int]:
        """Length in TOKENS, which is the unit the model actually truncates by.

        Phase 1's limits are in characters, so this is the one place the two
        unit systems get reconciled. add_special_tokens=True counts the [CLS]
        and [SEP] the model wraps around every input, because they occupy two
        of the 256 slots just like any other token.
        """
        encoded = self._model.tokenizer(texts, add_special_tokens=True)["input_ids"]
        return [len(ids) for ids in encoded]

    def encode(self, texts: list[str], batch_size: int) -> np.ndarray:
        """Turn texts into unit-length vectors, `batch_size` texts at a time.

        WHY NORMALIZE (normalize_embeddings=True)
        -----------------------------------------
        The model returns vectors of assorted lengths. Normalising divides each
        one by its own length, so every vector ends up exactly 1.0 long,
        pointing in the same direction as before. Only the direction carries
        meaning; the length is an artifact.

        That matters because cosine similarity is defined as

            cos(a, b) = (a . b) / (|a| * |b|)

        When |a| and |b| are both 1, the denominator is 1 and the formula
        collapses to just `a . b`. So Phase 4 can compare a query to an ad with
        a plain dot product -- multiply pairwise, add up -- instead of also
        computing two magnitudes and a square root on every single comparison.
        Over millions of comparisons per second in the hot path, that is the
        difference between a cheap inner loop and an expensive one.

        WHY BATCH
        ---------
        Two separate reasons:

        1. Hardware. A GPU is thousands of small cores that want one big job,
           not thousands of little ones. Handing it 256 ads as a single matrix
           multiply uses those cores at once; handing it one ad 256 times pays
           the scheduling and CPU-to-GPU transfer overhead 256 times over, with
           most of the chip idle in between.

        2. Padding. Every text in a batch must be the same length, so shorter
           ones are padded out to the longest in that batch -- and the model
           computes over the padding too. One 55-token ad batched with 255
           five-token ads pads everything to 55, wasting roughly 90% of the
           work. sentence-transformers sorts inputs by length internally so
           similar lengths land together, which keeps that waste small.
        """
        vectors = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.ascontiguousarray(vectors, dtype=np.float32)
