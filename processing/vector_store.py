"""
MOSAIC — Multi-Agent Orchestration System for Analyst Intelligence and Cognition
processing/vector_store.py

Stores chunk embeddings in a FAISS index and provides semantic search
over the entire document corpus.

Why FAISS:
    FAISS (Facebook AI Similarity Search) is a battle-tested library for
    finding nearest neighbours in high-dimensional vector spaces. Given
    a query vector, it finds the K most similar vectors in the index in
    milliseconds — even across tens of thousands of chunks.

    We chose FAISS over managed services like Pinecone because:
    - Free and runs locally — no additional API keys or costs
    - When we move to AWS, we swap to pgvector on RDS — one config change
    - For our corpus size (~15,000 chunks) FAISS is more than fast enough
    - Students can run it without any external accounts

Index type — IndexFlatL2:
    We use the simplest FAISS index — exact L2 distance search.
    For approximate nearest neighbour indexes (IVF, HNSW) you trade
    some accuracy for speed at very large scale (millions of vectors).
    At 15,000 vectors, exact search is both fast enough and more accurate.
    We revisit this if the corpus grows beyond 100,000 chunks.

Persistence:
    The FAISS index and chunk metadata are saved to disk after every build.
    Loading a saved index takes milliseconds vs minutes to re-embed the
    full corpus. The index is rebuilt from scratch only when new documents
    are added to the corpus.

Run standalone? No — called by run_processing.py
"""

import os
import json
import faiss
import numpy as np

from pathlib import Path
from typing import Optional

from config.settings import settings
from config.logging_config import get_logger

logger = get_logger(__name__)


# ── Storage paths ──────────────────────────────────────────────────────────────

# The FAISS index binary — contains the raw vectors
FAISS_INDEX_PATH = Path("data/vector_store/faiss.index")

# The chunk metadata — maps vector position to chunk dict (without embedding)
# We store metadata separately because FAISS only stores vectors, not the
# text or metadata associated with each vector.
METADATA_PATH = Path("data/vector_store/chunk_metadata.json")

# Stats file — quick summary of what is in the index without loading it
STATS_PATH = Path("data/vector_store/stats.json")


# ── VectorStore class ──────────────────────────────────────────────────────────

class VectorStore:
    """
    Wraps a FAISS index with chunk metadata for semantic search.

    The FAISS index stores vectors indexed by integer position (0, 1, 2...).
    We maintain a parallel list of chunk metadata dicts at the same positions.
    A search returns positions, which we use to look up the corresponding
    metadata — text, ticker, form_type, filing_date, section_name.

    Usage:
        store = VectorStore()
        store.build(chunks)          # build from embedded chunks
        store.save()                 # persist to disk
        store.load()                 # restore from disk

        results = store.search(
            query_vector = embed_query("supply chain risk"),
            k            = 5,
        )
    """

    def __init__(self) -> None:
        # FAISS index — None until build() or load() is called
        self.index: Optional[faiss.Index] = None

        # Parallel list of chunk metadata — position N in this list
        # corresponds to vector N in the FAISS index
        self.metadata: list[dict] = []

        # Dimension of the embedding vectors — must match the embedding model.
        # Set from settings so changing the model in one place updates everything.
        self.dimension: int = settings.embedding_dimension

    # ── Build ──────────────────────────────────────────────────────────────────

    def build(self, chunks: list[dict]) -> None:
        """
        Builds the FAISS index from a list of embedded chunk dicts.

        Expects chunks that have already been through embedder.embed_chunks()
        — each chunk dict must have an "embedding" key containing a list
        of floats. Chunks without an embedding are skipped with a warning.

        This replaces any previously built index — call this once after
        processing the full corpus, not incrementally per document.
        Incremental FAISS updates are possible but add complexity we do
        not need at this scale.

        Args:
            chunks: List of chunk dicts with "embedding" keys attached.
        """
        logger.info(f"Building FAISS index from {len(chunks)} chunks")

        # Separate chunks that have embeddings from those that do not.
        # A chunk missing its embedding means the embedder failed for it —
        # we skip it rather than crashing the entire index build.
        valid_chunks = [c for c in chunks if "embedding" in c]
        skipped      = len(chunks) - len(valid_chunks)

        if skipped > 0:
            logger.warning(
                f"Skipping {skipped} chunks with missing embeddings — "
                f"re-run the embedder to fill gaps"
            )

        if not valid_chunks:
            raise ValueError(
                "No chunks with embeddings found. "
                "Run embedder.embed_chunks() before building the index."
            )

        # Stack embeddings into a float32 numpy matrix — FAISS requires this.
        # Shape: (num_chunks, embedding_dimension) = (~15000, 1536)
        # float32 is FAISS's native type — float64 would need conversion.
        vectors = np.array(
            [chunk["embedding"] for chunk in valid_chunks],
            dtype = np.float32,
        )

        # Validate dimensions — a mismatch here means the embedding model
        # changed between runs, which would produce nonsensical search results.
        actual_dim = vectors.shape[1]
        if actual_dim != self.dimension:
            raise ValueError(
                f"Embedding dimension mismatch: index expects {self.dimension} "
                f"but chunks have {actual_dim}. "
                f"Check settings.embedding_model and settings.embedding_dimension."
            )

        # IndexFlatL2 — exact L2 distance search, no approximation.
        # The right choice at our scale — fast enough and maximally accurate.
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(vectors)

        # Store metadata without the embedding — we do not need the raw
        # vectors in metadata since FAISS holds them. Dropping embeddings
        # keeps the metadata JSON lean and fast to load.
        self.metadata = [
            {k: v for k, v in chunk.items() if k != "embedding"}
            for chunk in valid_chunks
        ]

        logger.info(
            f"FAISS index built — {self.index.ntotal} vectors, "
            f"{self.dimension} dimensions"
        )

    # ── Search ─────────────────────────────────────────────────────────────────

    def search(
        self,
        query_vector: list[float],
        k: int = 5,
        ticker: Optional[str] = None,
        form_type: Optional[str] = None,
        section_name: Optional[str] = None,
    ) -> list[dict]:
        """
        Finds the K most semantically similar chunks to a query vector.

        The optional filters (ticker, form_type, section_name) are applied
        AFTER the FAISS search — FAISS itself does not support filtered search
        in IndexFlatL2. We over-fetch by 10x and then filter down to K results.

        This post-search filtering approach works well at our scale. At
        millions of vectors we would use a filtered index or a different
        architecture — not a concern for MOSAIC's current scope.

        Args:
            query_vector:  1,536-dim float list from embedder.embed_query().
            k:             Number of results to return. Default 5.
            ticker:        Optional — filter results to one company.
            form_type:     Optional — filter to "10-K", "10-Q", "8-K", or "4".
            section_name:  Optional — filter to a specific document section.

        Returns:
            List of chunk dicts sorted by relevance (most similar first),
            each with an added "similarity_score" key (0.0 to 1.0,
            higher = more similar).
        """
        if self.index is None:
            raise RuntimeError(
                "Vector store is empty — call build() or load() first."
            )

        # Over-fetch to account for post-search filtering.
        # If we filter to one ticker and that ticker has 20% of the corpus,
        # we need to fetch ~5x more results to guarantee k filtered results.
        # 10x is generous and adds negligible cost at our scale.
        fetch_k = min(k * 10, self.index.ntotal)

        query_array = np.array([query_vector], dtype=np.float32)

        # FAISS returns distances and indices of the nearest vectors.
        # distances: L2 distances (lower = more similar)
        # indices:   positions in the index (matches positions in self.metadata)
        distances, indices = self.index.search(query_array, fetch_k)

        results = []

        for distance, idx in zip(distances[0], indices[0]):
            # FAISS uses -1 as a sentinel for "not enough results"
            if idx == -1:
                continue

            chunk = self.metadata[idx].copy()

            # Apply optional filters — skip chunks that do not match
            if ticker and chunk.get("ticker") != ticker:
                continue
            if form_type and chunk.get("form_type") != form_type:
                continue
            if section_name and chunk.get("section_name") != section_name:
                continue

            # Convert L2 distance to a 0-1 similarity score.
            # L2 distance of 0 = identical vectors (similarity 1.0).
            # The conversion formula 1 / (1 + distance) maps [0, inf) to (0, 1].
            # This is a simple approximation — for cosine similarity we would
            # need to normalise vectors before indexing, which we can add later.
            chunk["similarity_score"] = round(1 / (1 + float(distance)), 4)
            results.append(chunk)

            if len(results) == k:
                break

        return results

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self) -> None:
        """
        Persists the FAISS index and chunk metadata to disk.

        Called once after build() completes. Loading from disk on subsequent
        runs takes milliseconds vs minutes to re-embed the full corpus.

        Saves three files:
            faiss.index          — the binary FAISS index (vectors)
            chunk_metadata.json  — chunk text and metadata (no vectors)
            stats.json           — quick summary for sanity checks
        """
        if self.index is None:
            raise RuntimeError("Nothing to save — build() has not been called.")

        # Create the storage directory if it does not exist
        FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)

        # Save the FAISS binary index
        faiss.write_index(self.index, str(FAISS_INDEX_PATH))
        logger.info(f"Saved FAISS index to {FAISS_INDEX_PATH}")

        # Save chunk metadata as JSON
        with open(METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False)
        logger.info(f"Saved {len(self.metadata)} chunk metadata records to {METADATA_PATH}")

        # Save a lightweight stats file — useful for quick inspection
        # without loading the full index and metadata
        stats = self._compute_stats()
        with open(STATS_PATH, "w") as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Vector store saved — {self.index.ntotal} vectors")

    def load(self) -> None:
        """
        Loads a previously saved FAISS index and metadata from disk.

        Call this at the start of an agent run instead of rebuilding
        the index from scratch. The loaded index is immediately ready
        for search() calls.

        Raises:
            FileNotFoundError: If the index files do not exist — run
                               run_processing.py first to build them.
        """
        if not FAISS_INDEX_PATH.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {FAISS_INDEX_PATH}. "
                f"Run: python -m processing.run_processing"
            )

        if not METADATA_PATH.exists():
            raise FileNotFoundError(
                f"Chunk metadata not found at {METADATA_PATH}. "
                f"Run: python -m processing.run_processing"
            )

        self.index    = faiss.read_index(str(FAISS_INDEX_PATH))
        logger.info(f"Loaded FAISS index — {self.index.ntotal} vectors")

        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        logger.info(f"Loaded {len(self.metadata)} chunk metadata records")

    # ── Introspection ──────────────────────────────────────────────────────────

    def _compute_stats(self) -> dict:
        """
        Computes a summary of what is in the index by ticker and form type.

        Used to populate stats.json and to give a quick corpus overview
        in the run_processing.py summary output.

        Returns:
            Dict with total vector count and per-ticker, per-form-type breakdown.
        """
        from collections import Counter

        ticker_counts    = Counter(c["ticker"]    for c in self.metadata)
        form_type_counts = Counter(c["form_type"] for c in self.metadata)
        section_counts   = Counter(c["section_name"] for c in self.metadata)

        return {
            "total_vectors":   self.index.ntotal if self.index else 0,
            "dimension":       self.dimension,
            "by_ticker":       dict(ticker_counts),
            "by_form_type":    dict(form_type_counts),
            "by_section":      dict(section_counts),
        }

    def get_stats(self) -> dict:
        """
        Returns index stats — works whether the index is built or loaded.

        Returns:
            Stats dict, or an empty dict if the index has not been built yet.
        """
        if self.index is None:
            return {"status": "not built"}

        return self._compute_stats()

    @property
    def is_ready(self) -> bool:
        """True if the index has been built or loaded and is ready for search."""
        return self.index is not None and self.index.ntotal > 0


# ── Module-level singleton ─────────────────────────────────────────────────────

# One shared VectorStore instance for the entire application.
# Agents import this directly:
#     from processing.vector_store import vector_store
#     results = vector_store.search(embed_query("cloud revenue growth"))
#
# The instance starts empty — agents must call vector_store.load() at startup,
# or run_processing.py calls build() + save() to populate it.
vector_store = VectorStore()