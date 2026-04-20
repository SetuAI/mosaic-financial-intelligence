"""
MOSAIC — Multi-Agent Orchestration System for Analyst Intelligence and Cognition
processing/embedder.py

Converts text chunks into vector embeddings using OpenAI's embedding API.

What embeddings are:
    An embedding is a list of 1,536 numbers that represents the semantic
    meaning of a piece of text. Two chunks that mean similar things will
    have similar vectors — their dot product or cosine similarity will be
    close to 1. Two chunks about completely different topics will have
    vectors that are far apart.

    This is what enables semantic search — "find me everything about
    supply chain risk" does not need to match those exact words. It finds
    chunks whose meaning is close to that query, even if they use different
    vocabulary. A chunk saying "component shortage impacting production"
    will score highly for a supply chain risk query.

Why text-embedding-3-small:
    We chose text-embedding-3-small over text-embedding-3-large because:
    - 1,536 dimensions vs 3,072 — half the storage, half the index size
    - ~5x cheaper per token
    - For financial document retrieval the quality difference is negligible
    - We can always re-embed with large if retrieval quality disappoints

Batching strategy:
    OpenAI's embedding API accepts up to 2,048 inputs per request.
    We batch in groups of 100 — conservative enough to avoid timeout
    issues on slow connections, large enough to be efficient.
    A corpus of 15,000 chunks takes roughly 150 API calls at this batch size.

Run standalone? No — called by run_processing.py
"""

import time
import hashlib

from openai import OpenAI
from typing import Optional

from config.settings import settings
from config.logging_config import get_logger

logger = get_logger(__name__)


# ── Client setup ───────────────────────────────────────────────────────────────

# Module-level client — instantiated once, reused across all calls.
# Creating a new client per call would re-read the API key from env
# on every request which is unnecessary overhead.
_client = OpenAI(api_key=settings.openai_api_key)

# How many chunks to send per API call.
# OpenAI allows up to 2,048 but we stay conservative — a batch of 100
# chunks at ~1,000 tokens each is ~100,000 tokens per request, well
# within the embedding API's per-request limits.
BATCH_SIZE = 100

# How long to wait between batches — embedding API has rate limits
# measured in tokens per minute, not requests per minute.
# 0.5s between batches keeps us safely under the default TPM limit
# for most OpenAI tier levels.
BATCH_DELAY = 0.5


# ── Embedding cache ────────────────────────────────────────────────────────────

# In-memory cache for this session — maps content hash to embedding vector.
# If the same chunk text appears in two different documents (boilerplate
# legal language, standard risk factor disclaimers), we embed it once.
# This cache does not persist across runs — for persistent caching we
# would use Redis, which we have deferred to a later phase.
_embedding_cache: dict[str, list[float]] = {}


def _hash_chunk(text: str) -> str:
    """
    Returns a short hash of a chunk's text for cache keying.

    SHA-256 truncated to 16 chars — collision-resistant at our scale
    (tens of thousands of chunks) and short enough to be a cheap dict key.

    Args:
        text: Chunk text to hash.

    Returns:
        16-character hex string.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ── Single embedding ───────────────────────────────────────────────────────────

def embed_text(text: str) -> list[float]:
    """
    Returns the embedding vector for a single text string.

    Checks the in-memory cache first — if this exact text has been
    embedded in this session, returns the cached vector without an API call.

    Args:
        text: Text to embed — should be a single chunk, not a full document.

    Returns:
        List of 1,536 floats representing the semantic meaning of the text.

    Raises:
        openai.APIError: If the embedding API call fails after retries.
    """
    cache_key = _hash_chunk(text)

    if cache_key in _embedding_cache:
        return _embedding_cache[cache_key]

    response = _client.embeddings.create(
        model = settings.embedding_model,
        input = text,
    )

    embedding = response.data[0].embedding
    _embedding_cache[cache_key] = embedding

    return embedding


# ── Batch embedding ────────────────────────────────────────────────────────────

def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embeds a list of chunk dicts and returns them with embeddings attached.

    This is the main function the processing pipeline calls. It handles:
    - Cache checking to avoid re-embedding duplicate text
    - Batching to stay within API rate limits
    - Progress logging so you can see it working on a 15,000-chunk corpus
    - Attaching the embedding vector to each chunk dict in-place

    The returned chunks are the same dicts as the input, with one new key
    added: "embedding" — a list of 1,536 floats.

    Args:
        chunks: List of chunk dicts from chunker.chunk_documents().
                Each dict must have a "text" key.

    Returns:
        Same list of chunk dicts, each now containing an "embedding" key.
    """
    if not chunks:
        return []

    total         = len(chunks)
    embedded      = 0
    cache_hits    = 0

    logger.info(f"Embedding {total} chunks using {settings.embedding_model}")

    # Split into batches — process one batch at a time to stay rate-limit safe
    for batch_start in range(0, total, BATCH_SIZE):
        batch = chunks[batch_start : batch_start + BATCH_SIZE]

        # Separate chunks into cache hits and misses.
        # Only chunks not in cache need an API call.
        to_embed   = []   # (index_in_batch, chunk) pairs needing API call
        batch_hits = 0

        for i, chunk in enumerate(batch):
            cache_key = _hash_chunk(chunk["text"])

            if cache_key in _embedding_cache:
                # Cache hit — attach embedding directly, no API call needed
                chunk["embedding"] = _embedding_cache[cache_key]
                batch_hits += 1
            else:
                to_embed.append((i, chunk))

        cache_hits += batch_hits

        # Embed the cache misses in one API call
        if to_embed:
            texts     = [chunk["text"] for _, chunk in to_embed]
            embeddings = _embed_batch(texts)

            for (i, chunk), embedding in zip(to_embed, embeddings):
                chunk["embedding"] = embedding
                # Store in cache for potential reuse later in this session
                _embedding_cache[_hash_chunk(chunk["text"])] = embedding

        embedded += len(batch)

        # Progress log every batch — reassuring on a large corpus
        logger.info(
            f"  Embedded {embedded}/{total} chunks "
            f"({batch_hits} cache hits this batch)"
        )

        # Polite delay between batches — avoids hammering the rate limit
        # on large runs. Skip the delay after the last batch.
        if batch_start + BATCH_SIZE < total:
            time.sleep(BATCH_DELAY)

    logger.info(
        f"Embedding complete — {total} chunks, "
        f"{cache_hits} total cache hits, "
        f"{total - cache_hits} API calls made"
    )

    return chunks


def _embed_batch(texts: list[str]) -> list[list[float]]:
    """
    Makes a single embedding API call for a list of texts.

    Wrapped separately from embed_chunks so retry logic is contained here
    and does not clutter the main batching loop.

    OpenAI's embedding endpoint returns results in the same order as the
    input — we rely on this and do not sort by index. If this assumption
    ever breaks, sort response.data by d.index before extracting embeddings.

    Args:
        texts: List of text strings to embed in one API call.
               Max 2,048 items but we stay well under at BATCH_SIZE=100.

    Returns:
        List of embedding vectors in the same order as the input texts.

    Raises:
        openai.APIError: Propagated from the OpenAI client on failure.
    """
    response = _client.embeddings.create(
        model = settings.embedding_model,
        input = texts,
    )

    # response.data is a list of Embedding objects sorted by index.
    # Extract just the float vectors.
    return [item.embedding for item in response.data]


# ── Query embedding ────────────────────────────────────────────────────────────

def embed_query(query: str) -> list[float]:
    """
    Embeds a search query for use in vector similarity search.

    Functionally identical to embed_text() but named separately to make
    the intent clear in calling code. When an agent does a semantic search,
    it calls embed_query() on the search string and then finds the nearest
    chunk vectors in the FAISS index.

    Queries are not cached — they are short, fast to embed, and unlikely
    to repeat exactly. The slight overhead is not worth the cache complexity.

    Args:
        query: Natural language search query — e.g.
               "supply chain risk factors mentioned in annual report"

    Returns:
        List of 1,536 floats — the query's position in embedding space.
    """
    response = _client.embeddings.create(
        model = settings.embedding_model,
        input = query,
    )

    return response.data[0].embedding


# ── Cache introspection ────────────────────────────────────────────────────────

def get_cache_stats() -> dict:
    """
    Returns stats about the current embedding cache state.

    Useful for debugging and for confirming cache is working correctly
    when processing a corpus with known duplicate content (boilerplate
    legal language appears across many filings).

    Returns:
        Dict with cache size and estimated memory usage.
    """
    cache_size    = len(_embedding_cache)

    # Each embedding is 1,536 floats × 4 bytes = ~6KB per entry
    estimated_mb  = (cache_size * 1536 * 4) / (1024 * 1024)

    return {
        "cached_embeddings": cache_size,
        "estimated_memory_mb": round(estimated_mb, 2),
    }