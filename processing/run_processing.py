"""
MOSAIC — Multi-Agent Orchestration System for Analyst Intelligence and Cognition
processing/run_processing.py

Entry point for the processing pipeline.

Chains three steps together:
    1. Chunk    — splits parsed documents into overlapping token windows
    2. Embed    — converts each chunk to a 1,536-dim vector via OpenAI
    3. Index    — stores vectors in FAISS for millisecond semantic search

Run this after run_ingestion.py has populated data/processed/ with JSON files.
The output is a searchable vector index at data/vector_store/ that agents
query at runtime without re-reading raw documents.

Usage:
    python -m processing.run_processing                   # all tickers, all form types
    python -m processing.run_processing --ticker MSFT     # single ticker
    python -m processing.run_processing --ticker MSFT GOOG --form-type 10-K

Cost awareness:
    Embedding costs money — text-embedding-3-small is ~$0.02 per million tokens.
    Our full corpus of ~15,000 chunks at ~1,000 tokens each is ~15M tokens total,
    so roughly $0.30 for the entire corpus. Cheap, but worth knowing.
    The pipeline skips re-embedding on re-runs by checking if chunks are already
    in the index — you only pay once per unique chunk.

Run standalone? Yes
Command: python -m processing.run_processing
"""

import argparse

from config.settings import settings
from config.logging_config import setup_logging, get_logger
from ingestion.document_store import load_all_documents, load_all_tickers
from processing.chunker import chunk_documents
from processing.embedder import embed_chunks, get_cache_stats
from processing.vector_store import vector_store

logger = get_logger(__name__)


# ── Core pipeline ──────────────────────────────────────────────────────────────

def run_pipeline(
    tickers:    list[str] | None = None,
    form_types: list[str] | None = None,
) -> None:
    """
    Runs the full processing pipeline — chunk, embed, index.

    Loads documents from disk, chunks them, embeds each chunk, and builds
    a FAISS index. The resulting index is saved to data/vector_store/ and
    is immediately available for agent semantic search.

    If tickers or form_types are specified, only those documents are
    processed. This is useful during development — processing just MSFT
    10-K takes 2 minutes vs 20+ minutes for the full corpus.

    NOTE: This always rebuilds the index from scratch for the specified
    scope. If you process MSFT only, the index will only contain MSFT
    chunks — not the previously indexed GOOG or AMZN data. To build a
    complete index, run without filters.

    Args:
        tickers:    List of tickers to process. Defaults to all configured.
        form_types: List of form types to include. Defaults to all configured.
    """
    tickers    = tickers    or list(settings.ticker_cik_map.keys())
    form_types = form_types or settings.filing_types

    logger.info("=" * 60)
    logger.info("MOSAIC Processing Pipeline Starting")
    logger.info(f"Tickers    : {tickers}")
    logger.info(f"Form types : {form_types}")
    logger.info("=" * 60)

    # ── Step 1: Load documents from disk ──────────────────────────────────────

    logger.info("\n[1/3] Loading documents from disk...")

    all_documents = []

    for ticker in tickers:
        for form_type in form_types:
            docs = load_all_documents(ticker, form_type)

            if not docs:
                logger.debug(f"  No documents found for {ticker} {form_type} — skipping")
                continue

            logger.info(f"  Loaded {len(docs):3d} documents — {ticker} {form_type}")
            all_documents.extend(docs)

    if not all_documents:
        logger.error(
            "No documents found on disk. "
            "Run python -m ingestion.run_ingestion first."
        )
        return

    logger.info(f"\nTotal documents loaded: {len(all_documents)}")

    # ── Step 2: Chunk all documents ────────────────────────────────────────────

    logger.info("\n[2/3] Chunking documents...")

    all_chunks = chunk_documents(all_documents)

    logger.info(f"Total chunks produced: {len(all_chunks):,}")

    # Quick sanity check — if average chunk count seems too low, the parser
    # may have produced very short documents that chunked to single pieces.
    avg_chunks_per_doc = len(all_chunks) / len(all_documents)
    if avg_chunks_per_doc < 5:
        logger.warning(
            f"Average of {avg_chunks_per_doc:.1f} chunks per document is low. "
            f"Documents may be too short — check parsing output in data/processed/"
        )

    # Show chunk distribution by form type — useful sanity check on video
    _log_chunk_distribution(all_chunks)

    # ── Step 3: Embed all chunks ───────────────────────────────────────────────

    logger.info("\n[3/3] Embedding chunks...")
    logger.info(
        f"Model   : {settings.embedding_model}\n"
        f"Chunks  : {len(all_chunks):,}\n"
        f"Est cost: ~${len(all_chunks) * 1000 / 1_000_000 * 0.02:.3f} USD"
    )

    all_chunks = embed_chunks(all_chunks)

    cache_stats = get_cache_stats()
    logger.info(
        f"Embedding complete — "
        f"{cache_stats['cached_embeddings']} vectors in cache, "
        f"~{cache_stats['estimated_memory_mb']}MB"
    )

    # ── Step 4: Build and save the FAISS index ─────────────────────────────────

    logger.info("\n[4/4] Building FAISS index...")

    vector_store.build(all_chunks)
    vector_store.save()

    # ── Final summary ──────────────────────────────────────────────────────────

    stats = vector_store.get_stats()

    logger.info("\n" + "=" * 60)
    logger.info("MOSAIC Processing Pipeline Complete")
    logger.info("=" * 60)
    logger.info(f"Total vectors indexed : {stats['total_vectors']:,}")
    logger.info(f"Embedding dimension   : {stats['dimension']}")
    logger.info("\nVectors by ticker:")
    for ticker, count in sorted(stats["by_ticker"].items()):
        logger.info(f"  {ticker:6s} — {count:,} chunks")
    logger.info("\nVectors by form type:")
    for form_type, count in sorted(stats["by_form_type"].items()):
        logger.info(f"  {form_type:6s} — {count:,} chunks")
    logger.info(
        f"\nIndex saved to: data/vector_store/"
        f"\nAgents are ready to search."
    )


def _log_chunk_distribution(chunks: list[dict]) -> None:
    """
    Logs a breakdown of chunk counts by ticker and form type.

    Helps spot immediately if a ticker or form type is missing from the
    corpus — e.g. if GOOG has zero chunks, the GOOG 10-K was not ingested.

    Args:
        chunks: Full list of chunk dicts from chunk_documents().
    """
    from collections import defaultdict

    distribution: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for chunk in chunks:
        ticker    = chunk.get("ticker", "unknown")
        form_type = chunk.get("form_type", "unknown")
        distribution[ticker][form_type] += 1

    logger.info("\nChunk distribution:")
    for ticker in sorted(distribution.keys()):
        for form_type, count in sorted(distribution[ticker].items()):
            logger.info(f"  {ticker:6s} {form_type:6s} — {count:,} chunks")


# ── CLI interface ──────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments for flexible pipeline runs.

    During development you will almost always use --ticker MSFT to iterate
    fast on one company before committing to the full corpus build.
    """
    parser = argparse.ArgumentParser(
        description="MOSAIC processing pipeline — chunk, embed, and index documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m processing.run_processing
  python -m processing.run_processing --ticker MSFT
  python -m processing.run_processing --ticker MSFT GOOG --form-type 10-K 10-Q
  python -m processing.run_processing --log-level DEBUG
        """,
    )

    parser.add_argument(
        "--ticker",
        nargs="+",
        help="Tickers to process. Defaults to all configured tickers.",
        metavar="TICKER",
    )

    parser.add_argument(
        "--form-type",
        nargs="+",
        dest="form_types",
        help="Form types to include. Defaults to all configured form types.",
        metavar="FORM_TYPE",
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )

    return parser.parse_args()


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = _parse_args()

    setup_logging(level=args.log_level)

    run_pipeline(
        tickers    = args.ticker,
        form_types = args.form_types,
    )