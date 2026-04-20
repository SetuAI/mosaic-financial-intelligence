"""
MOSAIC — Multi-Agent Orchestration System for Analyst Intelligence and Cognition
ingestion/document_store.py

Handles all document persistence — saving parsed filings to disk and
loading them back when agents need them.

This module deliberately knows nothing about EDGAR or parsing. Its only
job is storage and retrieval. That clean boundary means when we move
from local disk to S3, this is the only file that changes — every other
module keeps calling the same functions.

Storage layout on disk:
    data/
    ├── raw/
    │   └── {TICKER}/{FORM_TYPE}/{FILING_DATE}_{ACCESSION}.html
    └── processed/
        └── {TICKER}/{FORM_TYPE}/{FILING_DATE}_{ACCESSION}.json

The processed JSON is what agents consume. Raw HTML is kept as a safety
net — if the parser has a bug, we can re-parse from raw without re-fetching
from EDGAR.
"""

import json
import os

from pathlib import Path
from typing import Optional

from config.settings import settings
from config.logging_config import get_logger

logger = get_logger(__name__)


# ── Path helpers ───────────────────────────────────────────────────────────────

def _raw_path(ticker: str, form_type: str, filing_date: str, accession_number: str) -> Path:
    """
    Builds the full file path for a raw HTML filing.

    Using pathlib.Path throughout — it handles OS path separators correctly
    on both Mac/Linux and Windows without any special casing.

    Args:
        ticker:           Stock ticker — e.g. "MSFT".
        form_type:        SEC form type — "10-K", "10-Q", "8-K", "4".
        filing_date:      Filing date string — "2024-07-30".
        accession_number: EDGAR accession number.

    Returns:
        Full Path object for where this raw file should be stored.
    """
    # Sanitise the accession number for use in a filename —
    # replace dashes with underscores so it does not look like a flag.
    safe_accession = accession_number.replace("-", "_")
    filename       = f"{filing_date}_{safe_accession}.html"

    # form_type goes into the path so we can list all 10-Ks for a ticker
    # without filtering — the directory structure does the filtering for us.
    return Path(settings.raw_data_path) / ticker / form_type / filename


def _processed_path(ticker: str, form_type: str, filing_date: str, accession_number: str) -> Path:
    """
    Builds the full file path for a processed JSON document.

    Mirror of _raw_path() but points to the processed/ directory
    and uses .json extension.
    """
    safe_accession = accession_number.replace("-", "_")
    filename       = f"{filing_date}_{safe_accession}.json"

    return Path(settings.processed_data_path) / ticker / form_type / filename


# ── Save operations ────────────────────────────────────────────────────────────

def save_raw_filing(
    raw_html: str,
    ticker: str,
    form_type: str,
    filing_date: str,
    accession_number: str,
) -> Path:
    """
    Saves raw HTML from EDGAR to disk before parsing.

    We always save the raw file first. If parsing fails halfway through,
    we have the original and can re-parse without hitting EDGAR again.
    This is especially important for large 10-K filings that take several
    seconds to download.

    Args:
        raw_html:         Raw HTML string from EDGAR.
        ticker:           Stock ticker.
        form_type:        SEC form type.
        filing_date:      Filing date string.
        accession_number: EDGAR accession number.

    Returns:
        Path where the file was saved.
    """
    path = _raw_path(ticker, form_type, filing_date, accession_number)

    # Create the directory tree if it does not exist yet.
    # exist_ok=True means no error if the directory is already there —
    # important because we call this in a loop and the dir exists after
    # the first filing for a ticker.
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(raw_html, encoding="utf-8")

    logger.debug(f"Saved raw filing: {path}")
    return path


def save_processed_document(document: dict) -> Path:
    """
    Saves a parsed document dict to disk as JSON.

    Called by run_ingestion.py after document_parser.py returns a
    structured document. The document dict already contains all the
    metadata we need to build the file path.

    Args:
        document: Parsed document dict from document_parser.parse_filing().
                  Must contain: ticker, form_type, filing_date, accession_number.

    Returns:
        Path where the JSON was saved.
    """
    path = _processed_path(
        ticker           = document["ticker"],
        form_type        = document["form_type"],
        filing_date      = document["filing_date"],
        accession_number = document["accession_number"],
    )

    path.parent.mkdir(parents=True, exist_ok=True)

    # indent=2 keeps the JSON human-readable — useful when debugging
    # a parsing issue and you want to open the file and read it directly.
    # ensure_ascii=False preserves any unicode characters in the filing text.
    with open(path, "w", encoding="utf-8") as f:
        json.dump(document, f, indent=2, ensure_ascii=False)

    logger.debug(f"Saved processed document: {path} ({document['char_count']:,} chars)")
    return path


# ── Load operations ────────────────────────────────────────────────────────────

def load_processed_document(
    ticker: str,
    form_type: str,
    filing_date: str,
    accession_number: str,
) -> Optional[dict]:
    """
    Loads a single processed document by its identifiers.

    Returns None rather than raising if the file does not exist —
    callers can decide whether a missing file is an error or just means
    the filing has not been ingested yet.

    Args:
        ticker:           Stock ticker.
        form_type:        SEC form type.
        filing_date:      Filing date string.
        accession_number: EDGAR accession number.

    Returns:
        Parsed document dict, or None if the file does not exist.
    """
    path = _processed_path(ticker, form_type, filing_date, accession_number)

    if not path.exists():
        logger.debug(f"Document not found on disk: {path}")
        return None

    with open(path, "r", encoding="utf-8") as f:
        document = json.load(f)

    logger.debug(f"Loaded processed document: {path}")
    return document


def load_all_documents(ticker: str, form_type: Optional[str] = None) -> list[dict]:
    """
    Loads all processed documents for a ticker, optionally filtered by form type.

    This is what the temporal drift agent calls — it needs all 8 quarters
    of transcripts for MSFT in chronological order to detect language drift.

    Args:
        ticker:    Stock ticker — e.g. "AMZN".
        form_type: Optional filter — "10-K", "10-Q", "8-K", or "4".
                   If None, returns all document types for this ticker.

    Returns:
        List of document dicts sorted by filing_date ascending —
        oldest first so temporal analysis works naturally.
    """
    base_path = Path(settings.processed_data_path) / ticker

    if not base_path.exists():
        logger.warning(f"No processed documents found for {ticker} at {base_path}")
        return []

    documents = []

    # Walk the ticker's directory tree — either all form type subdirs
    # or just the one we want.
    if form_type:
        search_dirs = [base_path / form_type]
    else:
        # All subdirectories — one per form type
        search_dirs = [d for d in base_path.iterdir() if d.is_dir()]

    for directory in search_dirs:
        if not directory.exists():
            continue

        for json_file in directory.glob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    doc = json.load(f)
                documents.append(doc)

            except json.JSONDecodeError as e:
                # A corrupt JSON file should not crash the whole load —
                # log it and move on. This can happen if a previous run
                # was interrupted mid-write.
                logger.error(f"Corrupt JSON file, skipping: {json_file} — {e}")

    # Sort chronologically — agents always want oldest-first for temporal analysis
    documents.sort(key=lambda d: d["filing_date"])

    logger.info(
        f"Loaded {len(documents)} documents for {ticker}"
        + (f" [{form_type}]" if form_type else " [all types]")
    )

    return documents


def load_all_tickers(form_type: Optional[str] = None) -> dict[str, list[dict]]:
    """
    Loads processed documents for every ticker in our coverage universe.

    The supply chain agent needs this — it reasons across all companies
    simultaneously to propagate upstream signals to downstream tickers.

    Args:
        form_type: Optional filter — if None, loads all document types.

    Returns:
        Dict mapping ticker -> list of documents, all sorted chronologically.
    """
    all_documents = {}

    for ticker in settings.ticker_cik_map.keys():
        docs = load_all_documents(ticker, form_type)

        # Only include tickers that actually have documents —
        # cleaner for agents to work with than empty lists.
        if docs:
            all_documents[ticker] = docs

    total = sum(len(docs) for docs in all_documents.values())
    logger.info(f"Loaded {total} total documents across {len(all_documents)} tickers")

    return all_documents


# ── Inspection utilities ───────────────────────────────────────────────────────

def document_exists(
    ticker: str,
    form_type: str,
    filing_date: str,
    accession_number: str,
) -> bool:
    """
    Checks if a processed document already exists on disk.

    Used by run_ingestion.py to skip re-downloading and re-parsing files
    that were already successfully processed in a previous run.
    Idempotent ingestion — safe to run multiple times without duplicating work.

    Args:
        ticker:           Stock ticker.
        form_type:        SEC form type.
        filing_date:      Filing date string.
        accession_number: EDGAR accession number.

    Returns:
        True if the processed JSON already exists, False otherwise.
    """
    path = _processed_path(ticker, form_type, filing_date, accession_number)
    return path.exists()


def get_ingestion_summary() -> dict:
    """
    Returns a count of how many documents have been ingested per ticker and form type.

    Useful for a quick sanity check before kicking off an agent run —
    confirms the corpus is complete before we start reasoning over it.

    Returns:
        Nested dict: {ticker: {form_type: count}}
    """
    summary = {}
    base_path = Path(settings.processed_data_path)

    if not base_path.exists():
        return summary

    for ticker_dir in sorted(base_path.iterdir()):
        if not ticker_dir.is_dir():
            continue

        ticker  = ticker_dir.name
        summary[ticker] = {}

        for form_dir in sorted(ticker_dir.iterdir()):
            if not form_dir.is_dir():
                continue

            form_type = form_dir.name
            count     = len(list(form_dir.glob("*.json")))

            if count > 0:
                summary[ticker][form_type] = count

    return summary