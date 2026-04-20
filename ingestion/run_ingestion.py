"""
MOSAIC — Multi-Agent Orchestration System for Analyst Intelligence and Cognition
ingestion/run_ingestion.py

The entry point for the ingestion pipeline.

Wires together edgar_client → document_parser → document_store into a
single runnable script. Run this file to pull all filings for all tickers
and get clean, structured JSON documents on disk ready for the agents.

Usage:
    python -m ingestion.run_ingestion                    # all tickers, all form types
    python -m ingestion.run_ingestion --ticker MSFT      # single ticker
    python -m ingestion.run_ingestion --form-type 10-K   # single form type

Design decisions:
    - Processes one ticker at a time, not in parallel. EDGAR's rate limit
      makes parallelism more trouble than it is worth at our scale.
    - Skips documents that already exist on disk — safe to re-run after
      a partial failure without re-downloading everything.
    - Logs a summary at the end so you know exactly what was ingested
      and what was skipped.
"""

import argparse
import sys

from typing import Optional

from config.settings import settings
from config.logging_config import setup_logging, get_logger
from ingestion.edgar_client import (
    get_recent_filings,
    get_filing_index,
    get_document_text,
    find_exhibit,
)
from ingestion.document_parser import parse_filing
from ingestion.document_store import (
    save_raw_filing,
    save_processed_document,
    document_exists,
    get_ingestion_summary,
)

logger = get_logger(__name__)


# ── Per-filing ingestion ───────────────────────────────────────────────────────

def ingest_filing(ticker: str, filing: dict) -> bool:
    """
    Ingests a single filing end-to-end — fetch, parse, save.

    Returns True if the filing was successfully processed,
    False if it was skipped or failed. The caller uses this
    to track progress without crashing on individual failures.

    Skipping logic:
        If the processed JSON already exists on disk, we skip the entire
        fetch-parse-save cycle. This makes the pipeline idempotent —
        you can kill it halfway and re-run without duplicating work.

    Args:
        ticker:  Stock ticker — e.g. "MSFT".
        filing:  Filing metadata dict from edgar_client.get_recent_filings().

    Returns:
        True if processed successfully or already existed, False on error.
    """
    form_type        = filing["form"]
    filing_date      = filing["filingDate"]
    accession_number = filing["accessionNumber"]
    primary_document = filing.get("primaryDocument", "")

    # Skip if we already have this document — idempotent ingestion.
    # This is the single most important feature for a long-running pipeline —
    # a crash at filing 47 of 100 should not force you to restart from zero.
    if document_exists(ticker, form_type, filing_date, accession_number):
        logger.info(f"  Skipping {ticker} {form_type} {filing_date} — already on disk")
        return True

    logger.info(f"  Processing {ticker} {form_type} {filing_date} ({accession_number})")

    try:
        # ── Step 1: Determine which document to fetch ──────────────────────
        filename = _resolve_document_filename(
            ticker, filing, form_type, primary_document
        )

        if not filename:
            logger.warning(
                f"  Could not resolve document filename for "
                f"{ticker} {form_type} {filing_date} — skipping"
            )
            return False

        # ── Step 2: Download the raw HTML ──────────────────────────────────
        raw_html = get_document_text(ticker, accession_number, filename)

        if not raw_html or len(raw_html) < 1000:
            logger.warning(
                f"  Downloaded content is too short ({len(raw_html)} chars) "
                f"for {ticker} {form_type} {filing_date} — skipping"
            )
            return False

        # Save the raw HTML immediately — before parsing.
        # If the parser crashes, we can re-parse from this file without
        # hitting EDGAR again.
        save_raw_filing(raw_html, ticker, form_type, filing_date, accession_number)

        # ── Step 3: Parse the raw HTML into structured text ────────────────
        document = parse_filing(
            raw_html         = raw_html,
            ticker           = ticker,
            form_type        = form_type,
            filing_date      = filing_date,
            accession_number = accession_number,
        )

        # ── Step 4: Save the structured document to disk ───────────────────
        save_processed_document(document)

        logger.info(
            f"  ✓ {ticker} {form_type} {filing_date} — "
            f"{document['char_count']:,} chars, "
            f"{document['section_count']} sections"
        )
        return True

    except Exception as e:
        # Catching broadly here is intentional — a single filing failure
        # should not crash the entire pipeline. We log it and move on.
        # The summary at the end will show what was missed.
        logger.error(
            f"  ✗ Failed to process {ticker} {form_type} {filing_date}: {e}",
            exc_info=True,
        )
        return False


def _resolve_document_filename(
    ticker: str,
    filing: dict,
    form_type: str,
    primary_document: str,
) -> Optional[str]:
    """
    Determines the correct filename to download for a given filing.

    This is more involved than it sounds because different form types
    and different companies store their content in different exhibits:

    - 10-K, 10-Q: primaryDocument is correct — it IS the filing.
    - 8-K: content is in an exhibit, usually EX-99.1. We need to fetch
      the filing index to find the right exhibit filename.
    - Form 4: primaryDocument is correct.

    AAPL edge case: they occasionally put prepared remarks in EX-99.1
    and Q&A in EX-99.2. We grab EX-99.1 and note the limitation —
    a future improvement would merge both exhibits.

    Args:
        ticker:           Stock ticker.
        filing:           Filing metadata dict.
        form_type:        SEC form type.
        primary_document: Filename from the filing metadata.

    Returns:
        The filename to download, or None if it cannot be determined.
    """
    if form_type in ("10-K", "10-Q", "4"):
        # For annual/quarterly reports and insider filings, the primary
        # document in the metadata is exactly what we want.
        return primary_document

    if form_type == "8-K":
        # 8-K filings bundle the transcript as an exhibit — we need to
        # look at the filing index to find the EX-99.1 filename.
        # The primaryDocument for an 8-K is the cover page, not the content.
        try:
            filing_index = get_filing_index(ticker, filing["accessionNumber"])
            filename     = find_exhibit(filing_index, "EX-99.1")

            if filename:
                return filename

            # Some companies label their exhibit differently —
            # try the primary document as a fallback.
            logger.debug(
                f"No EX-99.1 found for {ticker} 8-K {filing['filingDate']} "
                f"— falling back to primaryDocument"
            )
            return primary_document

        except Exception as e:
            logger.warning(
                f"Could not fetch filing index for {ticker} 8-K "
                f"{filing['filingDate']}: {e} — falling back to primaryDocument"
            )
            return primary_document

    # Unknown form type — should not happen given our FILING_TYPES config
    # but handle it gracefully rather than crashing.
    logger.warning(f"Unrecognised form type '{form_type}' for {ticker} — skipping")
    return None


# ── Per-ticker ingestion ───────────────────────────────────────────────────────

def ingest_ticker(ticker: str, form_types: list[str]) -> dict:
    """
    Ingests all filings of the specified types for a single ticker.

    Processes one form type at a time within the ticker — all 10-Ks first,
    then all 10-Qs, then all 8-Ks. Keeps the log output easy to follow.

    Args:
        ticker:     Stock ticker — e.g. "GOOG".
        form_types: List of form types to ingest — e.g. ["10-K", "10-Q", "8-K"].

    Returns:
        Dict with counts: {form_type: {"processed": N, "skipped": N, "failed": N}}
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Starting ingestion for {ticker}")
    logger.info(f"{'='*60}")

    results = {}

    for form_type in form_types:
        logger.info(f"\nFetching {form_type} filings for {ticker}...")

        try:
            filings = get_recent_filings(ticker, form_type)
        except Exception as e:
            logger.error(f"Failed to fetch {form_type} filing list for {ticker}: {e}")
            results[form_type] = {"processed": 0, "skipped": 0, "failed": 1}
            continue

        if not filings:
            logger.info(f"  No {form_type} filings found for {ticker} in lookback window")
            results[form_type] = {"processed": 0, "skipped": 0, "failed": 0}
            continue

        logger.info(f"  Found {len(filings)} {form_type} filings to process")

        processed = 0
        skipped   = 0
        failed    = 0

        for filing in filings:
            # Check upfront if this is already done — avoid unnecessary log noise
            if document_exists(
                ticker,
                form_type,
                filing["filingDate"],
                filing["accessionNumber"],
            ):
                skipped += 1
                logger.info(
                    f"  Skipping {ticker} {form_type} {filing['filingDate']} "
                    f"— already on disk"
                )
                continue

            success = ingest_filing(ticker, filing)

            if success:
                processed += 1
            else:
                failed += 1

        results[form_type] = {
            "processed": processed,
            "skipped":   skipped,
            "failed":    failed,
        }

        logger.info(
            f"  {form_type} complete — "
            f"processed: {processed}, skipped: {skipped}, failed: {failed}"
        )

    return results


# ── Full pipeline ──────────────────────────────────────────────────────────────

def run_pipeline(
    tickers: Optional[list[str]]    = None,
    form_types: Optional[list[str]] = None,
) -> None:
    """
    Runs the full ingestion pipeline across all tickers and form types.

    This is the top-level function — it loops over every ticker in our
    coverage universe and ingests all filing types we care about.

    Args:
        tickers:    List of tickers to process. Defaults to all tickers
                    in settings.ticker_cik_map if not specified.
        form_types: List of form types to fetch. Defaults to
                    settings.filing_types if not specified.
    """
    # Default to everything configured in settings
    tickers    = tickers    or list(settings.ticker_cik_map.keys())
    form_types = form_types or settings.filing_types

    logger.info("=" * 60)
    logger.info("MOSAIC Ingestion Pipeline Starting")
    logger.info(f"Tickers:    {tickers}")
    logger.info(f"Form types: {form_types}")
    logger.info(f"Lookback:   {settings.lookback_years} years")
    logger.info("=" * 60)

    all_results = {}

    for ticker in tickers:
        ticker_results       = ingest_ticker(ticker, form_types)
        all_results[ticker]  = ticker_results

    # ── Final summary ──────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("MOSAIC Ingestion Pipeline Complete")
    logger.info("=" * 60)

    total_processed = 0
    total_skipped   = 0
    total_failed    = 0

    for ticker, results in all_results.items():
        for form_type, counts in results.items():
            total_processed += counts["processed"]
            total_skipped   += counts["skipped"]
            total_failed    += counts["failed"]

    logger.info(f"Total processed : {total_processed}")
    logger.info(f"Total skipped   : {total_skipped} (already on disk)")
    logger.info(f"Total failed    : {total_failed}")

    # Show what is now on disk — the ground truth of what agents can use
    logger.info("\nIngestion summary (documents on disk):")
    summary = get_ingestion_summary()

    for ticker, form_counts in summary.items():
        for form_type, count in form_counts.items():
            logger.info(f"  {ticker:6s} {form_type:6s} — {count} documents")

    if total_failed > 0:
        logger.warning(
            f"\n{total_failed} filings failed — check logs above for details. "
            f"Re-run the pipeline to retry failed filings."
        )


# ── CLI interface ──────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments for flexible pipeline runs.

    Running without arguments processes everything. Passing --ticker or
    --form-type restricts the run — useful during development when you
    just want to test one ticker without waiting for all six.
    """
    parser = argparse.ArgumentParser(
        description="MOSAIC ingestion pipeline — fetches and parses SEC EDGAR filings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m ingestion.run_ingestion
  python -m ingestion.run_ingestion --ticker MSFT
  python -m ingestion.run_ingestion --ticker MSFT GOOG
  python -m ingestion.run_ingestion --form-type 10-K
  python -m ingestion.run_ingestion --ticker AMZN --form-type 8-K
        """,
    )

    parser.add_argument(
        "--ticker",
        nargs="+",
        help="One or more tickers to process. Defaults to all configured tickers.",
        metavar="TICKER",
    )

    parser.add_argument(
        "--form-type",
        nargs="+",
        dest="form_types",
        help="One or more form types to fetch. Defaults to all configured form types.",
        metavar="FORM_TYPE",
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity. Use DEBUG to see every EDGAR request.",
    )

    return parser.parse_args()


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = _parse_args()

    # Initialise logging before anything else runs —
    # nothing in the pipeline logs correctly until this is called.
    setup_logging(level=args.log_level)
 
    run_pipeline(
        tickers    = args.ticker,
        form_types = args.form_types,
    )
    
    
'''
to execute this:
Start with just MSFT and 10-K to validate everything works end to end before pulling all 6 tickers:
python -m ingestion.run_ingestion --ticker MSFT --form-type 10-K --log-level INFO


Output stored in data/raw/MSFT and processed too 
EDGAR confirmed 2 annual reports exist for Microsoft in our 2-year window — FY2024 and FY2025.
✓ MSFT 10-K 2025-07-30 — 313,430 chars, 3 sections
✓ MSFT 10-K 2024-07-30 — 354,495 chars, 3 sections
Both downloaded, parsed, and saved to disk successfully. 313K and 354K characters of clean readable text — that is roughly 250–280 pages of prose each, stripped of all HTML noise.
Total processed : 2
Total skipped   : 0
Total failed    : 0

data/
├── raw/
│   └── MSFT/
│       └── 10-K/
│           ├── 2025-07-30_0000950170_25_100235.html   ← 34MB raw HTML from EDGAR
│           └── 2024-07-30_0000950170_24_087843.html   ← raw HTML backup
└── processed/
    └── MSFT/
        └── 10-K/
            ├── 2025-07-30_0000950170_25_100235.json   ← clean structured JSON
            └── 2024-07-30_0000950170_24_087843.json   ← clean structured JSON
Open one of those JSON files in VS Code — you will see the full structured document with sections, 
metadata, and clean text ready for agents.

command to run entire ingestion 
python -m ingestion.run_ingestion
That is it. No flags — it defaults to all 6 tickers and all 4 form types configured in settings.py.
This will pull:

6 tickers — AMZN, MSFT, GOOG, ORCL, AAPL, NVDA
4 form types — 8-K, 10-K, 10-Q, Form 4
Last 2 years per ticker per form type

Expect it to take 8–12 minutes — EDGAR's 0.5s delay between requests is the bottleneck, not your machine or network.
Do not kill it mid-run. If it does get interrupted, just re-run the same command — it will skip everything already on disk and pick up where it left off.

'''