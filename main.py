"""
MOSAIC — Multi-Agent Orchestration System for Analyst Intelligence and Cognition
main.py

The local entry point for running MOSAIC from the project root.

This is what you run during development and demos — it gives you a clean
command-line interface to every major MOSAIC operation without having to
remember individual module paths and arguments.

In production (AWS), the Lambda handlers replace this file.
Locally, this is your control panel.

Usage:
    python main.py pipeline --tickers MSFT NVDA
    python main.py pipeline --tickers MSFT --query "Analyse for supply chain risk"
    python main.py ingest --ticker MSFT --form-type 10-K
    python main.py process
    python main.py eval --no-langsmith
    python main.py hitl --file data/hitl_queue_latest.json
    python main.py status
"""

import argparse
import json
import os
import sys

from pathlib import Path

from config.logging_config import setup_logging, get_logger
from config.settings import settings

setup_logging()
logger = get_logger(__name__)


# ── Commands ───────────────────────────────────────────────────────────────────

def cmd_pipeline(args: argparse.Namespace) -> None:
    """
    Runs the full MOSAIC agent pipeline and prints the intelligence brief.

    This is the main command — loads the vector store, runs all five agents,
    routes signals through HITL, and prints the final brief to the terminal.
    Also saves the HITL queue to data/hitl_queue_latest.json for review.

    Args:
        args: Parsed CLI arguments — tickers, query, no_langsmith flag.
    """
    if args.no_langsmith:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
    else:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"]    = settings.langchain_api_key
        os.environ["LANGCHAIN_PROJECT"]    = "mosaic-prod"

    from graph.graph_builder import run_mosaic

    tickers        = args.tickers or list(settings.ticker_cik_map.keys())
    query          = args.query or "Analyse all tickers for risk signals across all dimensions"
    primary_ticker = args.primary or (tickers[0] if tickers else None)

    print("\n" + "=" * 60)
    print("MOSAIC Pipeline Run")
    print("=" * 60)
    print(f"Tickers : {tickers}")
    print(f"Query   : {query}")
    print("=" * 60 + "\n")

    result = run_mosaic(
        query          = query,
        tickers        = tickers,
        primary_ticker = primary_ticker,
    )

    # ── Print statistics ───────────────────────────────────────────────────
    summary = result.get("run_summary", {})

    print("\n" + "=" * 60)
    print("RUN STATISTICS")
    print("=" * 60)
    print(f"Run ID          : {result.get('run_id')}")
    print(f"Total signals   : {len(result.get('signals', []))}")
    print(f"Auto approved   : {summary.get('auto_approved', 0)}")
    print(f"HITL queue      : {len(result.get('hitl_queue', []))}")
    print(f"Chunks retrieved: {summary.get('chunks_retrieved', 0)}")
    print(f"Overall risk    : {summary.get('overall_risk', 'unknown').upper()}")

    # ── Print signals ──────────────────────────────────────────────────────
    signals = result.get("signals", [])
    if signals:
        print("\n" + "=" * 60)
        print("SIGNALS GENERATED")
        print("=" * 60)
        for i, s in enumerate(signals, 1):
            marker = "⚠ HITL" if s.get("requires_hitl") else "✓ AUTO"
            print(
                f"\n[{i}] {marker} | {s.get('ticker')} | "
                f"{s.get('signal_type')} | "
                f"severity={s.get('severity')} | "
                f"confidence={s.get('confidence_score')}"
            )
            print(f"    {s.get('headline', '')}")

    # ── Print HITL queue ───────────────────────────────────────────────────
    hitl_queue = result.get("hitl_queue", [])
    if hitl_queue:
        print("\n" + "=" * 60)
        print(f"HITL REVIEW QUEUE ({len(hitl_queue)} items)")
        print("=" * 60)
        for item in hitl_queue:
            print(
                f"\n  [{item.get('urgency', '?').upper()}] "
                f"{item.get('ticker')} — {item.get('signal_type')}"
            )
            print(f"  {item.get('headline', '')}")

    # ── Print final brief ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FINAL INTELLIGENCE BRIEF")
    print("=" * 60)
    print(result.get("final_brief", "Brief not generated."))

    # ── Save HITL queue for review ─────────────────────────────────────────
    if hitl_queue:
        Path("data").mkdir(exist_ok=True)
        with open("data/hitl_queue_latest.json", "w") as f:
            json.dump({
                "hitl_queue": hitl_queue,
                "signals":    signals,
                "run_id":     result.get("run_id"),
            }, f, indent=2)
        print(
            f"\nHITL queue saved → data/hitl_queue_latest.json"
            f"\nReview with: python main.py hitl"
        )


def cmd_ingest(args: argparse.Namespace) -> None:
    """
    Runs the EDGAR ingestion pipeline to download new filings.

    Args:
        args: Parsed CLI arguments — ticker, form_type, log_level.
    """
    from ingestion.run_ingestion import run_pipeline

    print("\n" + "=" * 60)
    print("MOSAIC Ingestion Pipeline")
    print("=" * 60)

    tickers    = args.ticker or None
    form_types = [args.form_type] if args.form_type else None

    run_pipeline(tickers=tickers, form_types=form_types)


def cmd_process(args: argparse.Namespace) -> None:
    """
    Runs the processing pipeline — chunks, embeds, and indexes documents.

    Args:
        args: Parsed CLI arguments — ticker, form_type.
    """
    from processing.run_processing import run_pipeline

    print("\n" + "=" * 60)
    print("MOSAIC Processing Pipeline")
    print("=" * 60)

    tickers    = args.ticker or None
    form_types = [args.form_type] if args.form_type else None

    run_pipeline(tickers=tickers, form_types=form_types)


def cmd_eval(args: argparse.Namespace) -> None:
    """
    Runs the evaluation suite against all agents.

    Args:
        args: Parsed CLI arguments — no_langsmith, case.
    """
    from evaluation.eval_runner import run_evals, EVAL_TEST_CASES

    use_langsmith = not args.no_langsmith

    cases = EVAL_TEST_CASES
    if args.case:
        cases = [c for c in EVAL_TEST_CASES if c["name"] == args.case]
        if not cases:
            print(f"Test case '{args.case}' not found.")
            print(f"Available: {[c['name'] for c in EVAL_TEST_CASES]}")
            sys.exit(1)

    run_evals(test_cases=cases, use_langsmith=use_langsmith)


def cmd_hitl(args: argparse.Namespace) -> None:
    """
    Opens the interactive HITL review session.

    Args:
        args: Parsed CLI arguments — file path, reviewer name.
    """
    from graph.hitl import run_interactive_review, load_audit_log

    if args.audit:
        entries = load_audit_log()
        print(f"\nAudit log — {len(entries)} entries:")
        for e in entries:
            print(
                f"  {e['timestamp'][:19]} | {e['decision']:10s} | "
                f"{e['ticker']:6s} | {e['signal_type']:20s} | "
                f"{e.get('reviewer', '?')}"
            )
        return

    hitl_file = args.file or "data/hitl_queue_latest.json"

    if not Path(hitl_file).exists():
        print(f"HITL file not found: {hitl_file}")
        print("Run the pipeline first: python main.py pipeline --tickers MSFT")
        sys.exit(1)

    with open(hitl_file) as f:
        data = json.load(f)

    hitl_queue  = data.get("hitl_queue", [])
    all_signals = data.get("signals", [])
    reviewer    = args.reviewer or "analyst"

    if not hitl_queue:
        print("No signals in HITL queue — nothing to review.")
        return

    approved, rejected = run_interactive_review(
        hitl_queue  = hitl_queue,
        all_signals = all_signals,
        reviewer    = reviewer,
    )

    print(f"\nReview complete — approved: {len(approved)}, rejected: {len(rejected)}")


def cmd_status(args: argparse.Namespace) -> None:
    """
    Prints a status summary of the local MOSAIC installation.
    Checks what data is on disk, whether the vector store is built,
    and whether all API keys are configured.
    """
    print("\n" + "=" * 60)
    print("MOSAIC Status")
    print("=" * 60)

    # ── API Keys ───────────────────────────────────────────────────────────
    print("\nAPI Keys:")
    print(f"  OpenAI      : {'✓ configured' if settings.openai_api_key else '✗ missing'}")
    print(f"  LangSmith   : {'✓ configured' if settings.langchain_api_key else '✗ missing'}")
    print(f"  SEC Agent   : {'✓ configured' if settings.sec_user_agent else '✗ missing'}")

    # ── Documents on disk ──────────────────────────────────────────────────
    print("\nDocuments on disk:")
    from ingestion.document_store import get_ingestion_summary
    summary = get_ingestion_summary()

    if summary:
        total = 0
        for ticker, forms in summary.items():
            for form_type, count in forms.items():
                print(f"  {ticker:6s} {form_type:6s} — {count} documents")
                total += count
        print(f"  Total: {total} documents")
    else:
        print("  No documents found — run: python main.py ingest")

    # ── Vector store ───────────────────────────────────────────────────────
    print("\nVector store:")
    vector_index = Path("data/vector_store/faiss.index")
    metadata     = Path("data/vector_store/chunk_metadata.json")

    if vector_index.exists() and metadata.exists():
        size_mb = vector_index.stat().st_size / (1024 * 1024)
        print(f"  ✓ Built — index size: {size_mb:.1f}MB")

        # Load and check vector count without loading everything
        stats_path = Path("data/vector_store/stats.json")
        if stats_path.exists():
            with open(stats_path) as f:
                stats = json.load(f)
            print(f"  Total vectors: {stats.get('total_vectors', '?')}")
    else:
        print("  ✗ Not built — run: python main.py process")

    # ── HITL audit log ─────────────────────────────────────────────────────
    print("\nHITL Audit Log:")
    audit_path = Path("data/hitl_audit_log.jsonl")
    if audit_path.exists():
        with open(audit_path) as f:
            entries = f.readlines()
        print(f"  {len(entries)} review decisions logged")
    else:
        print("  No reviews yet")

    print()


# ── CLI parser ─────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    """
    Builds the main argument parser with subcommands for each operation.
    """
    parser = argparse.ArgumentParser(
        prog        = "python main.py",
        description = "MOSAIC — Multi-Agent Financial Intelligence System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  pipeline   Run the full agent pipeline and generate an intelligence brief
  ingest     Download new SEC filings from EDGAR
  process    Chunk, embed, and index documents into the vector store
  eval       Run the evaluation suite against all agents
  hitl       Review signals in the HITL queue interactively
  status     Show what data is on disk and whether everything is configured

Examples:
  python main.py pipeline --tickers MSFT NVDA
  python main.py pipeline --tickers MSFT --query "Supply chain risk analysis"
  python main.py ingest --ticker MSFT --form-type 10-K
  python main.py process --ticker MSFT
  python main.py eval --no-langsmith
  python main.py eval --case msft_contradiction
  python main.py hitl --reviewer "Chirantan"
  python main.py hitl --audit
  python main.py status
        """,
    )

    subparsers = parser.add_subparsers(dest="command")

    # ── pipeline ───────────────────────────────────────────────────────────
    p_pipeline = subparsers.add_parser("pipeline", help="Run the full agent pipeline")
    p_pipeline.add_argument(
        "--tickers", nargs="+",
        help="Tickers to analyse. Default: all 6 configured tickers.",
        metavar="TICKER",
    )
    p_pipeline.add_argument(
        "--query", type=str,
        help="Analysis query. Default: full risk analysis.",
    )
    p_pipeline.add_argument(
        "--primary", type=str,
        help="Primary focus ticker.",
        metavar="TICKER",
    )
    p_pipeline.add_argument(
        "--no-langsmith", action="store_true",
        help="Disable LangSmith tracing for this run.",
    )

    # ── ingest ─────────────────────────────────────────────────────────────
    p_ingest = subparsers.add_parser("ingest", help="Download SEC filings from EDGAR")
    p_ingest.add_argument(
        "--ticker", nargs="+",
        help="Tickers to ingest. Default: all configured tickers.",
        metavar="TICKER",
    )
    p_ingest.add_argument(
        "--form-type", type=str,
        help="Form type to ingest: 10-K, 10-Q, 8-K, or 4.",
        metavar="FORM",
    )

    # ── process ────────────────────────────────────────────────────────────
    p_process = subparsers.add_parser("process", help="Chunk, embed, and index documents")
    p_process.add_argument(
        "--ticker", nargs="+",
        help="Tickers to process. Default: all configured tickers.",
        metavar="TICKER",
    )
    p_process.add_argument(
        "--form-type", type=str,
        help="Form type to process.",
        metavar="FORM",
    )

    # ── eval ───────────────────────────────────────────────────────────────
    p_eval = subparsers.add_parser("eval", help="Run the evaluation suite")
    p_eval.add_argument(
        "--no-langsmith", action="store_true",
        help="Run locally without LangSmith tracing.",
    )
    p_eval.add_argument(
        "--case", type=str,
        help="Run a single named test case.",
        metavar="CASE",
    )

    # ── hitl ───────────────────────────────────────────────────────────────
    p_hitl = subparsers.add_parser("hitl", help="Review HITL signals interactively")
    p_hitl.add_argument(
        "--file", type=str,
        help="Path to HITL queue JSON file. Default: data/hitl_queue_latest.json",
    )
    p_hitl.add_argument(
        "--reviewer", type=str,
        help="Reviewer name for audit log. Default: analyst",
    )
    p_hitl.add_argument(
        "--audit", action="store_true",
        help="Print the audit log and exit.",
    )

    # ── status ─────────────────────────────────────────────────────────────
    subparsers.add_parser("status", help="Show system status")

    return parser


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = build_parser()
    args   = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    command_map = {
        "pipeline": cmd_pipeline,
        "ingest":   cmd_ingest,
        "process":  cmd_process,
        "eval":     cmd_eval,
        "hitl":     cmd_hitl,
        "status":   cmd_status,
    }

    command_map[args.command](args)