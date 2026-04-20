"""
MOSAIC — Multi-Agent Orchestration System for Analyst Intelligence and Cognition
graph/hitl.py

Human-in-the-loop review interface for signals requiring analyst approval.

What HITL does in MOSAIC:
    Three signals from the full pipeline run are sitting in the review queue:
    - MSFT contradiction (URGENT) — revenue guidance vs 10-K figures
    - MSFT insider alignment (ELEVATED) — cluster selling after bullish call
    - NVDA insider alignment (ELEVATED) — cluster selling after bullish call

    Before any of these signals are published to the final brief or acted upon,
    a human analyst reviews them. The analyst can:
    - APPROVE  — signal is valid, publish it as-is
    - REJECT   — signal is wrong or misleading, do not publish, log the reason
    - EDIT     — signal is partially right, analyst corrects the headline/detail
    - ESCALATE — signal needs senior review before any decision

    Every decision is logged with timestamp, reviewer identity, and reason.
    Rejected signals with reason codes feed back to improve agent prompts
    over time — this is how the system learns from human judgment.

Two modes of operation:
    1. Interactive CLI — analyst reviews signals one at a time in the terminal.
       Used during development, demos, and for small queues.
    2. Programmatic — signals are passed to review functions directly.
       Used by the graph when running in automated/API mode.

Why HITL is not just a nice-to-have:
    Financial signals carry legal and reputational implications. A wrong
    "insider trading" signal published without review could damage the
    system's credibility permanently. A wrong "critical contradiction" signal
    acted upon could lead to incorrect investment decisions. The human gate
    is not a limitation of the AI — it is a deliberate design choice that
    makes the system trustworthy enough to use in production.

Run standalone? Yes — for interactive review of HITL queue
Command: python -m graph.hitl --run-id {run_id}
"""

import json
import argparse

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from config.logging_config import get_logger, setup_logging
from config.settings import settings

logger = get_logger(__name__)


# ── Review decision constants ──────────────────────────────────────────────────

APPROVED  = "approved"
REJECTED  = "rejected"
EDITED    = "edited"
ESCALATED = "escalated"
PENDING   = "pending"

VALID_DECISIONS = {APPROVED, REJECTED, EDITED, ESCALATED}

# Audit log path — every review decision is written here permanently
AUDIT_LOG_PATH = Path("data/hitl_audit_log.jsonl")


# ── Core review functions ──────────────────────────────────────────────────────

def review_signal(
    signal: dict,
    decision: str,
    reviewer: str,
    reason: str = "",
    edited_headline: Optional[str] = None,
    edited_detail: Optional[str] = None,
) -> dict:
    """
    Records a human review decision for a single signal.

    This is the core HITL function — every review decision goes through here.
    It validates the decision, applies any edits, logs the audit trail,
    and returns the updated signal with review outcome attached.

    Args:
        signal:          The signal dict being reviewed.
        decision:        One of: "approved", "rejected", "edited", "escalated".
        reviewer:        Name or ID of the analyst making the decision.
        reason:          Why this decision was made — required for rejections,
                         strongly recommended for all decisions.
        edited_headline: New headline if decision is "edited".
        edited_detail:   New detail text if decision is "edited".

    Returns:
        Updated signal dict with review_status, reviewer_notes, and
        review_timestamp populated.

    Raises:
        ValueError: If decision is not one of the valid options.
    """
    if decision not in VALID_DECISIONS:
        raise ValueError(
            f"Invalid decision '{decision}'. "
            f"Must be one of: {VALID_DECISIONS}"
        )

    if decision == REJECTED and not reason:
        raise ValueError(
            "A reason is required when rejecting a signal. "
            "This helps agents learn from human judgment."
        )

    reviewed_signal = signal.copy()

    # Apply edits if provided
    if decision == EDITED:
        if edited_headline:
            reviewed_signal["headline"] = edited_headline
        if edited_detail:
            reviewed_signal["detail"] = edited_detail
        if not edited_headline and not edited_detail:
            logger.warning(
                "Decision is 'edited' but no edits provided — "
                "treating as approved"
            )
            decision = APPROVED

    # Record the review outcome on the signal
    reviewed_signal["review_status"]    = decision
    reviewed_signal["reviewer"]         = reviewer
    reviewed_signal["reviewer_notes"]   = reason
    reviewed_signal["review_timestamp"] = datetime.now(timezone.utc).isoformat()

    # Write to audit log — permanent record of every review decision
    _write_audit_log(reviewed_signal, decision, reviewer, reason)

    logger.info(
        f"Signal reviewed: {reviewed_signal.get('signal_id')} "
        f"decision={decision} reviewer={reviewer}"
    )

    return reviewed_signal


def process_hitl_queue(
    hitl_queue: list[dict],
    all_signals: list[dict],
    reviewer: str,
    decisions: dict[str, dict],
) -> tuple[list[dict], list[dict]]:
    """
    Processes a batch of HITL review decisions programmatically.

    Used when an external system (API, UI) submits review decisions
    rather than using the interactive CLI. Each signal in the queue
    can be reviewed independently — partial processing is supported.

    Args:
        hitl_queue:  List of HITL queue items from state["hitl_queue"].
        all_signals: Full signals list from state["signals"].
        reviewer:    Name or ID of the analyst submitting decisions.
        decisions:   Dict mapping signal_id to decision dict:
                     {
                         "signal_id": {
                             "decision": "approved" | "rejected" | "edited",
                             "reason": "...",
                             "edited_headline": "...",  # optional
                             "edited_detail": "...",    # optional
                         }
                     }

    Returns:
        Tuple of (approved_signals, rejected_signals) — both lists of
        fully reviewed signal dicts.
    """
    approved = []
    rejected = []

    # Build a lookup of signal_id → signal for quick access
    signals_by_id = {s.get("signal_id"): s for s in all_signals}

    for queue_item in hitl_queue:
        signal_id = queue_item.get("signal_id")

        if signal_id not in decisions:
            logger.warning(
                f"No decision provided for signal {signal_id} — "
                f"leaving in pending state"
            )
            continue

        signal = signals_by_id.get(signal_id)
        if not signal:
            logger.error(f"Signal {signal_id} not found in signals list")
            continue

        decision_data = decisions[signal_id]
        decision      = decision_data.get("decision", APPROVED)

        reviewed = review_signal(
            signal          = signal,
            decision        = decision,
            reviewer        = reviewer,
            reason          = decision_data.get("reason", ""),
            edited_headline = decision_data.get("edited_headline"),
            edited_detail   = decision_data.get("edited_detail"),
        )

        if decision in (APPROVED, EDITED):
            approved.append(reviewed)
        elif decision == REJECTED:
            rejected.append(reviewed)
        elif decision == ESCALATED:
            # Escalated signals go back to pending for senior review —
            # treat as neither approved nor rejected at this stage
            logger.info(f"Signal {signal_id} escalated for senior review")

    logger.info(
        f"HITL batch complete — "
        f"approved={len(approved)} rejected={len(rejected)} "
        f"reviewer={reviewer}"
    )

    return approved, rejected


# ── Interactive CLI review ─────────────────────────────────────────────────────

def run_interactive_review(
    hitl_queue: list[dict],
    all_signals: list[dict],
    reviewer: str,
) -> tuple[list[dict], list[dict]]:
    """
    Runs an interactive terminal review session for the HITL queue.

    Presents each signal one at a time, shows the full headline, detail,
    and evidence, and prompts the analyst for a decision. Decisions are
    recorded immediately — if the session is interrupted, all decisions
    made so far are preserved in the audit log.

    Args:
        hitl_queue:  HITL queue from state["hitl_queue"].
        all_signals: Full signals list from state["signals"].
        reviewer:    Analyst name for audit logging.

    Returns:
        Tuple of (approved_signals, rejected_signals).
    """
    if not hitl_queue:
        print("\nNo signals in HITL queue — nothing to review.")
        return [], []

    signals_by_id = {s.get("signal_id"): s for s in all_signals}

    approved = []
    rejected = []

    print("\n" + "=" * 60)
    print(f"MOSAIC HITL REVIEW SESSION")
    print(f"Reviewer : {reviewer}")
    print(f"Queue    : {len(hitl_queue)} signals to review")
    print("=" * 60)

    for i, queue_item in enumerate(hitl_queue, 1):
        signal_id = queue_item.get("signal_id")
        signal    = signals_by_id.get(signal_id)

        if not signal:
            logger.error(f"Signal {signal_id} not found — skipping")
            continue

        _display_signal_for_review(signal, queue_item, i, len(hitl_queue))

        # Get decision from analyst
        decision, reason, edited_headline, edited_detail = _prompt_for_decision(signal)

        # Record the review
        reviewed = review_signal(
            signal          = signal,
            decision        = decision,
            reviewer        = reviewer,
            reason          = reason,
            edited_headline = edited_headline,
            edited_detail   = edited_detail,
        )

        if decision in (APPROVED, EDITED):
            approved.append(reviewed)
            print(f"\n  ✓ Signal approved and added to final brief")
        elif decision == REJECTED:
            rejected.append(reviewed)
            print(f"\n  ✗ Signal rejected — reason logged")
        elif decision == ESCALATED:
            print(f"\n  ↑ Signal escalated for senior review")

        if i < len(hitl_queue):
            input("\n  Press Enter to continue to next signal...")

    print("\n" + "=" * 60)
    print("REVIEW SESSION COMPLETE")
    print(f"Approved : {len(approved)}")
    print(f"Rejected : {len(rejected)}")
    print(f"Audit log: {AUDIT_LOG_PATH}")
    print("=" * 60)

    return approved, rejected


def _display_signal_for_review(
    signal: dict,
    queue_item: dict,
    position: int,
    total: int,
) -> None:
    """
    Prints a formatted signal for analyst review in the terminal.

    Args:
        signal:     The signal dict to display.
        queue_item: The HITL queue item with urgency and reason.
        position:   Current position in review queue.
        total:      Total signals in queue.
    """
    urgency   = queue_item.get("urgency", "routine").upper()
    severity  = signal.get("severity", "?").upper()
    sig_type  = signal.get("signal_type", "?").replace("_", " ").upper()
    ticker    = signal.get("ticker", "?")
    confidence = signal.get("confidence_score", 0)

    print(f"\n{'='*60}")
    print(f"SIGNAL {position} of {total}  [{urgency}]")
    print(f"{'='*60}")
    print(f"Ticker     : {ticker}")
    print(f"Type       : {sig_type}")
    print(f"Severity   : {severity}")
    print(f"Confidence : {confidence}")
    print(f"Signal ID  : {signal.get('signal_id', '?')}")
    print(f"\nHEADLINE:\n  {signal.get('headline', 'No headline')}")
    print(f"\nDETAIL:\n  {signal.get('detail', 'No detail')[:600]}")

    # Show evidence if available
    evidence = signal.get("evidence", [])
    if evidence:
        print(f"\nEVIDENCE:")
        for e in evidence[:2]:
            if isinstance(e, dict):
                source  = e.get("source", e.get("form_type", "?"))
                excerpt = e.get(
                    "text_excerpt",
                    e.get("call_evidence", e.get("transactions_summary", ""))
                )
                if excerpt:
                    print(f"  [{source}]: {str(excerpt)[:200]}")

    # Show HITL reason
    hitl_reason = signal.get("hitl_reason", "")
    if hitl_reason:
        print(f"\nFLAGGED BECAUSE: {hitl_reason}")


def _prompt_for_decision(signal: dict) -> tuple[str, str, str, str]:
    """
    Prompts the analyst for a review decision in the terminal.

    Handles input validation — keeps asking until a valid decision
    is entered. All inputs are stripped and lowercased for robustness.

    Args:
        signal: The signal being reviewed — used for context in EDIT mode.

    Returns:
        Tuple of (decision, reason, edited_headline, edited_detail).
        Last two are empty strings if decision is not EDITED.
    """
    print("\nDECISION OPTIONS:")
    print("  [a] Approve  — signal is valid, add to final brief")
    print("  [r] Reject   — signal is wrong or misleading")
    print("  [e] Edit     — partially correct, I will fix the headline/detail")
    print("  [s] Escalate — needs senior review before decision")

    edited_headline = ""
    edited_detail   = ""

    while True:
        raw = input("\n  Your decision [a/r/e/s]: ").strip().lower()

        decision_map = {
            "a": APPROVED,
            "approve": APPROVED,
            "r": REJECTED,
            "reject": REJECTED,
            "e": EDITED,
            "edit": EDITED,
            "s": ESCALATED,
            "escalate": ESCALATED,
        }

        if raw not in decision_map:
            print("  Invalid input. Please enter a, r, e, or s.")
            continue

        decision = decision_map[raw]
        break

    # Get reason — required for rejections, optional for others
    if decision == REJECTED:
        while True:
            reason = input("  Reason for rejection (required): ").strip()
            if reason:
                break
            print("  A reason is required for rejection.")
    else:
        reason = input("  Notes (optional, press Enter to skip): ").strip()

    # Get edits if decision is EDITED
    if decision == EDITED:
        print(f"\n  Current headline: {signal.get('headline', '')}")
        new_headline = input("  New headline (or Enter to keep current): ").strip()
        edited_headline = new_headline if new_headline else ""

        print(f"\n  Current detail (first 200 chars): {signal.get('detail', '')[:200]}")
        new_detail = input("  New detail (or Enter to keep current): ").strip()
        edited_detail = new_detail if new_detail else ""

    return decision, reason, edited_headline, edited_detail


# ── Audit logging ──────────────────────────────────────────────────────────────

def _write_audit_log(
    signal: dict,
    decision: str,
    reviewer: str,
    reason: str,
) -> None:
    """
    Appends a review decision to the permanent audit log.

    The audit log is a JSONL file — one JSON object per line.
    This format is easy to parse, append-safe, and human-readable.

    We log every review decision permanently — even rejections.
    Rejection reasons are the feedback signal that improves agent
    prompts over time.

    Args:
        signal:   The reviewed signal dict.
        decision: The review decision made.
        reviewer: Who made the decision.
        reason:   Why the decision was made.
    """
    AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    log_entry = {
        "timestamp":     datetime.now(timezone.utc).isoformat(),
        "signal_id":     signal.get("signal_id", ""),
        "ticker":        signal.get("ticker", ""),
        "signal_type":   signal.get("signal_type", ""),
        "severity":      signal.get("severity", ""),
        "confidence":    signal.get("confidence_score", 0),
        "decision":      decision,
        "reviewer":      reviewer,
        "reason":        reason,
        "headline":      signal.get("headline", ""),
        # Store rejection reasons separately for easy filtering —
        # these are the training signal for agent improvement
        "rejection_reason": reason if decision == REJECTED else "",
    }

    with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")


# ── Audit log reader ───────────────────────────────────────────────────────────

def load_audit_log(
    ticker: Optional[str] = None,
    decision: Optional[str] = None,
    signal_type: Optional[str] = None,
) -> list[dict]:
    """
    Loads and filters the HITL audit log.

    Useful for analysing review patterns — which signal types get
    rejected most often, which tickers have the most HITL activity,
    what reasons analysts give for rejections.

    Args:
        ticker:      Optional filter by company ticker.
        decision:    Optional filter by decision type.
        signal_type: Optional filter by signal type.

    Returns:
        List of matching audit log entries.
    """
    if not AUDIT_LOG_PATH.exists():
        return []

    entries = []
    with open(AUDIT_LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                entries.append(entry)
            except json.JSONDecodeError:
                logger.warning(f"Corrupt audit log entry, skipping: {line[:50]}")

    # Apply filters
    if ticker:
        entries = [e for e in entries if e.get("ticker") == ticker]
    if decision:
        entries = [e for e in entries if e.get("decision") == decision]
    if signal_type:
        entries = [e for e in entries if e.get("signal_type") == signal_type]

    return entries


def get_rejection_patterns() -> dict:
    """
    Analyses the audit log to find common rejection patterns.

    Used to identify which agents are producing low-quality signals
    most frequently — the starting point for prompt improvement.

    Returns:
        Dict summarising rejection patterns by agent/signal type.
    """
    from collections import Counter

    entries   = load_audit_log(decision=REJECTED)
    by_type   = Counter(e.get("signal_type") for e in entries)
    by_ticker = Counter(e.get("ticker") for e in entries)
    reasons   = [e.get("rejection_reason", "") for e in entries if e.get("rejection_reason")]

    return {
        "total_rejections":    len(entries),
        "by_signal_type":      dict(by_type),
        "by_ticker":           dict(by_ticker),
        "rejection_reasons":   reasons,
        "most_rejected_type":  by_type.most_common(1)[0] if by_type else None,
    }


# ── CLI entry point ────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MOSAIC HITL review interface",
        epilog="""
Examples:
  python -m graph.hitl --file data/hitl_queue.json --reviewer "Jane Smith"
  python -m graph.hitl --audit
  python -m graph.hitl --rejections
        """,
    )

    parser.add_argument(
        "--file",
        help="Path to JSON file containing hitl_queue and signals from a pipeline run",
        metavar="PATH",
    )

    parser.add_argument(
        "--reviewer",
        default="analyst",
        help="Reviewer name for audit logging. Default: analyst",
    )

    parser.add_argument(
        "--audit",
        action="store_true",
        help="Print the full audit log and exit",
    )

    parser.add_argument(
        "--rejections",
        action="store_true",
        help="Print rejection pattern analysis and exit",
    )

    return parser.parse_args()


if __name__ == "__main__":
    setup_logging()
    args = _parse_args()

    if args.audit:
        entries = load_audit_log()
        print(f"Audit log: {len(entries)} entries")
        for e in entries:
            print(
                f"  {e['timestamp'][:19]} | {e['decision']:10s} | "
                f"{e['ticker']:6s} | {e['signal_type']:20s} | "
                f"{e['reviewer']} | {e.get('reason', '')[:60]}"
            )

    elif args.rejections:
        patterns = get_rejection_patterns()
        print(f"Total rejections : {patterns['total_rejections']}")
        print(f"By signal type   : {patterns['by_signal_type']}")
        print(f"By ticker        : {patterns['by_ticker']}")
        print(f"Most rejected    : {patterns['most_rejected_type']}")
        if patterns["rejection_reasons"]:
            print("\nRejection reasons:")
            for r in patterns["rejection_reasons"]:
                print(f"  - {r}")

    elif args.file:
        # Load pipeline output from file and run interactive review
        with open(args.file, "r") as f:
            data = json.load(f)

        hitl_queue  = data.get("hitl_queue", [])
        all_signals = data.get("signals", [])

        if not hitl_queue:
            print("No signals in HITL queue — nothing to review.")
        else:
            approved, rejected = run_interactive_review(
                hitl_queue  = hitl_queue,
                all_signals = all_signals,
                reviewer    = args.reviewer,
            )
            print(f"\nApproved: {len(approved)}, Rejected: {len(rejected)}")

    else:
        print("MOSAIC HITL Interface")
        print("Run with --help to see options")
        print("Run with --audit to see the review history")
        print("Run with --file <path> to review a pipeline output")