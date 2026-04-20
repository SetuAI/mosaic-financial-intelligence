"""
MOSAIC — Multi-Agent Orchestration System for Analyst Intelligence and Cognition
graph/state.py

Defines the shared state schema for the LangGraph agent graph.

What the state is:
    In LangGraph, every node (agent or tool) in the graph reads from and
    writes to a single shared state object. Think of it as a whiteboard
    that every agent in the room can see and write on. When the Temporal
    Drift Agent finds a signal, it writes it to the state. When the
    Supervisor reads it and decides to escalate, it updates the state.
    When the HITL gate fires, it reads the state to know what to review.

    The state is the single source of truth for the entire pipeline run.

Why TypedDict and not a plain dict:
    TypedDict gives us type hints without runtime overhead. Every agent
    knows exactly what fields exist, what type they are, and what they
    mean. This prevents the classic multi-agent bug where Agent A writes
    "signals" as a list and Agent B expects "signal" as a dict.

Why Annotated with operator.add:
    LangGraph nodes run and return partial state updates. For list fields
    like retrieved_chunks and signals, we want each agent to ADD to the
    list rather than replace it. Annotated[list, operator.add] tells
    LangGraph to merge lists rather than overwrite them — critical when
    multiple agents are contributing signals to the same run.

Run standalone? No — imported by all agents and graph_builder.py
"""

import operator

from typing import Annotated, Any, Optional
from typing_extensions import TypedDict


class SignalDict(TypedDict, total=False):
    """
    Structure of a single intelligence signal produced by an agent.

    Every signal an agent generates must conform to this schema.
    The Supervisor reads these to compose the final brief.
    The HITL gate reads confidence_score to decide routing.

    Fields marked Optional may not be present for all signal types —
    for example, contradiction signals have two source documents while
    temporal drift signals have many. total=False makes all fields optional
    at the TypedDict level, but agents should populate as many as possible.
    """

    # ── Identity ───────────────────────────────────────────────────────────
    signal_id:      str    # unique ID — "{ticker}_{signal_type}_{timestamp}"
    signal_type:    str    # "temporal_drift" | "contradiction" | "credibility"
                           # | "supply_chain" | "insider_alignment"
    ticker:         str    # primary company this signal is about
    generated_by:   str    # which agent produced this — e.g. "temporal_drift_agent"
    generated_at:   str    # ISO timestamp

    # ── Content ────────────────────────────────────────────────────────────
    headline:       str    # one-sentence summary — what the signal says
    detail:         str    # full explanation with evidence and reasoning
    evidence:       list   # list of source chunk dicts that support the signal

    # ── Scoring ────────────────────────────────────────────────────────────
    confidence_score: float  # 0.0 to 1.0 — agent's self-assessed confidence
                             # signals below settings.hitl_confidence_threshold
                             # are routed to HITL review automatically

    severity:       str    # "low" | "medium" | "high" | "critical"
                           # critical signals always go to HITL regardless of confidence

    # ── Routing ────────────────────────────────────────────────────────────
    requires_hitl:  bool   # True if this signal must be human-reviewed
    hitl_reason:    str    # why HITL was triggered — logged in audit trail

    # ── Review outcome (populated by HITL gate) ────────────────────────────
    review_status:  str    # "pending" | "approved" | "rejected" | "edited"
    reviewer_notes: str    # analyst comments from HITL review


class MosaicState(TypedDict, total=False):
    """
    The complete shared state for one MOSAIC pipeline run.

    A pipeline run begins when a new filing is detected or when an analyst
    requests analysis of a specific ticker. The state flows through the graph:

        Supervisor → routes to agents → agents write signals → Supervisor
        reads signals → HITL gate → final brief

    Fields use Annotated[list, operator.add] where multiple agents contribute
    to the same list — LangGraph merges rather than overwrites these fields.
    Plain fields (str, dict, bool) are overwritten by the last writer.
    """

    # ── Run identity ───────────────────────────────────────────────────────

    run_id:         str    # unique ID for this pipeline run
    run_started_at: str    # ISO timestamp when the run began
    triggered_by:   str    # "new_filing" | "scheduled" | "analyst_request"

    # ── Target ─────────────────────────────────────────────────────────────

    # Primary ticker this run is focused on — may be None for cross-company runs
    primary_ticker: Optional[str]

    # All tickers involved in this run — always populated
    # For a single-company run: ["MSFT"]
    # For a supply chain run: ["TSMC", "AAPL", "NVDA", "AMD"]
    tickers:        list[str]

    # The analyst's question or the event that triggered this run
    # e.g. "Analyse MSFT Q2 2025 earnings call for risk signals"
    query:          str

    # ── Agent routing ──────────────────────────────────────────────────────

    # Which agents the Supervisor has decided to activate for this run.
    # Supervisor sets this in its first pass based on the query and trigger.
    active_agents:  list[str]

    # Current step in the graph — used by the Supervisor to track progress
    # and decide what to do next.
    current_step:   str

    # ── Retrieved context ──────────────────────────────────────────────────

    # All chunks retrieved from the vector store during this run.
    # Annotated with operator.add so each agent's retrieved chunks
    # accumulate rather than overwriting each other.
    retrieved_chunks: Annotated[list[dict], operator.add]

    # Financial data fetched from EDGAR and yFinance during this run —
    # stored so agents do not re-fetch the same data independently.
    financial_data: dict[str, Any]

    # ── Signals ────────────────────────────────────────────────────────────

    # All signals produced by all agents in this run.
    # operator.add means each agent appends its signals to the shared list
    # rather than replacing what other agents wrote.
    signals: Annotated[list[SignalDict], operator.add]

    # Signals that have been approved after HITL review —
    # only these make it into the final brief.
    approved_signals: Annotated[list[SignalDict], operator.add]

    # Signals that were rejected in HITL review —
    # kept for audit trail and agent learning.
    rejected_signals: Annotated[list[SignalDict], operator.add]

    # ── HITL ───────────────────────────────────────────────────────────────

    # True if any signal in this run requires human review
    hitl_required:  bool

    # Signals currently waiting for human review
    hitl_queue:     list[SignalDict]

    # True once all HITL reviews in this run are complete
    hitl_complete:  bool

    # ── Agent outputs ──────────────────────────────────────────────────────

    # Each agent writes its structured output here under its own key.
    # This lets the Supervisor read individual agent results without
    # scanning through the full signals list.
    temporal_drift_output:  Optional[dict]
    contradiction_output:   Optional[dict]
    credibility_output:     Optional[dict]
    supply_chain_output:    Optional[dict]
    insider_signal_output:  Optional[dict]

    # ── Final output ───────────────────────────────────────────────────────

    # The Supervisor's composed intelligence brief — written last,
    # after all agents have run and all HITL reviews are complete.
    final_brief:    Optional[str]

    # Structured summary of the run — signal counts, confidence scores,
    # which agents ran, how long it took.
    run_summary:    Optional[dict]

    # True once the run is fully complete and the brief is ready
    run_complete:   bool


# ── State initialiser ──────────────────────────────────────────────────────────

def create_initial_state(
    query:          str,
    tickers:        list[str],
    triggered_by:   str = "analyst_request",
    primary_ticker: Optional[str] = None,
) -> MosaicState:
    """
    Creates a clean initial state for a new pipeline run.

    Always use this function to start a run — never construct MosaicState
    manually. This ensures all list fields are initialised to empty lists
    rather than None, which would break operator.add merging.

    Args:
        query:          The analyst's question or analysis request.
        tickers:        Companies to include in this run.
        triggered_by:   What initiated this run.
        primary_ticker: Optional focus company — useful for single-company analysis.

    Returns:
        MosaicState dict ready to pass into the LangGraph graph.
    """
    import uuid
    from datetime import datetime, timezone

    run_id = str(uuid.uuid4())[:8]   # short ID — readable in logs

    return MosaicState(
        # ── Run identity ──
        run_id         = run_id,
        run_started_at = datetime.now(timezone.utc).isoformat(),
        triggered_by   = triggered_by,

        # ── Target ──
        primary_ticker = primary_ticker or (tickers[0] if tickers else None),
        tickers        = tickers,
        query          = query,

        # ── Routing ──
        active_agents  = [],
        current_step   = "initialised",

        # ── Context — initialise all list fields to empty lists ──
        # Never None — operator.add breaks on None + list.
        retrieved_chunks = [],
        financial_data   = {},

        # ── Signals — all start empty ──
        signals          = [],
        approved_signals = [],
        rejected_signals = [],

        # ── HITL ──
        hitl_required  = False,
        hitl_queue     = [],
        hitl_complete  = False,

        # ── Agent outputs — None until each agent runs ──
        temporal_drift_output = None,
        contradiction_output  = None,
        credibility_output    = None,
        supply_chain_output   = None,
        insider_signal_output = None,

        # ── Final output ──
        final_brief  = None,
        run_summary  = None,
        run_complete = False,
    )


# ── Signal builder ─────────────────────────────────────────────────────────────

def build_signal(
    signal_type:      str,
    ticker:           str,
    generated_by:     str,
    headline:         str,
    detail:           str,
    confidence_score: float,
    severity:         str,
    evidence:         list[dict],
) -> SignalDict:
    """
    Constructs a properly formed SignalDict.

    Every agent uses this function to build its signals — ensures consistent
    structure and handles ID generation and HITL routing logic automatically.

    HITL is triggered automatically when:
        - confidence_score is below settings.hitl_confidence_threshold (0.75)
        - severity is "critical" — regardless of confidence

    Args:
        signal_type:      Type identifier for this signal.
        ticker:           Primary company this signal concerns.
        generated_by:     Name of the agent that produced this signal.
        headline:         One-sentence summary of what the signal found.
        detail:           Full explanation with reasoning and evidence citations.
        confidence_score: Agent's self-assessed confidence — 0.0 to 1.0.
        severity:         "low" | "medium" | "high" | "critical".
        evidence:         List of source chunk dicts supporting this signal.

    Returns:
        Fully populated SignalDict ready to append to state["signals"].
    """
    import uuid
    from datetime import datetime, timezone
    from config.settings import settings

    signal_id = f"{ticker}_{signal_type}_{str(uuid.uuid4())[:6]}"

    # Determine if this signal needs human review.
    # Critical signals always go to HITL — a wrong critical signal
    # published without review could damage the system's credibility.
    requires_hitl = (
        confidence_score < settings.hitl_confidence_threshold
        or severity == "critical"
    )

    hitl_reason = ""
    if requires_hitl:
        reasons = []
        if confidence_score < settings.hitl_confidence_threshold:
            reasons.append(
                f"confidence {confidence_score:.2f} below threshold "
                f"{settings.hitl_confidence_threshold}"
            )
        if severity == "critical":
            reasons.append("severity is critical")
        hitl_reason = "; ".join(reasons)

    return SignalDict(
        signal_id        = signal_id,
        signal_type      = signal_type,
        ticker           = ticker,
        generated_by     = generated_by,
        generated_at     = datetime.now(timezone.utc).isoformat(),
        headline         = headline,
        detail           = detail,
        evidence         = evidence,
        confidence_score = confidence_score,
        severity         = severity,
        requires_hitl    = requires_hitl,
        hitl_reason      = hitl_reason,
        review_status    = "pending" if requires_hitl else "auto_approved",
        reviewer_notes   = "",
    )