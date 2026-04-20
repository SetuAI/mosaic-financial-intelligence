"""
MOSAIC — Multi-Agent Orchestration System for Analyst Intelligence and Cognition
agents/supervisor.py

The Supervisor agent — reads all signals from state, manages HITL routing,
and composes the final intelligence brief.

What the Supervisor does:
    After all five specialist agents have run and written their signals to state,
    the Supervisor has one job: make sense of everything and produce a coherent,
    actionable intelligence brief for the analyst.

    It does not re-analyse documents. It reads the signals that other agents
    produced, resolves any conflicts between them, decides the overall risk
    posture for each company, and writes a brief that a senior analyst could
    hand to a portfolio manager.

    It also manages HITL routing — scanning all signals and building the
    review queue for any that require human approval before publishing.

Supervisor responsibilities:
    1. Read all signals from state["signals"]
    2. Group signals by ticker and by type
    3. Identify signals that require HITL review
    4. Compose a structured final brief covering each ticker
    5. Write the brief and summary back to state

Run standalone? No — final node in graph_builder.py
"""

import json

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langsmith import traceable

from graph.state import MosaicState
from config.settings import settings
from config.logging_config import get_logger

logger = get_logger(__name__)


# ── LLM setup ─────────────────────────────────────────────────────────────────

_llm = ChatOpenAI(
    model       = settings.openai_model,
    temperature = 0.1,    # slight creativity for brief writing, still factual
    max_tokens  = settings.openai_max_tokens,
    api_key     = settings.openai_api_key,
)


# ── System prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are the Chief Investment Officer of a financial intelligence 
firm. You receive structured signals from a team of specialist analysts and your 
job is to synthesise them into a clear, actionable intelligence brief.

Your brief should read like something a senior analyst would hand to a portfolio 
manager — concise, evidence-based, and directly actionable. Not a list of AI 
outputs. A professional document.

BRIEF STRUCTURE:
1. Executive Summary (2-3 sentences): Overall risk posture across coverage universe
2. Per-company sections (one per ticker with signals):
   - Risk signals found (with severity and key evidence)
   - Positive signals (high credibility, clean supply chain)
   - Recommended analyst actions
3. HITL review items: List of signals pending human review with reasons
4. Data limitations: Honest acknowledgment of what the system could not assess

TONE AND STYLE:
- Write in professional financial analyst prose — not bullet point AI output
- Be specific — cite actual numbers, dates, and quotes where available
- Be honest about uncertainty — "the system detected X, which warrants review"
  not "X is definitely happening"
- Distinguish between confirmed findings and patterns worth monitoring
- Never overstate — a medium severity signal is not a crisis

OUTPUT FORMAT:
Respond with a JSON object:
{
    "executive_summary": "2-3 sentence overall assessment",
    "company_briefs": {
        "TICKER": {
            "overall_risk": "low | medium | high | critical",
            "key_findings": "paragraph summarising main signals",
            "positive_indicators": "what looks healthy",
            "recommended_actions": ["list", "of", "analyst", "actions"],
            "signals_count": number
        }
    },
    "hitl_items": [
        {
            "signal_id": "...",
            "ticker": "...",
            "reason_for_review": "...",
            "urgency": "routine | elevated | urgent"
        }
    ],
    "cross_company_themes": "paragraph on themes that cut across multiple companies",
    "data_limitations": "honest statement of what was not assessed",
    "overall_risk_posture": "low | medium | high | critical"
}"""


# ── Supervisor node ────────────────────────────────────────────────────────────

@traceable(name="supervisor")
def supervisor(state: MosaicState) -> dict:
    """
    Final LangGraph node — synthesises all agent signals into a brief.

    Reads the completed state after all agents have run, identifies HITL
    items, and asks GPT-4o to compose the final intelligence brief.

    Args:
        state: Fully populated MosaicState after all agents have run.

    Returns:
        Final state update with hitl_queue, final_brief, run_summary, run_complete.
    """
    signals  = state.get("signals", [])
    tickers  = state.get("tickers", [])
    query    = state.get("query", "")
    run_id   = state.get("run_id", "unknown")

    logger.info(
        f"Supervisor starting — run_id={run_id} "
        f"signals={len(signals)} tickers={tickers}"
    )

    # ── Step 1: Identify HITL items ────────────────────────────────────────
    hitl_queue = _build_hitl_queue(signals)

    hitl_required = len(hitl_queue) > 0

    logger.info(
        f"HITL assessment: {len(hitl_queue)} signals require review, "
        f"{len(signals) - len(hitl_queue)} auto-approved"
    )

    # ── Step 2: Group signals for the brief ────────────────────────────────
    signals_by_ticker = _group_signals_by_ticker(signals, tickers)

    # ── Step 3: Compose the final brief ────────────────────────────────────
    brief_analysis = _compose_brief(query, signals, signals_by_ticker, tickers)

    if brief_analysis is None:
        # If LLM fails, produce a minimal brief from the raw signals
        brief_analysis = _fallback_brief(signals, tickers)

    # ── Step 4: Build run summary ──────────────────────────────────────────
    run_summary = _build_run_summary(state, signals, hitl_queue, brief_analysis)

    # Format final brief as readable text for output
    final_brief = _format_brief_as_text(brief_analysis, run_id)

    logger.info(
        f"Supervisor complete — "
        f"overall risk: {brief_analysis.get('overall_risk_posture', 'unknown')}, "
        f"HITL items: {len(hitl_queue)}"
    )

    return {
        "hitl_required":  hitl_required,
        "hitl_queue":     hitl_queue,
        "hitl_complete":  False,   # HITL reviews happen outside the graph
        "final_brief":    final_brief,
        "run_summary":    run_summary,
        "run_complete":   True,
        "current_step":   "complete",
    }


# ── HITL queue builder ─────────────────────────────────────────────────────────

def _build_hitl_queue(signals: list[dict]) -> list[dict]:
    """
    Scans all signals and builds the HITL review queue.

    A signal goes to HITL when:
    - requires_hitl is True (set by build_signal() or forced by insider agent)
    - review_status is "pending"

    Signals are sorted by severity so reviewers see critical items first.

    Args:
        signals: All signals from state["signals"].

    Returns:
        List of HITL queue items sorted by urgency.
    """
    severity_order = {"critical": 4, "high": 3, "medium": 2, "low": 1}

    hitl_items = []

    for signal in signals:
        if not signal.get("requires_hitl", False):
            continue
        if signal.get("review_status") not in ("pending", None, ""):
            continue

        severity = signal.get("severity", "low")

        # Map severity to urgency for the review queue
        urgency_map = {
            "critical": "urgent",
            "high":     "elevated",
            "medium":   "routine",
            "low":      "routine",
        }

        hitl_items.append({
            "signal_id":         signal.get("signal_id", ""),
            "ticker":            signal.get("ticker", ""),
            "signal_type":       signal.get("signal_type", ""),
            "headline":          signal.get("headline", ""),
            "severity":          severity,
            "confidence_score":  signal.get("confidence_score", 0),
            "reason_for_review": signal.get("hitl_reason", ""),
            "urgency":           urgency_map.get(severity, "routine"),
        })

    # Sort: urgent first, then by severity, then by confidence desc
    hitl_items.sort(
        key=lambda x: (
            -{"urgent": 3, "elevated": 2, "routine": 1}.get(x["urgency"], 0),
            -severity_order.get(x["severity"], 0),
            -x.get("confidence_score", 0),
        )
    )

    return hitl_items


def _group_signals_by_ticker(
    signals: list[dict],
    tickers: list[str],
) -> dict[str, list[dict]]:
    """
    Groups signals by their primary ticker for the brief composer.

    Args:
        signals: All signals from state.
        tickers: All tickers in this run — ensures every ticker has an entry.

    Returns:
        Dict mapping ticker -> list of signals for that ticker.
    """
    by_ticker: dict[str, list] = {ticker: [] for ticker in tickers}

    for signal in signals:
        ticker = signal.get("ticker", "MULTI")
        if ticker in by_ticker:
            by_ticker[ticker].append(signal)
        else:
            # Supply chain signals may have a ticker not in the primary list
            by_ticker.setdefault(ticker, []).append(signal)

    return by_ticker


# ── Brief composition ──────────────────────────────────────────────────────────

def _compose_brief(
    query: str,
    all_signals: list[dict],
    signals_by_ticker: dict[str, list[dict]],
    tickers: list[str],
) -> dict | None:
    """
    Asks GPT-4o to compose the final intelligence brief from all signals.

    Args:
        query:              The original analyst question.
        all_signals:        All signals across all agents.
        signals_by_ticker:  Signals grouped by company.
        tickers:            All tickers in this run.

    Returns:
        Parsed brief dict, or None if LLM call failed.
    """
    # Format signals for the LLM — structured summary per signal
    signals_context = _format_signals_for_brief(all_signals, signals_by_ticker)

    human_prompt = f"""You have received intelligence signals from a multi-agent 
financial analysis system covering: {', '.join(tickers)}

Original analyst query: {query}

SIGNALS RECEIVED:
{signals_context}

Compose a professional intelligence brief synthesising these findings.
Write as a senior analyst would — not as an AI system reporting outputs.

Respond with the JSON format specified in your instructions."""

    try:
        response = _llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=human_prompt),
        ])

        raw = response.content.strip()

        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        return json.loads(raw.strip())

    except json.JSONDecodeError as e:
        logger.error(f"JSON parse failed for brief composition: {e}")
        return None

    except Exception as e:
        logger.error(f"LLM call failed for brief composition: {e}")
        return None


def _format_signals_for_brief(
    all_signals: list[dict],
    signals_by_ticker: dict[str, list[dict]],
) -> str:
    """
    Formats signals into a readable context block for the brief composer.

    Groups by ticker and formats each signal with its key information.

    Args:
        all_signals:        All signals.
        signals_by_ticker:  Signals grouped by ticker.

    Returns:
        Formatted string for the LLM prompt.
    """
    sections = []

    for ticker, signals in signals_by_ticker.items():
        if not signals:
            sections.append(f"--- {ticker} ---\nNo signals generated — all clear.")
            continue

        signal_lines = []
        for s in signals:
            signal_lines.append(
                f"  [{s.get('signal_type', '?').upper()}] "
                f"Severity={s.get('severity', '?')} "
                f"Confidence={s.get('confidence_score', '?')} "
                f"HITL={s.get('requires_hitl', False)}\n"
                f"  Headline: {s.get('headline', '')}\n"
                f"  Detail: {s.get('detail', '')[:300]}"
            )

        sections.append(
            f"--- {ticker} ({len(signals)} signals) ---\n" +
            "\n\n".join(signal_lines)
        )

    return "\n\n".join(sections)


def _fallback_brief(
    signals: list[dict],
    tickers: list[str],
) -> dict:
    """
    Produces a minimal brief when the LLM call fails.

    Ensures the pipeline always produces output even in degraded conditions.

    Args:
        signals: All signals.
        tickers: All tickers.

    Returns:
        Minimal brief dict.
    """
    by_ticker = {t: [] for t in tickers}
    for s in signals:
        t = s.get("ticker", "?")
        if t in by_ticker:
            by_ticker[t].append(s)

    company_briefs = {}
    for ticker, sigs in by_ticker.items():
        severities = [s.get("severity", "low") for s in sigs]
        overall    = max(severities, key=lambda x: {"critical": 4, "high": 3, "medium": 2, "low": 1}.get(x, 0)) if severities else "low"
        company_briefs[ticker] = {
            "overall_risk":       overall,
            "key_findings":       f"{len(sigs)} signals detected. Review individual signal details.",
            "positive_indicators": "Manual review required.",
            "recommended_actions": ["Review individual signals", "Check HITL queue"],
            "signals_count":      len(sigs),
        }

    return {
        "executive_summary":    f"Analysis complete. {len(signals)} signals detected across {len(tickers)} companies. LLM brief composition failed — review raw signals.",
        "company_briefs":       company_briefs,
        "hitl_items":           [],
        "cross_company_themes": "Manual review required.",
        "data_limitations":     "Brief composition failed. Raw signals available in state.",
        "overall_risk_posture": "medium",
    }


def _build_run_summary(
    state: MosaicState,
    signals: list[dict],
    hitl_queue: list[dict],
    brief: dict,
) -> dict:
    """
    Builds a structured run summary for logging and monitoring.

    Args:
        state:      Full pipeline state.
        signals:    All signals.
        hitl_queue: HITL review queue.
        brief:      Composed brief.

    Returns:
        Run summary dict.
    """
    from datetime import datetime, timezone
    from collections import Counter

    signal_types    = Counter(s.get("signal_type", "?") for s in signals)
    severities      = Counter(s.get("severity", "?") for s in signals)
    auto_approved   = sum(1 for s in signals if not s.get("requires_hitl", False))

    return {
        "run_id":              state.get("run_id", ""),
        "run_started_at":      state.get("run_started_at", ""),
        "run_completed_at":    datetime.now(timezone.utc).isoformat(),
        "tickers_covered":     state.get("tickers", []),
        "query":               state.get("query", ""),
        "total_signals":       len(signals),
        "signals_by_type":     dict(signal_types),
        "signals_by_severity": dict(severities),
        "auto_approved":       auto_approved,
        "hitl_required":       len(hitl_queue),
        "overall_risk":        brief.get("overall_risk_posture", "unknown"),
        "chunks_retrieved":    len(state.get("retrieved_chunks", [])),
    }


def _format_brief_as_text(brief: dict, run_id: str) -> str:
    """
    Converts the brief JSON into readable plain text for display.

    The JSON structure is used internally by the system. This text
    version is what gets shown to analysts and stored as the final output.

    Args:
        brief:  Parsed brief dict from _compose_brief().
        run_id: Run identifier for the header.

    Returns:
        Formatted plain text brief.
    """
    lines = []

    lines.append(f"MOSAIC INTELLIGENCE BRIEF — Run {run_id}")
    lines.append("=" * 60)

    # Executive summary
    lines.append("\nEXECUTIVE SUMMARY")
    lines.append(brief.get("executive_summary", "No summary available."))

    # Overall risk
    lines.append(f"\nOverall risk posture: {brief.get('overall_risk_posture', 'unknown').upper()}")

    # Per-company sections
    company_briefs = brief.get("company_briefs", {})
    if company_briefs:
        lines.append("\n" + "=" * 60)
        lines.append("COMPANY ANALYSIS")

        for ticker, cb in company_briefs.items():
            lines.append(f"\n{ticker} — Risk: {cb.get('overall_risk', '?').upper()}")
            lines.append("-" * 40)
            lines.append(cb.get("key_findings", ""))
            if cb.get("positive_indicators"):
                lines.append(f"\nPositive indicators: {cb.get('positive_indicators')}")
            actions = cb.get("recommended_actions", [])
            if actions:
                lines.append("\nRecommended actions:")
                for action in actions:
                    lines.append(f"  • {action}")

    # Cross-company themes
    themes = brief.get("cross_company_themes", "")
    if themes:
        lines.append("\n" + "=" * 60)
        lines.append("CROSS-COMPANY THEMES")
        lines.append(themes)

    # HITL items
    hitl_items = brief.get("hitl_items", [])
    if hitl_items:
        lines.append("\n" + "=" * 60)
        lines.append(f"PENDING HUMAN REVIEW ({len(hitl_items)} items)")
        for item in hitl_items:
            lines.append(
                f"\n  [{item.get('urgency', '?').upper()}] "
                f"{item.get('ticker', '?')} — {item.get('headline', item.get('reason_for_review', ''))}"
            )

    # Data limitations
    limitations = brief.get("data_limitations", "")
    if limitations:
        lines.append("\n" + "=" * 60)
        lines.append("DATA LIMITATIONS")
        lines.append(limitations)

    lines.append("\n" + "=" * 60)
    lines.append("End of MOSAIC Intelligence Brief")

    return "\n".join(lines)