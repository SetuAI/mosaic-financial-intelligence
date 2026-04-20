"""
MOSAIC — Multi-Agent Orchestration System for Analyst Intelligence and Cognition
agents/credibility_agent.py

Scores management credibility by comparing every forward guidance statement
they made against what they actually delivered in subsequent filings.

The core insight this agent operationalises:
    Every earnings call contains forward-looking statements — revenue guidance,
    margin targets, hiring plans, product timelines, capex commitments. These
    statements are the most actionable output of any earnings call — analysts
    build models on them, investors make decisions based on them.

    But management teams vary enormously in their guidance accuracy. Some CEOs
    consistently guide conservatively and overdeliver. Others habitually guide
    optimistically and underdeliver. Some are accurate on revenue but
    consistently wrong on margins.

    This agent extracts guidance statements, finds the corresponding actuals,
    and builds a credibility score per management team per metric category.

    A credibility score of 85 means management delivered on 85% of stated
    targets. A score of 45 means analysts should apply a significant haircut
    to any current guidance before building models.

Why this is impossible with a single prompt:
    Guidance is given in Q1. Results are reported in Q2, Q3, or Q4 depending
    on the guidance horizon. Matching guidance to actuals requires reading
    across multiple quarters simultaneously. No single context window can hold
    8 quarters of call transcripts and financials simultaneously.

Run standalone? No — called by graph_builder.py
"""

import json

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langsmith import traceable

from graph.state import MosaicState, build_signal
from tools.vector_search_tools import search_temporal, search_filings
from tools.edgar_tools import get_financial_facts
from tools.yfinance_tools import get_earnings_history
from config.settings import settings
from config.logging_config import get_logger

logger = get_logger(__name__)


# ── LLM setup ─────────────────────────────────────────────────────────────────

_llm = ChatOpenAI(
    model       = settings.openai_model,
    temperature = settings.openai_temperature,
    max_tokens  = settings.openai_max_tokens,
    api_key     = settings.openai_api_key,
)


# ── System prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a senior buy-side analyst specialising in management 
credibility assessment. You have extensive experience evaluating whether 
management teams deliver on their stated commitments.

Your job is to:
1. Extract every forward guidance statement from the provided earnings call excerpts
2. Compare each guidance statement against the actual reported results
3. Score management credibility based on their delivery track record

GUIDANCE EXTRACTION RULES:
Extract statements that are:
- Specific and measurable: "we expect revenue growth of 15-20%"
- Time-bound: "for the next quarter", "for fiscal year 2025", "by end of year"
- Attributable to management: said by CEO, CFO, or named executives

Do NOT extract:
- Generic aspirational statements: "we aim to be the best"
- Historical descriptions: "last quarter we achieved..."
- Standard legal disclaimers

SCORING METHODOLOGY:
For each guidance statement, classify the outcome as:
- DELIVERED: actual result met or exceeded the guidance
- MISSED: actual result fell short of guidance by more than 5%
- PARTIAL: actual result was within 5% below guidance
- UNVERIFIABLE: cannot determine actual result from available data

Overall credibility score = (DELIVERED + 0.5 * PARTIAL) / (total verifiable statements) * 100

CREDIBILITY TIERS:
- 80-100: High credibility — management consistently delivers or beats guidance
- 60-79:  Moderate credibility — some misses but generally reliable
- 40-59:  Low credibility — significant pattern of guidance misses
- 0-39:   Very low credibility — management guidance should be heavily discounted

OUTPUT FORMAT:
Respond with a JSON object:
{
    "credibility_score": 0 to 100,
    "credibility_tier": "high" | "moderate" | "low" | "very_low",
    "confidence_score": 0.0 to 1.0,
    "severity": "low" | "medium" | "high" | "critical",
    "headline": "one sentence summary of credibility finding",
    "detail": "2-3 paragraphs: what guidance was given, what was delivered, what the pattern means",
    "guidance_statements": [
        {
            "statement": "exact quote of the guidance",
            "metric": "revenue | margin | eps | capex | headcount | other",
            "period": "the time period the guidance covered",
            "guidance_value": "the specific target stated",
            "actual_value": "what was actually reported",
            "outcome": "DELIVERED | MISSED | PARTIAL | UNVERIFIABLE",
            "notes": "brief explanation"
        }
    ],
    "strongest_pattern": "what management consistently gets right or wrong",
    "recommendation": "how analysts should adjust their models based on this credibility profile"
}"""


# ── Agent node ─────────────────────────────────────────────────────────────────

@traceable(name="credibility_agent")
def credibility_agent(state: MosaicState) -> dict:
    """
    LangGraph node — scores management credibility for each ticker in state.

    Args:
        state: Current MosaicState — reads tickers from here.

    Returns:
        Partial state update with signals, retrieved_chunks, credibility_output.
    """
    tickers = state.get("tickers", [])

    logger.info(f"Credibility agent starting — tickers: {tickers}")

    all_signals  = []
    all_chunks   = []
    agent_output = {}

    for ticker in tickers:
        logger.info(f"Scoring management credibility for {ticker}")

        result = _analyse_ticker(ticker)

        if result:
            signal, chunks, analysis = result
            # signal is None when credibility is high — still capture chunks
            if signal:
                all_signals.append(signal)
            if chunks:
                all_chunks.extend(chunks)
            if analysis:
                agent_output[ticker] = analysis

    logger.info(
        f"Credibility agent complete — "
        f"{len(all_signals)} signals generated across {len(tickers)} tickers"
    )

    return {
        "signals":            all_signals,
        "retrieved_chunks":   all_chunks,
        "credibility_output": agent_output,
        "current_step":       "credibility_complete",
    }


# ── Per-ticker analysis ────────────────────────────────────────────────────────

# Guard — ensure vector store is loaded before any search calls.
from processing.vector_store import vector_store
if not vector_store.is_ready:
    logger.info("Vector store not ready — loading now")
    vector_store.load()

def _analyse_ticker(ticker: str) -> tuple[dict, list, dict] | None:
    """
    Runs the full credibility analysis for one ticker.

    Combines three data sources:
    1. Earnings call guidance excerpts — what management promised
    2. Subsequent filing actuals — what they delivered
    3. yFinance earnings history — EPS beat/miss track record

    Args:
        ticker: Stock ticker to analyse.

    Returns:
        Tuple of (signal, chunks, raw_analysis) if meaningful score produced.
        None if insufficient data.
    """
    # Guard — ensure vector store is loaded before any search calls.
    # This protects against the case where the agent is called directly
    # in a test without the full pipeline startup sequence.
    from processing.vector_store import vector_store
    if not vector_store.is_ready:
        logger.info("Vector store not ready — loading now")
        vector_store.load()

    all_chunks = []

    # ── Source 1: Forward guidance from earnings calls ─────────────────────
    guidance_chunks = search_temporal.invoke({
        "query":     f"{ticker} revenue guidance outlook expect growth next quarter fiscal year",
        "ticker":    ticker,
        "form_type": "8-K",
        "k":         8,
    })

    guidance_10q = search_filings.invoke({
    "query":     f"{ticker} outlook guidance revenue margin expect next quarter",
    "ticker":    ticker,
    "form_type": "10-Q",
    "k":         6,
    })

    all_chunks.extend(guidance_chunks)
    all_chunks.extend(guidance_10q)

    # ── Source 2: Actual reported financials from EDGAR ────────────────────
    actual_revenue = get_financial_facts.invoke({
        "ticker": ticker,
        "metric": "RevenueFromContractWithCustomerExcludingAssessedTax",
    })

    # Fall back to Revenues if the primary metric is not available
    if not actual_revenue or "error" in str(actual_revenue[0] if actual_revenue else ""):
        actual_revenue = get_financial_facts.invoke({
            "ticker": ticker,
            "metric": "Revenues",
        })

    actual_net_income = get_financial_facts.invoke({
        "ticker": ticker,
        "metric": "NetIncomeLoss",
    })

    # ── Source 3: EPS beat/miss history from yFinance ─────────────────────
    earnings_history = get_earnings_history.invoke({"ticker": ticker})

    if not guidance_chunks and not guidance_10q:
        logger.warning(f"No guidance chunks found for {ticker} — insufficient data")
        return None

    guidance_text = _format_guidance_context(guidance_chunks, guidance_10q)
    actuals_text  = _format_actuals_context(
        actual_revenue, actual_net_income, earnings_history
    )

    analysis = _call_llm(ticker, guidance_text, actuals_text)

    if analysis is None:
        return None

    credibility_score = analysis.get("credibility_score", 50)
    severity          = _score_to_severity(credibility_score)

    # Override the LLM's severity with our scoring-based severity —
    # ensures consistent mapping between score and severity tier.
    analysis["severity"] = severity

    # Only signal on moderate credibility or below —
    # high credibility is a positive data point, not a risk signal.
    if credibility_score >= 80:
        logger.info(
            f"{ticker} credibility score {credibility_score} — "
            f"high credibility, no signal generated"
        )
        # Return chunks and analysis even with no signal —
        # Supervisor uses this context in the final brief.
        return None, all_chunks, analysis

    signal = build_signal(
        signal_type      = "credibility",
        ticker           = ticker,
        generated_by     = "credibility_agent",
        headline         = analysis.get(
            "headline", f"Management credibility score: {credibility_score}"
        ),
        detail           = analysis.get("detail", ""),
        confidence_score = float(analysis.get("confidence_score", 0.6)),
        severity         = severity,
        evidence         = [
            {
                "credibility_score": credibility_score,
                "credibility_tier":  analysis.get("credibility_tier"),
                "statements_count":  len(analysis.get("guidance_statements", [])),
                "strongest_pattern": analysis.get("strongest_pattern", ""),
                "recommendation":    analysis.get("recommendation", ""),
            }
        ],
    )

    return signal, all_chunks, analysis


# ── Scoring helpers ────────────────────────────────────────────────────────────

def _score_to_severity(score: float) -> str:
    """
    Maps a credibility score to a signal severity level.

    Args:
        score: Credibility score 0-100.

    Returns:
        Severity string for build_signal().
    """
    if score >= 80:
        return "low"
    elif score >= 60:
        return "medium"
    elif score >= 40:
        return "high"
    else:
        return "critical"


def _format_guidance_context(
    call_chunks: list[dict],
    filing_chunks: list[dict],
) -> str:
    """
    Formats guidance-rich chunks into a readable context block for the LLM.

    Args:
        call_chunks:   Chunks from 8-K earnings calls.
        filing_chunks: Chunks from 10-Q management discussion sections.

    Returns:
        Formatted string ready for insertion into the LLM prompt.
    """
    sections = []

    if call_chunks:
        call_text = "\n\n".join(
            f"[Earnings Call {c['filing_date']}]\n{c['text'][:500]}"
            for c in call_chunks
        )
        sections.append(f"EARNINGS CALL GUIDANCE:\n{call_text}")

    if filing_chunks:
        filing_text = "\n\n".join(
            f"[10-Q {c['filing_date']} | MD&A]\n{c['text'][:500]}"
            for c in filing_chunks
        )
        sections.append(f"10-Q MANAGEMENT DISCUSSION GUIDANCE:\n{filing_text}")

    return "\n\n".join(sections)


def _format_actuals_context(
    revenue_data: list[dict],
    income_data: list[dict],
    earnings_history: list[dict],
) -> str:
    """
    Formats actual reported financials into a readable context block.

    Args:
        revenue_data:     EDGAR revenue actuals.
        income_data:      EDGAR net income actuals.
        earnings_history: yFinance EPS beat/miss history.

    Returns:
        Formatted string of actual results for the LLM.
    """
    sections = []

    if revenue_data and isinstance(revenue_data, list) and revenue_data:
        rev_lines = "\n".join(
            f"  {r['period_end']}: ${r['value']:,.0f} [{r['form_type']}]"
            for r in revenue_data[:6]
            if isinstance(r, dict) and r.get("value")
        )
        if rev_lines:
            sections.append(f"ACTUAL REVENUE (EDGAR reported):\n{rev_lines}")

    if income_data and isinstance(income_data, list) and income_data:
        inc_lines = "\n".join(
            f"  {r['period_end']}: ${r['value']:,.0f} [{r['form_type']}]"
            for r in income_data[:6]
            if isinstance(r, dict) and r.get("value")
        )
        if inc_lines:
            sections.append(f"ACTUAL NET INCOME (EDGAR reported):\n{inc_lines}")

    if earnings_history and isinstance(earnings_history, list):
        eps_lines = "\n".join(
            f"  {e['date']}: actual={e['eps_actual']} "
            f"estimate={e['eps_estimate']} "
            f"surprise={e['surprise_percent']}%"
            for e in earnings_history[:6]
            if isinstance(e, dict) and e.get("eps_actual") is not None
        )
        if eps_lines:
            sections.append(f"EPS BEAT/MISS HISTORY (yFinance):\n{eps_lines}")

    return "\n\n".join(sections) if sections else "No actuals data available."


def _call_llm(
    ticker: str,
    guidance_text: str,
    actuals_text: str,
) -> dict | None:
    """
    Calls GPT-4o to extract guidance statements and score credibility.

    Args:
        ticker:        Company being analysed.
        guidance_text: Formatted guidance excerpts from calls and filings.
        actuals_text:  Formatted actual results from EDGAR and yFinance.

    Returns:
        Parsed credibility analysis dict, or None if call failed.
    """
    human_prompt = f"""Analyse management credibility for {ticker}.

GUIDANCE STATEMENTS (what management said they would deliver):
{guidance_text}

ACTUAL RESULTS (what was actually reported):
{actuals_text}

Extract all forward guidance statements from the earnings call and filing 
excerpts, compare each against the actual results, and produce a credibility 
score. Focus on specific, measurable commitments — not general aspirations.

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
        logger.error(f"JSON parse failed for {ticker} credibility: {e}")
        return None

    except Exception as e:
        logger.error(f"LLM call failed for {ticker} credibility: {e}")
        return None