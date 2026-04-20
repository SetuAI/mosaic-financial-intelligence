"""
MOSAIC — Multi-Agent Orchestration System for Analyst Intelligence and Cognition
agents/insider_signal_agent.py

Cross-references insider transaction filings (Form 4) against earnings call
sentiment to detect meaningful divergence between what executives say publicly
and what they do privately with their own shares.

The core insight this agent operationalises:
    Words are cheap. Stock transactions are not.

    When a CEO stands on an earnings call and says "we have never been more
    confident in our business trajectory" and then sells $40 million of stock
    three weeks later, that divergence is meaningful. It does not necessarily
    mean fraud — executives sell stock for many legitimate reasons (diversification,
    tax planning, pre-scheduled 10b5-1 plans). But the pattern, the timing,
    and the scale of the transaction relative to their stated confidence level
    is worth a human analyst reviewing.

    Conversely, when executives buy stock in the open market during a period
    where they are publicly cautious, that is a bullish signal — they are
    putting their own money behind a more optimistic private view.

    The most powerful signal is cluster selling — when multiple executives
    sell simultaneously around the same period. A single executive sale can
    be a 10b5-1 plan. Five executives selling in the same two-week window
    is statistically unusual and warrants attention.

Key limitations to be honest about:
    - Many insider sales are pre-scheduled under 10b5-1 plans, which are
      legitimate and do not reflect current views. We flag this as a
      mitigating factor but cannot always confirm plan existence from Form 4.
    - This agent analyses patterns, not intent. All signals are routed to
      HITL because insider trading implications require human judgment.
    - Form 4 text in our corpus is structured data, not prose — the agent
      searches for transaction patterns, not narrative language.

Run standalone? No — called by graph_builder.py
"""

import json

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langsmith import traceable

from graph.state import MosaicState, build_signal
from tools.vector_search_tools import search_filings
from tools.yfinance_tools import get_price_on_date, get_earnings_history
from tools.edgar_tools import get_insider_transactions
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

SYSTEM_PROMPT = """You are a senior financial investigator specialising in insider 
trading pattern analysis. You have deep expertise in SEC Form 4 filings and the 
relationship between executive transactions and public communications.

Your job is to identify meaningful divergence between:
1. What executives say publicly on earnings calls (their stated confidence)
2. What they do privately with their own company shares (their revealed preference)

SIGNAL PATTERNS TO IDENTIFY:

HIGH SIGNIFICANCE:
- Large sales (>$10M) within 30 days of a bullish earnings call
- Cluster selling: 3+ executives selling within the same 2-week window
- CEO or CFO selling >20% of their holdings after expressing strong confidence
- Open market purchases during a period of public caution (bullish contrarian signal)

MEDIUM SIGNIFICANCE:
- Repeated pattern of selling before guidance reductions over multiple quarters
- Sales timed consistently with the peaks of post-earnings stock rallies
- New executive selling immediately after joining (may indicate concern)

LOW SIGNIFICANCE / MITIGATING FACTORS:
- Sales explicitly under 10b5-1 plan (pre-scheduled, not discretionary)
- Small sales relative to total holdings (<5%)
- Diversification sales consistent with long-term pattern
- Exercise of options near expiration (obligatory, not discretionary)

IMPORTANT CAVEATS TO ALWAYS NOTE:
- You cannot confirm 10b5-1 plan status from Form 4 alone
- Correlation between call sentiment and transactions is not evidence of intent
- All findings require human review before any action

OUTPUT FORMAT:
{
    "divergence_detected": true or false,
    "signal_type": "sell_after_bullish" | "buy_during_cautious" | "cluster_selling" | 
                   "pattern_selling" | "no_divergence",
    "severity": "low" | "medium" | "high" | "critical",
    "confidence_score": 0.0 to 1.0,
    "headline": "one sentence describing the specific divergence pattern",
    "detail": "2-3 paragraphs: what was said on the call, what transactions occurred, why the pattern is notable",
    "call_sentiment": "bullish | cautious | neutral",
    "call_sentiment_evidence": "specific quote from earnings call supporting the sentiment assessment",
    "transactions_summary": "summary of insider transactions found",
    "mitigating_factors": ["list of factors that could explain the divergence innocently"],
    "requires_hitl": true
}

IMPORTANT: Always set requires_hitl to true for this signal type. Insider trading
implications are serious enough that human review is mandatory regardless of 
confidence level."""


# ── Agent node ─────────────────────────────────────────────────────────────────

@traceable(name="insider_signal_agent")
def insider_signal_agent(state: MosaicState) -> dict:
    """
    LangGraph node — analyses insider transaction patterns for each ticker.

    Args:
        state: Current MosaicState — reads tickers from here.

    Returns:
        Partial state update with signals, retrieved_chunks, insider_signal_output.
    """
    tickers = state.get("tickers", [])

    logger.info(f"Insider signal agent starting — tickers: {tickers}")

    # Guard — ensure vector store is loaded
    from processing.vector_store import vector_store
    if not vector_store.is_ready:
        logger.info("Vector store not ready — loading now")
        vector_store.load()

    all_signals  = []
    all_chunks   = []
    agent_output = {}

    for ticker in tickers:
        logger.info(f"Analysing insider signals for {ticker}")

        result = _analyse_ticker(ticker)

        if result:
            signal, chunks, analysis = result
            all_signals.append(signal)
            all_chunks.extend(chunks)
            agent_output[ticker] = analysis

    logger.info(
        f"Insider signal agent complete — "
        f"{len(all_signals)} signals generated across {len(tickers)} tickers"
    )

    return {
        "signals":              all_signals,
        "retrieved_chunks":     all_chunks,
        "insider_signal_output": agent_output,
        "current_step":         "insider_signal_complete",
    }


# ── Per-ticker analysis ────────────────────────────────────────────────────────

def _analyse_ticker(ticker: str) -> tuple[dict, list, dict] | None:
    """
    Analyses insider transaction patterns vs earnings call sentiment for one ticker.

    Combines three data sources:
    1. Earnings call text — what management said publicly
    2. Form 4 chunks — insider transaction records from vector store
    3. Form 4 filing metadata — transaction dates and filing counts from EDGAR

    Args:
        ticker: Stock ticker to analyse.

    Returns:
        Tuple of (signal, chunks, raw_analysis) if divergence detected.
        None if no meaningful divergence found.
    """
    all_chunks = []

    # ── Source 1: Earnings call sentiment ─────────────────────────────────
    # Find the most recent earnings call content — we want to know what
    # management said MOST RECENTLY because we compare against recent transactions
    call_chunks = search_filings.invoke({
        "query":     f"{ticker} confident outlook growth strong performance guidance",
        "ticker":    ticker,
        "form_type": "8-K",
        "k":         4,
    })

    # Also search for cautious language — if management was cautious AND
    # insiders were buying, that is a bullish contrarian signal
    cautious_chunks = search_filings.invoke({
        "query":     f"{ticker} challenges headwinds uncertainty cautious risk",
        "ticker":    ticker,
        "form_type": "8-K",
        "k":         3,
    })

    all_chunks.extend(call_chunks)
    all_chunks.extend(cautious_chunks)

    # ── Source 2: Insider transactions from vector store ──────────────────
    # Form 4 chunks in our vector store contain structured transaction data —
    # executive name, transaction type (buy/sell), shares, price, date
    form4_chunks = search_filings.invoke({
        "query":     f"{ticker} shares sold purchased transaction insider",
        "ticker":    ticker,
        "form_type": "4",
        "k":         8,
    })

    all_chunks.extend(form4_chunks)

    # ── Source 3: Form 4 metadata from EDGAR ──────────────────────────────
    # Get the list of recent Form 4 filings — gives us dates and volume
    # of insider activity without needing to read each filing
    insider_filings = get_insider_transactions.invoke({"ticker": ticker})

    if not call_chunks and not form4_chunks:
        logger.warning(
            f"Insufficient data for {ticker} insider analysis — "
            f"no call chunks and no Form 4 chunks"
        )
        return None

    if not form4_chunks:
        logger.info(f"{ticker}: no Form 4 chunks in vector store — skipping")
        return None

    logger.info(
        f"{ticker}: {len(call_chunks)} call chunks, "
        f"{len(form4_chunks)} Form 4 chunks, "
        f"{len(insider_filings)} Form 4 filings"
    )

    # ── Build the analysis ─────────────────────────────────────────────────
    call_context     = _format_call_context(call_chunks, cautious_chunks)
    insider_context  = _format_insider_context(form4_chunks, insider_filings)

    analysis = _call_llm(ticker, call_context, insider_context)

    if analysis is None:
        return None

    if not analysis.get("divergence_detected", False):
        logger.info(f"{ticker}: no meaningful insider divergence detected")
        return None

    # Insider signals always go to HITL — override confidence threshold.
    # The stakes of a wrong insider trading signal are too high to auto-publish.
    analysis["requires_hitl"] = True

    signal = build_signal(
        signal_type      = "insider_alignment",
        ticker           = ticker,
        generated_by     = "insider_signal_agent",
        headline         = analysis.get("headline", "Insider divergence detected"),
        detail           = analysis.get("detail", ""),
        # Force low confidence to always trigger HITL — even a high-confidence
        # insider signal needs human review given the legal implications.
        confidence_score = min(float(analysis.get("confidence_score", 0.5)), 0.70),
        severity         = analysis.get("severity", "medium"),
        evidence         = [
            {
                "call_sentiment":          analysis.get("call_sentiment", ""),
                "call_evidence":           analysis.get("call_sentiment_evidence", "")[:200],
                "transactions_summary":    analysis.get("transactions_summary", ""),
                "mitigating_factors":      analysis.get("mitigating_factors", []),
                "form4_filing_count":      len(insider_filings),
            }
        ],
    )

    # Force HITL regardless of what build_signal computed —
    # insider signals are always reviewed by a human
    signal["requires_hitl"]  = True
    signal["hitl_reason"]    = (
        signal.get("hitl_reason", "") +
        "; insider transaction signals always require human review"
    ).strip("; ")
    signal["review_status"]  = "pending"

    logger.info(
        f"{ticker} insider signal: {analysis.get('signal_type')} "
        f"(severity={analysis.get('severity')}, HITL=True)"
    )

    return signal, all_chunks, analysis


def _format_call_context(
    bullish_chunks: list[dict],
    cautious_chunks: list[dict],
) -> str:
    """
    Formats earnings call excerpts into a context block for the LLM.

    Separates bullish and cautious language so the LLM can assess
    the overall tone of management's public communications.

    Args:
        bullish_chunks:  Call excerpts with confident/positive language.
        cautious_chunks: Call excerpts with hedged/cautious language.

    Returns:
        Formatted string for the LLM prompt.
    """
    sections = []

    if bullish_chunks:
        bullish_text = "\n\n".join(
            f"[{c['filing_date']} | {c['section_name']}]\n{c['text'][:400]}"
            for c in bullish_chunks[:3]
        )
        sections.append(f"CONFIDENT/BULLISH LANGUAGE FROM EARNINGS CALLS:\n{bullish_text}")

    if cautious_chunks:
        cautious_text = "\n\n".join(
            f"[{c['filing_date']} | {c['section_name']}]\n{c['text'][:400]}"
            for c in cautious_chunks[:2]
        )
        sections.append(f"CAUTIOUS/HEDGED LANGUAGE FROM EARNINGS CALLS:\n{cautious_text}")

    return "\n\n".join(sections) if sections else "No earnings call data available."


def _format_insider_context(
    form4_chunks: list[dict],
    insider_filings: list[dict],
) -> str:
    """
    Formats insider transaction data into a context block for the LLM.

    Combines vector store chunk content (transaction details) with
    filing metadata (dates, volume) for a complete picture.

    Args:
        form4_chunks:    Form 4 chunks from the vector store.
        insider_filings: Form 4 filing metadata from EDGAR API.

    Returns:
        Formatted string for the LLM prompt.
    """
    sections = []

    if form4_chunks:
        transaction_text = "\n\n".join(
            f"[Form 4 Filed: {c['filing_date']}]\n{c['text'][:400]}"
            for c in form4_chunks[:6]
        )
        sections.append(f"INSIDER TRANSACTIONS (Form 4 content):\n{transaction_text}")

    if insider_filings and not isinstance(insider_filings[0], dict) or \
       (insider_filings and "error" not in insider_filings[0]):
        # Show filing dates and count — gives LLM a sense of transaction volume
        filing_dates = [
            f"  Form 4 filed: {f['filing_date']}"
            for f in insider_filings[:10]
            if isinstance(f, dict) and "filing_date" in f
        ]
        if filing_dates:
            filing_summary = "\n".join(filing_dates)
            sections.append(
                f"FORM 4 FILING DATES (recent insider activity):\n{filing_summary}"
            )

    return "\n\n".join(sections) if sections else "No insider transaction data available."


def _call_llm(
    ticker: str,
    call_context: str,
    insider_context: str,
) -> dict | None:
    """
    Asks GPT-4o to assess divergence between call sentiment and insider transactions.

    Args:
        ticker:          Company being analysed.
        call_context:    Formatted earnings call excerpts.
        insider_context: Formatted insider transaction data.

    Returns:
        Parsed analysis dict, or None if call failed.
    """
    human_prompt = f"""Analyse insider transaction patterns vs earnings call sentiment for {ticker}.

EARNINGS CALL COMMUNICATIONS (what management said publicly):
{call_context}

INSIDER TRANSACTIONS (what executives did privately with their shares):
{insider_context}

Assess whether there is a meaningful divergence between the public communications
and private transactions. Consider timing, scale, and pattern across executives.

Remember: correlation is not evidence of wrongdoing. Note all mitigating factors.

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
        logger.error(f"JSON parse failed for {ticker} insider signal: {e}")
        return None

    except Exception as e:
        logger.error(f"LLM call failed for {ticker} insider signal: {e}")
        return None