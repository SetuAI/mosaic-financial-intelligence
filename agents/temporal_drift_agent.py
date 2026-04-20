"""
MOSAIC — Multi-Agent Orchestration System for Analyst Intelligence and Cognition
agents/temporal_drift_agent.py

Detects how a company's language shifts across multiple consecutive filings.

The core insight this agent operationalises:
    Management teams are very careful with language. When things are going well,
    they use confident, direct language — "we are delivering strong growth",
    "demand remains robust", "we are confident in our outlook". When things
    start deteriorating, they shift to hedged, qualified language — "we believe
    results may reflect", "subject to market conditions", "we remain cautious".

    This shift rarely happens abruptly. It happens gradually over 2-3 quarters
    before the bad news becomes undeniable. By the time an analyst notices the
    tone has changed, they have already missed the early warning.

    This agent reads 8 quarters of filings chronologically and asks: has the
    language gotten materially more hedged, more cautious, or more qualified?

Why this is impossible with a single prompt:
    A single LLM call has no memory of what was said in prior quarters.
    You would have to manually assemble all 8 quarters of text, fit them
    into one context window, and hope the model notices the drift. This agent
    does that assembly automatically, uses vector search to find the most
    relevant passages per quarter, and applies a structured analytical framework
    across all of them simultaneously.

Run standalone? No — called by graph_builder.py
"""

import json

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langsmith import traceable

from graph.state import MosaicState, build_signal
from tools.vector_search_tools import search_temporal
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

SYSTEM_PROMPT = """You are a senior financial analyst specialising in qualitative 
signal detection from SEC filings. You have deep expertise in how management 
language evolves as business conditions change.

Your job is to analyse how a company's language has shifted across consecutive 
quarterly and annual filings. You are looking for meaningful changes in tone, 
confidence, and hedging — not superficial word count differences.

WHAT TO LOOK FOR:

1. Hedge escalation — increasing use of "may", "could", "subject to", 
   "we believe", "we expect", "conditions permitting". Track whether this 
   frequency is trending up across quarters.

2. Confidence degradation — language shifting from declarative ("we are 
   delivering") to conditional ("we aim to deliver", "we intend to deliver").

3. Specificity reduction — guidance becoming less specific over time. 
   "Revenue will grow 20%" becoming "revenue will grow in the mid-teens" 
   becoming "we expect continued growth".

4. Topic avoidance — subjects that were discussed openly in earlier quarters 
   disappearing from later quarters without explanation.

5. New risk language — risk factors or cautionary language appearing that was 
   absent in prior quarters.

WHAT TO IGNORE:
- Seasonal language variation (Q4 always mentions annual results)
- Standard boilerplate legal disclaimers that appear unchanged every quarter
- Minor stylistic differences between different speakers

OUTPUT FORMAT:
Respond with a JSON object containing exactly these fields:
{
    "drift_detected": true or false,
    "drift_direction": "more_cautious" | "more_confident" | "neutral",
    "severity": "low" | "medium" | "high" | "critical",
    "confidence_score": 0.0 to 1.0,
    "headline": "one sentence describing the key finding",
    "detail": "2-3 paragraphs explaining what changed, when it changed, and what specific language evidence supports the finding",
    "key_phrases": ["list", "of", "specific", "phrases", "that", "show", "the", "drift"],
    "quarters_analysed": ["list of filing dates analysed"],
    "earliest_signal_quarter": "the filing date where drift first appeared"
}

Be conservative — only flag drift_detected as true when you see a clear, 
consistent pattern across at least 2 consecutive quarters. A single quarter 
anomaly is noise, not signal."""


# ── Agent node ─────────────────────────────────────────────────────────────────

@traceable(name="temporal_drift_agent")
def temporal_drift_agent(state: MosaicState) -> dict:
    """
    LangGraph node — analyses temporal language drift for each ticker in state.

    Reads the tickers from state, fetches chronologically ordered chunks
    for each one via the vector search tool, and asks GPT-4o to identify
    meaningful language drift patterns across the filing history.

    One signal is generated per ticker where drift is detected. Tickers
    with no meaningful drift produce no signal — clean is clean.

    Args:
        state: Current MosaicState — reads tickers and query from here.

    Returns:
        Partial state update dict — LangGraph merges this into the full state.
        Contains: signals (list), retrieved_chunks (list), temporal_drift_output (dict)
    """
    tickers = state.get("tickers", [])
    query   = state.get("query", "Analyse language drift across filings")

    logger.info(f"Temporal drift agent starting — tickers: {tickers}")

    all_signals  = []
    all_chunks   = []
    agent_output = {}

    for ticker in tickers:
        logger.info(f"Analysing temporal drift for {ticker}")

        result = _analyse_ticker(ticker, query)

        if result:
            signal, chunks, analysis = result
            all_signals.append(signal)
            all_chunks.extend(chunks)
            agent_output[ticker] = analysis

    logger.info(
        f"Temporal drift agent complete — "
        f"{len(all_signals)} signals generated across {len(tickers)} tickers"
    )

    return {
        "signals":               all_signals,
        "retrieved_chunks":      all_chunks,
        "temporal_drift_output": agent_output,
        "current_step":          "temporal_drift_complete",
    }


# ── Per-ticker analysis ────────────────────────────────────────────────────────
from processing.vector_store import vector_store
if not vector_store.is_ready:
    logger.info("Vector store not ready — loading now")
    vector_store.load()
    
def _analyse_ticker(ticker: str, query: str) -> tuple[dict, list, dict] | None:
    """
    Runs the full temporal drift analysis for one ticker.

    Fetches chronologically ordered chunks, assembles them into a
    structured prompt, calls GPT-4o, parses the response, and builds
    a signal if drift is detected.

    Args:
        ticker: Stock ticker to analyse.
        query:  The analyst's original question — provides context for search.

    Returns:
        Tuple of (signal, chunks, raw_analysis) if drift detected.
        None if no meaningful drift found or insufficient data.
    """
    # Guard — ensure vector store is loaded before any search calls.
    # This protects against the case where the agent is called directly
    # in a test without going through the full pipeline startup.
    from processing.vector_store import vector_store
    if not vector_store.is_ready:
        logger.info("Vector store not ready — loading now")
        vector_store.load()

    # Search for the most relevant passages across all filings for this ticker,
    # sorted chronologically so the LLM reads them in time order.
    chunks_10k = search_temporal.invoke({
        "query":        f"{ticker} business outlook revenue growth guidance confidence",
        "ticker":       ticker,
        "form_type":    "10-K",
        "section_name": "management_discussion",
        "k":            6,
    })

    chunks_10q = search_temporal.invoke({
        "query":        f"{ticker} business outlook revenue growth guidance confidence",
        "ticker":       ticker,
        "form_type":    "10-Q",
        "section_name": "management_discussion",
        "k":            8,
    })

    all_chunks = chunks_10k + chunks_10q

    # Sort all retrieved chunks chronologically — the LLM needs to read
    # them in time order to detect drift, not in relevance order.
    all_chunks.sort(key=lambda c: c["filing_date"])

    if len(all_chunks) < 3:
        # Not enough data points to detect drift meaningfully.
        # Need at least 3 quarters to distinguish trend from noise.
        logger.warning(
            f"Insufficient chunks for {ticker} temporal analysis "
            f"({len(all_chunks)} chunks) — skipping"
        )
        return None

    logger.info(
        f"{ticker}: analysing {len(all_chunks)} chunks across "
        f"{len(set(c['filing_date'] for c in all_chunks))} filing dates"
    )

    filing_context = _format_chunks_as_timeline(all_chunks)
    analysis       = _call_llm(ticker, filing_context)

    if analysis is None:
        return None

    if not analysis.get("drift_detected", False):
        logger.info(f"{ticker}: no significant temporal drift detected")
        return None

    signal = build_signal(
        signal_type      = "temporal_drift",
        ticker           = ticker,
        generated_by     = "temporal_drift_agent",
        headline         = analysis.get("headline", "Language drift detected"),
        detail           = analysis.get("detail", ""),
        confidence_score = float(analysis.get("confidence_score", 0.5)),
        severity         = analysis.get("severity", "medium"),
        evidence         = [
            {
                "filing_date":  c["filing_date"],
                "form_type":    c["form_type"],
                "section_name": c["section_name"],
                "text_excerpt": c["text"][:300],
            }
            for c in all_chunks[:5]
        ],
    )

    logger.info(
        f"{ticker} temporal drift signal: {analysis.get('drift_direction')} "
        f"(confidence: {analysis.get('confidence_score')}, "
        f"severity: {analysis.get('severity')})"
    )

    return signal, all_chunks, analysis


def _format_chunks_as_timeline(chunks: list[dict]) -> str:
    """
    Formats retrieved chunks as a readable chronological timeline for the LLM.

    Groups chunks by filing date so the LLM sees a clear quarterly progression
    rather than a flat list of passages.

    Args:
        chunks: List of chunk dicts sorted chronologically.

    Returns:
        Formatted string ready to insert into the LLM prompt.
    """
    from collections import defaultdict

    by_date: dict[str, list] = defaultdict(list)
    for chunk in chunks:
        by_date[chunk["filing_date"]].append(chunk)

    sections = []
    for date in sorted(by_date.keys()):
        date_chunks = by_date[date]
        form_type   = date_chunks[0]["form_type"]

        text_excerpts = "\n\n".join(
            f"[{c['section_name']}]\n{c['text'][:600]}"
            for c in date_chunks[:3]
        )

        sections.append(f"--- {form_type} Filed: {date} ---\n{text_excerpts}")

    return "\n\n".join(sections)


def _call_llm(ticker: str, filing_context: str) -> dict | None:
    """
    Calls GPT-4o with the temporal drift analysis prompt.

    Args:
        ticker:          Ticker being analysed.
        filing_context:  Formatted chronological timeline of filing excerpts.

    Returns:
        Parsed analysis dict, or None if the LLM response could not be parsed.
    """
    human_prompt = f"""Analyse the following chronologically ordered excerpts from 
{ticker}'s SEC filings for temporal language drift.

{filing_context}

Based on these filings, identify whether there has been a meaningful shift in 
management's language, confidence, or hedging behaviour over time.

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
        logger.error(f"Failed to parse LLM JSON response for {ticker}: {e}")
        return None

    except Exception as e:
        logger.error(f"LLM call failed for {ticker} temporal drift: {e}")
        return None