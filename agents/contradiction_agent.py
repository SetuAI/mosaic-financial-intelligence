"""
MOSAIC — Multi-Agent Orchestration System for Analyst Intelligence and Cognition
agents/contradiction_agent.py

Detects meaningful contradictions between what management says on earnings
calls and what they file legally with the SEC in their 10-K annual reports.

The core insight this agent operationalises:
    Earnings calls are marketing. CEOs and CFOs are trained communicators
    who know how to sound confident and reassuring even when things are not
    going well. The earnings call is unaudited, carries no legal liability
    for optimism, and is designed to maintain investor confidence.

    The 10-K is legal. Every word in a 10-K has been reviewed by lawyers,
    auditors, and the board. When a company adds a new risk factor, expands
    a legal proceedings section, or changes the language around a business
    segment, they are doing so because they legally must — not because they
    want to.

    The gap between these two documents is where the real signal lives.

Run standalone? No — called by graph_builder.py
"""

import json

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langsmith import traceable

from graph.state import MosaicState, build_signal
from tools.vector_search_tools import search_filings
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

SYSTEM_PROMPT = """You are a senior financial analyst and forensic accountant 
specialising in identifying discrepancies between public company communications 
and their legal SEC filings.

Your job is to compare what management said on earnings calls against what they 
filed in their 10-K annual report for the same period. You are looking for 
meaningful contradictions — cases where the tone, content, or specific claims 
in one document materially diverge from the other.

WHAT CONSTITUTES A MEANINGFUL CONTRADICTION:

1. Topic presence asymmetry — a significant risk, litigation, or business 
   development mentioned in the 10-K that was completely absent from the 
   earnings call, or vice versa.

2. Tone divergence — management expresses strong confidence on the call 
   while the 10-K uses significantly more cautious language about the same topic.

3. Factual inconsistency — specific numbers, timelines, or claims that differ 
   between the two documents for the same metric or event.

4. Risk escalation without disclosure — new or expanded risk factors in the 
   10-K that were not acknowledged or disclosed on the earnings call.

5. Guidance vs filing gap — forward guidance given on the call that is 
   materially inconsistent with the trajectory shown in the financial statements.

WHAT IS NOT A CONTRADICTION:
- Different levels of detail (10-K is always more detailed than a call)
- Standard legal boilerplate in 10-K that management does not read aloud
- Slightly different phrasing of the same underlying message
- Appropriate optimism on the call vs appropriate conservatism in legal filings

SEVERITY ASSESSMENT:
- critical: factual inconsistency or active omission of material information
- high: significant tone divergence on a material business topic
- medium: topic presence asymmetry on a non-material issue
- low: minor framing differences with no substantive impact

OUTPUT FORMAT:
Respond with a JSON object:
{
    "contradiction_detected": true or false,
    "severity": "low" | "medium" | "high" | "critical",
    "confidence_score": 0.0 to 1.0,
    "headline": "one sentence describing the specific contradiction found",
    "detail": "2-3 paragraphs. First: what the earnings call said. Second: what the 10-K said. Third: why this gap is meaningful.",
    "call_excerpt": "the specific quote or passage from the earnings call",
    "filing_excerpt": "the specific passage from the 10-K that contradicts it",
    "contradiction_type": "topic_asymmetry" | "tone_divergence" | "factual_inconsistency" | "risk_omission" | "guidance_gap",
    "topics_compared": ["list of topics examined"]
}

Be precise and evidence-based. Every contradiction must be supported by 
specific text from both documents."""


# ── Topics to compare ──────────────────────────────────────────────────────────

COMPARISON_TOPICS = [
    {
        "topic":        "revenue_guidance",
        "call_query":   "revenue growth guidance outlook next quarter year",
        "filing_query": "revenue growth risk factors forward looking",
    },
    {
        "topic":        "litigation_legal",
        "call_query":   "litigation legal proceedings regulatory investigation",
        "filing_query": "legal proceedings litigation regulatory risk",
    },
    {
        "topic":        "competitive_position",
        "call_query":   "competitive advantage market position differentiation",
        "filing_query": "competition competitive risk market share",
    },
    {
        "topic":        "operational_risk",
        "call_query":   "operational challenges supply chain headwinds",
        "filing_query": "operational risk supply chain disruption",
    },
    {
        "topic":        "ai_investment",
        "call_query":   "artificial intelligence AI investment infrastructure",
        "filing_query": "artificial intelligence capital expenditure investment risk",
    },
]


# ── Agent node ─────────────────────────────────────────────────────────────────

@traceable(name="contradiction_agent")
def contradiction_agent(state: MosaicState) -> dict:
    """
    LangGraph node — compares earnings call content against 10-K filings
    for each ticker in state and identifies meaningful contradictions.

    Args:
        state: Current MosaicState — reads tickers from here.

    Returns:
        Partial state update with signals, retrieved_chunks, contradiction_output.
    """
    tickers = state.get("tickers", [])

    logger.info(f"Contradiction agent starting — tickers: {tickers}")

    all_signals  = []
    all_chunks   = []
    agent_output = {}

    for ticker in tickers:
        logger.info(f"Checking contradictions for {ticker}")

        result = _analyse_ticker(ticker)

        if result:
            signal, chunks, analysis = result
            all_signals.append(signal)
            all_chunks.extend(chunks)
            agent_output[ticker] = analysis

    logger.info(
        f"Contradiction agent complete — "
        f"{len(all_signals)} contradictions found across {len(tickers)} tickers"
    )

    return {
        "signals":              all_signals,
        "retrieved_chunks":     all_chunks,
        "contradiction_output": agent_output,
        "current_step":         "contradiction_complete",
    }


# ── Per-ticker analysis ────────────────────────────────────────────────────────

# Guard — ensure vector store is loaded before any search calls.
from processing.vector_store import vector_store
if not vector_store.is_ready:
    logger.info("Vector store not ready — loading now")
    vector_store.load()

def _analyse_ticker(ticker: str) -> tuple[dict, list, dict] | None:
    """
    Runs the full contradiction analysis for one ticker across all topics.

    Args:
        ticker: Stock ticker to analyse.

    Returns:
        Tuple of (signal, chunks, raw_analysis) if contradiction found.
        None if no meaningful contradiction detected.
    """
    # Guard — ensure vector store is loaded before any search calls.
    from processing.vector_store import vector_store
    if not vector_store.is_ready:
        logger.info("Vector store not ready — loading now")
        vector_store.load()

    all_chunks     = []
    contradictions = []

    for topic_config in COMPARISON_TOPICS:
        topic        = topic_config["topic"]
        call_query   = topic_config["call_query"]
        filing_query = topic_config["filing_query"]

        call_chunks = search_filings.invoke({
            "query":     f"{ticker} {call_query}",
            "ticker":    ticker,
            "form_type": "8-K",
            "k":         3,
        })

        filing_chunks = search_filings.invoke({
            "query":     f"{ticker} {filing_query}",
            "ticker":    ticker,
            "form_type": "10-K",
            "k":         3,
        })

        if not call_chunks or not filing_chunks:
            logger.debug(
                f"{ticker} {topic}: insufficient chunks "
                f"(call={len(call_chunks)}, filing={len(filing_chunks)}) — skipping"
            )
            continue

        all_chunks.extend(call_chunks)
        all_chunks.extend(filing_chunks)

        analysis = _compare_topic(ticker, topic, call_chunks, filing_chunks)

        if analysis and analysis.get("contradiction_detected"):
            analysis["topic"] = topic
            contradictions.append(analysis)
            logger.info(
                f"{ticker} contradiction on '{topic}': "
                f"severity={analysis.get('severity')} "
                f"confidence={analysis.get('confidence_score')}"
            )

    if not contradictions:
        logger.info(
            f"{ticker}: no contradictions detected across "
            f"{len(COMPARISON_TOPICS)} topics"
        )
        return None

    # Surface the most severe contradiction as the primary signal
    severity_order    = {"critical": 4, "high": 3, "medium": 2, "low": 1}
    top_contradiction = max(
        contradictions,
        key=lambda c: severity_order.get(c.get("severity", "low"), 0)
    )

    signal = build_signal(
        signal_type      = "contradiction",
        ticker           = ticker,
        generated_by     = "contradiction_agent",
        headline         = top_contradiction.get("headline", "Contradiction detected"),
        detail           = top_contradiction.get("detail", ""),
        confidence_score = float(top_contradiction.get("confidence_score", 0.5)),
        severity         = top_contradiction.get("severity", "medium"),
        evidence         = [
            {
                "source":       "earnings_call",
                "text_excerpt": top_contradiction.get("call_excerpt", "")[:300],
            },
            {
                "source":       "10-K_filing",
                "text_excerpt": top_contradiction.get("filing_excerpt", "")[:300],
            },
        ],
    )

    agent_output = {
        "contradictions_found": len(contradictions),
        "top_contradiction":    top_contradiction,
        "all_contradictions":   contradictions,
    }

    return signal, all_chunks, agent_output


def _compare_topic(
    ticker: str,
    topic: str,
    call_chunks: list[dict],
    filing_chunks: list[dict],
) -> dict | None:
    """
    Asks GPT-4o to compare earnings call passages against 10-K passages
    for a specific topic.

    Args:
        ticker:        Company being analysed.
        topic:         The topic being compared.
        call_chunks:   Relevant passages from the earnings call (8-K).
        filing_chunks: Relevant passages from the annual report (10-K).

    Returns:
        Parsed analysis dict, or None if the LLM call failed.
    """
    call_text = "\n\n".join(
        f"[8-K {c['filing_date']} | {c['section_name']}]\n{c['text'][:500]}"
        for c in call_chunks
    )

    filing_text = "\n\n".join(
        f"[10-K {c['filing_date']} | {c['section_name']}]\n{c['text'][:500]}"
        for c in filing_chunks
    )

    human_prompt = f"""Compare the following passages from {ticker}'s earnings 
call (8-K) and annual report (10-K) regarding the topic of: {topic.replace('_', ' ')}.

EARNINGS CALL PASSAGES:
{call_text}

10-K ANNUAL REPORT PASSAGES:
{filing_text}

Identify any meaningful contradictions between what management communicated 
on the earnings call versus what they filed in the 10-K. Focus specifically 
on the topic of {topic.replace('_', ' ')}.

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
        logger.error(f"JSON parse failed for {ticker} {topic}: {e}")
        return None

    except Exception as e:
        logger.error(f"LLM call failed for {ticker} contradiction {topic}: {e}")
        return None