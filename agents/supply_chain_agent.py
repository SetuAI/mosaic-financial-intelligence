"""
MOSAIC — Multi-Agent Orchestration System for Analyst Intelligence and Cognition
agents/supply_chain_agent.py

Detects supply chain stress signals and propagates them across dependent companies.

The core insight this agent operationalises:
    No company operates in isolation. Every large tech company depends on
    a network of suppliers, contract manufacturers, and infrastructure providers.
    When a key supplier signals stress — capacity constraints, pricing pressure,
    demand softness, geopolitical risk — every downstream company that depends
    on them is exposed.

    The problem is that analysts typically cover one company deeply. They know
    NVIDIA's filings inside out but may not have read TSMC's latest earnings
    call, which mentioned a meaningful shift in advanced node demand. That TSMC
    signal is directly relevant to NVIDIA, AMD, Apple, and Qualcomm — but
    connecting those dots requires reading across many companies simultaneously.

    This agent does exactly that. It reads filings across all companies in the
    coverage universe, identifies supply chain language, and asks: when Company A
    discloses a supply chain issue, which other companies in our universe are
    exposed, and how severely?

    Examples of high-value supply chain signals:
    - NVDA mentions "supply constraints on CoWoS packaging" →
      impacts AMD, Marvell, Broadcom who use the same packaging
    - ORCL discloses "extended lead times on networking equipment" →
      impacts MSFT and GOOG who build competing cloud infrastructure
    - AMZN mentions "labour cost pressures in logistics" →
      relevant context for understanding MSFT's Azure delivery costs

Why this is impossible with a single prompt:
    Propagating a signal from Company A to Company B requires simultaneously
    knowing Company A's disclosure AND Company B's supply chain dependencies
    from their 10-K. With 6 companies that is 30 possible pairs. No single
    context window holds all of that simultaneously.

Run standalone? No — called by graph_builder.py
"""

import json

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langsmith import traceable

from graph.state import MosaicState, build_signal
from tools.vector_search_tools import search_filings, search_across_companies
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

SYSTEM_PROMPT = """You are a senior supply chain analyst with deep expertise in 
technology sector dependencies. You specialise in identifying how supply chain 
stress signals propagate across interconnected companies.

Your job is to:
1. Identify supply chain stress language in company filings
2. Assess which other companies in the coverage universe are exposed
3. Score the severity and propagation risk of the signal

SUPPLY CHAIN STRESS INDICATORS TO LOOK FOR:
- Capacity constraints or shortages: "limited availability", "supply constraints", 
  "component shortages", "extended lead times", "allocation constraints"
- Cost pressures: "input cost inflation", "pricing pressure from suppliers",
  "margin compression from supply chain"
- Geopolitical risk: "export restrictions", "trade policy uncertainty",
  "concentration risk in [region]", "single-source dependency"
- Demand signals: "inventory correction", "demand softness from customers",
  "order push-outs", "channel inventory excess"
- Infrastructure stress: "data centre capacity constraints", "power availability",
  "networking equipment lead times"

PROPAGATION ASSESSMENT:
When you identify a supply chain signal in one company's filing, assess:
1. Which other companies in the coverage universe share this dependency?
2. How material is this dependency to each exposed company?
3. Is the signal a leading indicator (will affect others soon) or lagging?

COVERAGE UNIVERSE: AMZN, MSFT, GOOG, ORCL, AAPL, NVDA

SEVERITY SCORING:
- critical: affects multiple companies, material financial impact likely
- high: affects 2+ companies, meaningful but manageable impact
- medium: affects 1-2 companies, worth monitoring
- low: minor or speculative dependency

OUTPUT FORMAT:
{
    "stress_detected": true or false,
    "stress_type": "capacity_constraint" | "cost_pressure" | "geopolitical" | 
                   "demand_signal" | "infrastructure",
    "source_company": "ticker of company where signal originated",
    "severity": "low" | "medium" | "high" | "critical",
    "confidence_score": 0.0 to 1.0,
    "headline": "one sentence describing the supply chain signal and who it affects",
    "detail": "2-3 paragraphs: what was disclosed, which companies are exposed, why it matters",
    "source_excerpt": "the specific text from the filing that signals the stress",
    "exposed_companies": [
        {
            "ticker": "exposed company ticker",
            "exposure_reason": "why this company is exposed to the source signal",
            "estimated_severity": "low | medium | high",
            "evidence": "what in their own filings confirms this dependency"
        }
    ],
    "propagation_timeline": "immediate | next_quarter | within_year | speculative"
}"""


# ── Supply chain topics to search ──────────────────────────────────────────────

# Each topic represents a category of supply chain risk we search for
# across all companies. The query is designed to surface the most relevant
# supply chain language in the filings corpus.
SUPPLY_CHAIN_TOPICS = [
    {
        "topic": "semiconductor_supply",
        "query": "semiconductor chip supply constraint shortage allocation capacity",
    },
    {
        "topic": "data_centre_infrastructure",
        "query": "data centre capacity power supply networking equipment lead time",
    },
    {
        "topic": "manufacturing_logistics",
        "query": "manufacturing capacity contract manufacturer logistics supply chain disruption",
    },
    {
        "topic": "geopolitical_trade",
        "query": "export controls trade restrictions geopolitical supply chain concentration",
    },
    {
        "topic": "cost_pressure",
        "query": "input cost supplier pricing component cost inflation margin pressure",
    },
]

# All tickers in our coverage universe — the agent searches across all of them
ALL_TICKERS = list(settings.ticker_cik_map.keys())


# ── Agent node ─────────────────────────────────────────────────────────────────

@traceable(name="supply_chain_agent")
def supply_chain_agent(state: MosaicState) -> dict:
    """
    LangGraph node — identifies supply chain stress and propagates signals
    across dependent companies in the coverage universe.

    Unlike other agents that analyse one ticker at a time, this agent
    searches across all companies simultaneously — the supply chain graph
    does not respect company boundaries.

    Args:
        state: Current MosaicState.

    Returns:
        Partial state update with signals, retrieved_chunks, supply_chain_output.
    """
    logger.info(
        f"Supply chain agent starting — "
        f"searching across {len(ALL_TICKERS)} tickers"
    )

    # Guard — ensure vector store is loaded
    from processing.vector_store import vector_store
    if not vector_store.is_ready:
        logger.info("Vector store not ready — loading now")
        vector_store.load()

    all_signals  = []
    all_chunks   = []
    agent_output = {}

    for topic_config in SUPPLY_CHAIN_TOPICS:
        topic = topic_config["topic"]
        query = topic_config["query"]

        logger.info(f"Searching supply chain topic: {topic}")

        result = _analyse_topic(topic, query)

        if result:
            signal, chunks, analysis = result
            all_signals.append(signal)
            all_chunks.extend(chunks)
            agent_output[topic] = analysis

    logger.info(
        f"Supply chain agent complete — "
        f"{len(all_signals)} signals generated across "
        f"{len(SUPPLY_CHAIN_TOPICS)} topics"
    )

    return {
        "signals":              all_signals,
        "retrieved_chunks":     all_chunks,
        "supply_chain_output":  agent_output,
        "current_step":         "supply_chain_complete",
    }


# ── Per-topic analysis ─────────────────────────────────────────────────────────

def _analyse_topic(topic: str, query: str) -> tuple[dict, list, dict] | None:
    """
    Analyses one supply chain topic across all companies simultaneously.

    Searches for the topic across all tickers, identifies the company
    with the strongest signal, then assesses which other companies
    are exposed to that signal.

    Args:
        topic: Topic identifier — e.g. "semiconductor_supply".
        query: Search query for finding relevant filing passages.

    Returns:
        Tuple of (signal, chunks, raw_analysis) if stress detected.
        None if no meaningful supply chain stress found for this topic.
    """
    # Search across all companies simultaneously for this topic.
    # search_across_companies returns results grouped by ticker.
    results_by_ticker = search_across_companies.invoke({
        "query":        query,
        "tickers":      ALL_TICKERS,
        "form_type":    "10-K",
        "k_per_ticker": 2,
    })

    # Also search 10-Q for more recent signals —
    # supply chain issues often surface in quarterly filings first
    results_10q = search_across_companies.invoke({
        "query":        query,
        "tickers":      ALL_TICKERS,
        "form_type":    "10-Q",
        "k_per_ticker": 2,
    })

    # Merge results — combine 10-K and 10-Q chunks per ticker
    all_chunks_by_ticker: dict[str, list] = {}
    for ticker, chunks in results_by_ticker.items():
        all_chunks_by_ticker[ticker] = chunks
    for ticker, chunks in results_10q.items():
        if ticker in all_chunks_by_ticker:
            all_chunks_by_ticker[ticker].extend(chunks)
        else:
            all_chunks_by_ticker[ticker] = chunks

    if not all_chunks_by_ticker:
        logger.debug(f"No chunks found for supply chain topic: {topic}")
        return None

    # Flatten all chunks for state storage
    all_chunks = [
        chunk
        for chunks in all_chunks_by_ticker.values()
        for chunk in chunks
    ]

    logger.info(
        f"Topic '{topic}': found relevant chunks in "
        f"{list(all_chunks_by_ticker.keys())}"
    )

    # Ask GPT-4o to identify stress signals and propagation paths
    analysis = _call_llm(topic, all_chunks_by_ticker)

    if analysis is None:
        return None

    if not analysis.get("stress_detected", False):
        logger.info(f"No supply chain stress detected for topic: {topic}")
        return None

    # Build exposed companies evidence list for the signal
    exposed = analysis.get("exposed_companies", [])
    evidence = [
        {
            "source_company":    analysis.get("source_company", ""),
            "stress_type":       analysis.get("stress_type", ""),
            "source_excerpt":    analysis.get("source_excerpt", "")[:300],
            "exposed_companies": [e["ticker"] for e in exposed],
            "propagation":       analysis.get("propagation_timeline", ""),
        }
    ]

    signal = build_signal(
        signal_type      = "supply_chain",
        ticker           = analysis.get("source_company", "MULTI"),
        generated_by     = "supply_chain_agent",
        headline         = analysis.get("headline", "Supply chain stress detected"),
        detail           = analysis.get("detail", ""),
        confidence_score = float(analysis.get("confidence_score", 0.5)),
        severity         = analysis.get("severity", "medium"),
        evidence         = evidence,
    )

    logger.info(
        f"Supply chain signal: {topic} — "
        f"source={analysis.get('source_company')} "
        f"exposed={[e['ticker'] for e in exposed]} "
        f"severity={analysis.get('severity')}"
    )

    return signal, all_chunks, analysis


def _call_llm(
    topic: str,
    chunks_by_ticker: dict[str, list[dict]],
) -> dict | None:
    """
    Asks GPT-4o to identify supply chain stress and propagation paths
    across the company filings for a given topic.

    Args:
        topic:            The supply chain topic being analysed.
        chunks_by_ticker: Dict of ticker -> list of relevant filing chunks.

    Returns:
        Parsed analysis dict, or None if call failed.
    """
    # Format each company's relevant passages separately
    company_sections = []

    for ticker, chunks in chunks_by_ticker.items():
        if not chunks:
            continue

        excerpts = "\n\n".join(
            f"[{c['form_type']} {c['filing_date']} | {c['section_name']}]\n"
            f"{c['text'][:400]}"
            for c in chunks[:2]
        )

        company_sections.append(f"--- {ticker} ---\n{excerpts}")

    if not company_sections:
        return None

    filing_context = "\n\n".join(company_sections)

    human_prompt = f"""Analyse the following filing excerpts from multiple tech companies 
for supply chain stress signals related to: {topic.replace('_', ' ')}.

{filing_context}

Identify:
1. Which company shows the strongest supply chain stress signal for this topic?
2. Which other companies in the coverage universe (AMZN, MSFT, GOOG, ORCL, AAPL, NVDA) 
   are likely exposed to this signal based on their own disclosures?
3. How severe is the propagation risk?

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
        logger.error(f"JSON parse failed for supply chain topic {topic}: {e}")
        return None

    except Exception as e:
        logger.error(f"LLM call failed for supply chain topic {topic}: {e}")
        return None