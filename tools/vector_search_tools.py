"""
MOSAIC — Multi-Agent Orchestration System for Analyst Intelligence and Cognition
tools/vector_search_tools.py

Wraps the vector store as LangGraph-compatible tools that agents call
during their reasoning loop.

Why tools rather than direct function calls:
    LangGraph agents decide at runtime which tools to call and with what
    arguments. Wrapping search as a tool means the agent can reason:
    "I need to find what MSFT said about supply chain risk across their
    last two 10-Ks" and call the right tool with the right parameters —
    rather than us hardcoding which documents each agent reads.

    Tools also give us automatic LangSmith tracing — every tool call,
    its inputs, and its outputs are logged without any extra code.

Run standalone? No — imported by agents
"""

from langchain_core.tools import tool
from processing.vector_store import vector_store
from processing.embedder import embed_query
from config.logging_config import get_logger

logger = get_logger(__name__)

# Load the vector store once at import time — agents share this instance.
# If the index has not been built yet this raises FileNotFoundError with
# a clear message telling the user to run run_processing.py first.
try:
    if not vector_store.is_ready:
        vector_store.load()
        logger.info(f"Vector store loaded — {vector_store.index.ntotal} vectors ready")
except FileNotFoundError:
    logger.warning(
        "Vector store not found at startup — "
        "call vector_store.load() before running agents"
    )


@tool
def search_filings(
    query: str,
    ticker: str = "",
    form_type: str = "",
    section_name: str = "",
    k: int = 5,
) -> list[dict]:
    """
    Searches the MOSAIC vector store for filing chunks semantically similar
    to the query. Returns the most relevant passages from SEC filings.

    Use this tool when you need to find what a company said about a specific
    topic across their filings — earnings calls, annual reports, or quarterly
    filings. Results are ranked by semantic similarity, not keyword match.

    Args:
        query:        Natural language search query. Be specific —
                      "Azure cloud revenue growth guidance" returns better
                      results than just "revenue".
        ticker:       Optional. Filter to one company — "MSFT", "AMZN", etc.
                      Leave empty to search across all companies.
        form_type:    Optional. Filter to one filing type — "10-K", "10-Q",
                      "8-K", or "4". Leave empty to search all types.
        section_name: Optional. Filter to a document section —
                      "risk_factors", "management_discussion",
                      "prepared_remarks", "qa_session".
        k:            Number of results to return. Default 5, max 20.

    Returns:
        List of chunk dicts, each containing:
            ticker, form_type, filing_date, section_name,
            text (the passage), similarity_score (0-1, higher is better)
    """
    if not vector_store.is_ready:
        return [{"error": "Vector store not loaded. Run processing pipeline first."}]

    # Cap k at 20 — returning more than 20 chunks floods the agent context
    # and dilutes the quality of the top results.
    k = min(k, 20)

    logger.info(
        f"Vector search: '{query[:60]}' "
        f"[ticker={ticker or 'all'} form={form_type or 'all'} k={k}]"
    )

    query_vector = embed_query(query)

    results = vector_store.search(
        query_vector = query_vector,
        k            = k,
        ticker       = ticker or None,
        form_type    = form_type or None,
        section_name = section_name or None,
    )

    logger.info(f"Search returned {len(results)} results")

    # Return only the fields agents need — dropping chunk_id and position
    # metadata keeps the context clean and focused on content.
    return [
        {
            "ticker":           r["ticker"],
            "form_type":        r["form_type"],
            "filing_date":      r["filing_date"],
            "section_name":     r["section_name"],
            "text":             r["text"],
            "similarity_score": r["similarity_score"],
        }
        for r in results
    ]


@tool
def search_across_companies(
    query: str,
    tickers: list[str],
    form_type: str = "10-K",
    k_per_ticker: int = 2,
) -> dict[str, list[dict]]:
    """
    Searches for a topic across multiple companies simultaneously.

    Use this tool when you need to compare what multiple companies said
    about the same topic — for example, how MSFT, GOOG, and AMZN each
    discussed AI investment in their annual reports.

    This is the supply chain agent's primary tool — it searches all
    downstream companies when an upstream signal is detected.

    Args:
        query:        What to search for across all specified companies.
        tickers:      List of tickers to search — e.g. ["MSFT", "GOOG", "AMZN"].
        form_type:    Filing type to search. Default "10-K".
        k_per_ticker: Results per company. Default 2 — keeps context manageable
                      when searching across many companies simultaneously.

    Returns:
        Dict mapping ticker -> list of relevant chunks for that ticker.
        Companies with no relevant results are excluded from the output.
    """
    if not vector_store.is_ready:
        return {"error": "Vector store not loaded. Run processing pipeline first."}

    query_vector = embed_query(query)
    results_by_ticker: dict[str, list[dict]] = {}

    logger.info(
        f"Cross-company search: '{query[:60]}' "
        f"across {tickers} [{form_type}]"
    )

    for ticker in tickers:
        results = vector_store.search(
            query_vector = query_vector,
            k            = k_per_ticker,
            ticker       = ticker,
            form_type    = form_type or None,
        )

        if results:
            results_by_ticker[ticker] = [
                {
                    "ticker":           r["ticker"],
                    "form_type":        r["form_type"],
                    "filing_date":      r["filing_date"],
                    "section_name":     r["section_name"],
                    "text":             r["text"],
                    "similarity_score": r["similarity_score"],
                }
                for r in results
            ]

    logger.info(
        f"Cross-company search complete — "
        f"results for {list(results_by_ticker.keys())}"
    )

    return results_by_ticker


@tool
def search_temporal(
    query: str,
    ticker: str,
    form_type: str = "10-K",
    k: int = 8,
) -> list[dict]:
    """
    Searches for a topic across all filings for one company, sorted
    chronologically oldest to newest.

    Use this tool when you need to track how a company's language or
    position on a topic has evolved over time — the core input for
    temporal drift analysis.

    Results are sorted by filing_date ascending so the temporal drift
    agent can read them in chronological order and detect shifts.

    Args:
        query:     What topic to track over time.
        ticker:    The company to analyse — e.g. "MSFT".
        form_type: Filing type. Default "10-K" for annual drift analysis.
                   Use "10-Q" for quarterly drift.
        k:         Number of results. Default 8 — covers 2 years of
                   quarterly filings. Increase for longer history.

    Returns:
        List of chunk dicts sorted oldest-first by filing_date.
    """
    if not vector_store.is_ready:
        return [{"error": "Vector store not loaded. Run processing pipeline first."}]

    query_vector = embed_query(query)

    results = vector_store.search(
        query_vector = query_vector,
        k            = k,
        ticker       = ticker,
        form_type    = form_type or None,
    )

    # Sort chronologically — temporal drift agent reads oldest to newest
    results.sort(key=lambda r: r["filing_date"])

    logger.info(
        f"Temporal search: '{query[:60]}' for {ticker} "
        f"— {len(results)} results across "
        f"{len(set(r['filing_date'] for r in results))} filing dates"
    )

    return [
        {
            "ticker":           r["ticker"],
            "form_type":        r["form_type"],
            "filing_date":      r["filing_date"],
            "section_name":     r["section_name"],
            "text":             r["text"],
            "similarity_score": r["similarity_score"],
        }
        for r in results
    ]