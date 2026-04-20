"""
MOSAIC — Multi-Agent Orchestration System for Analyst Intelligence and Cognic
tools/edgar_tools.py

Wraps EDGAR API calls as LangGraph tools for agent use.

These tools give agents the ability to fetch fresh data from EDGAR on demand —
useful when an agent needs a specific filing that may not be in the vector
store yet, or when it needs structured metadata about filings rather than
the full text content.

Run standalone? No — imported by agents
"""

from langchain_core.tools import tool
from ingestion.edgar_client import (
    get_recent_filings,
    get_company_facts,
    get_cik,
)
from config.logging_config import get_logger

logger = get_logger(__name__)


@tool
def get_filing_list(ticker: str, form_type: str) -> list[dict]:
    """
    Returns the list of recent filings for a company from EDGAR.

    Use this tool when you need to know what filings exist for a company
    and their filing dates — for example, to check when AMZN last filed
    a 10-K or to get the accession numbers for specific quarters.

    Args:
        ticker:    Stock ticker — "AMZN", "MSFT", "GOOG", "ORCL", "AAPL", "NVDA".
        form_type: SEC form type — "10-K", "10-Q", "8-K", or "4".

    Returns:
        List of filing dicts, each containing:
            accessionNumber, filingDate, form, primaryDocument
        Sorted newest-first.
    """
    logger.info(f"EDGAR tool: fetching {form_type} filing list for {ticker}")

    try:
        filings = get_recent_filings(ticker, form_type)

        # Return only the fields agents need — full metadata has many
        # irrelevant fields that waste context window space.
        return [
            {
                "accession_number": f["accessionNumber"],
                "filing_date":      f["filingDate"],
                "form_type":        f["form"],
                "primary_document": f.get("primaryDocument", ""),
                "report_date":      f.get("reportDate", ""),
            }
            for f in filings
        ]

    except Exception as e:
        logger.error(f"EDGAR filing list failed for {ticker} {form_type}: {e}")
        return [{"error": str(e)}]


@tool
def get_financial_facts(ticker: str, metric: str) -> list[dict]:
    """
    Fetches structured financial data for a company from EDGAR's XBRL database.

    Use this tool when you need actual reported numbers — not what management
    said they expected, but what the company actually filed with the SEC.
    This is the primary data source for the credibility scorer — it compares
    management guidance against these actual reported figures.

    Common metrics to query:
        "Revenues"                    — total revenue
        "NetIncomeLoss"               — net income
        "OperatingIncomeLoss"         — operating income
        "GrossProfit"                 — gross profit
        "EarningsPerShareBasic"       — basic EPS
        "EarningsPerShareDiluted"     — diluted EPS
        "OperatingExpenses"           — total operating expenses

    Args:
        ticker: Stock ticker.
        metric: XBRL concept name — use the exact strings listed above.
                XBRL names are case-sensitive.

    Returns:
        List of reported values sorted newest-first, each containing:
            value, period_end, form_type, filed_date, unit
        Returns empty list if the metric is not found for this company.
    """
    logger.info(f"EDGAR tool: fetching {metric} for {ticker}")

    try:
        facts = get_company_facts(ticker)

        # EDGAR structures facts as:
        # facts["us-gaap"][metric_name]["units"]["USD"][list of reports]
        # Some metrics are in shares or pure numbers rather than USD.
        us_gaap = facts.get("facts", {}).get("us-gaap", {})

        if metric not in us_gaap:
            logger.warning(f"Metric '{metric}' not found in EDGAR facts for {ticker}")
            return []

        metric_data = us_gaap[metric]
        units       = metric_data.get("units", {})

        # Try USD first — most financial metrics are in dollars.
        # Fall back to shares or pure numbers for EPS and per-share metrics.
        values = (
            units.get("USD") or
            units.get("shares") or
            units.get("pure") or
            []
        )

        if not values:
            return []

        # Filter to annual and quarterly reports only — ignore 8-K amendments
        # which can create duplicate entries for the same period.
        filtered = [
            v for v in values
            if v.get("form") in ("10-K", "10-Q")
        ]

        # Sort newest first and return the last 8 periods —
        # enough for 2 years of quarterly data without flooding context.
        filtered.sort(key=lambda v: v.get("end", ""), reverse=True)
        recent = filtered[:8]

        return [
            {
                "value":       v.get("val"),
                "period_end":  v.get("end"),
                "period_start": v.get("start", ""),
                "form_type":   v.get("form"),
                "filed_date":  v.get("filed"),
                "unit":        list(units.keys())[0],
            }
            for v in recent
        ]

    except Exception as e:
        logger.error(f"EDGAR financial facts failed for {ticker} {metric}: {e}")
        return [{"error": str(e)}]


@tool
def get_insider_transactions(ticker: str) -> list[dict]:
    """
    Returns recent insider transaction filings (Form 4) for a company.

    Use this tool when investigating whether insider trading patterns
    align with or contradict management's public statements on earnings
    calls. A CEO selling large positions shortly after bullish guidance
    is a signal worth flagging for HITL review.

    Args:
        ticker: Stock ticker.

    Returns:
        List of Form 4 filing metadata sorted newest-first.
        Note: Returns filing metadata only — not the transaction details.
        The transaction details are in the processed Form 4 documents
        in the vector store, searchable via search_filings().
    """
    logger.info(f"EDGAR tool: fetching Form 4 filings for {ticker}")

    try:
        filings = get_recent_filings(ticker, "4")

        return [
            {
                "accession_number": f["accessionNumber"],
                "filing_date":      f["filingDate"],
                "form_type":        "4",
                "primary_document": f.get("primaryDocument", ""),
            }
            for f in filings
        ]

    except Exception as e:
        logger.error(f"Form 4 fetch failed for {ticker}: {e}")
        return [{"error": str(e)}]


@tool
def get_company_cik(ticker: str) -> dict:
    """
    Returns the SEC CIK number for a ticker.

    Utility tool — agents use this when they need to construct EDGAR
    URLs directly or reference a company by its SEC identifier.

    Args:
        ticker: Stock ticker.

    Returns:
        Dict with ticker and cik fields.
    """
    try:
        cik = get_cik(ticker)
        return {"ticker": ticker, "cik": cik}
    except ValueError as e:
        return {"error": str(e)}