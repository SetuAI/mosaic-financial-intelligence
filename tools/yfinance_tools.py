"""
MOSAIC — Multi-Agent Orchestration System for Analyst Intelligence and Cognition
tools/yfinance_tools.py

Wraps yFinance market data calls as LangGraph tools for agent use.

Why yFinance alongside EDGAR:
    EDGAR gives us what companies officially reported — audited financials,
    legal filings. yFinance gives us market reaction and consensus data —
    what the market thought about those reports. Together they let the
    credibility scorer ask: "management guided 20% growth, reported 15%,
    and the stock dropped 8% the next day" — that full picture is only
    possible by combining both sources.

Run standalone? No — imported by agents
"""

import yfinance as yf
from langchain_core.tools import tool
from config.logging_config import get_logger

logger = get_logger(__name__)


@tool
def get_stock_price_history(
    ticker: str,
    period: str = "2y",
) -> list[dict]:
    """
    Returns historical stock price data for a company.

    Use this tool to correlate price movements with filing dates and
    earnings announcements — a key input for the credibility scorer.
    A stock that drops sharply after an earnings call where management
    sounded confident is a signal worth investigating.

    Args:
        ticker: Stock ticker — "MSFT", "AMZN", etc.
        period: Time period — "1y", "2y", "5y". Default "2y" matches
                our filing lookback window.

    Returns:
        List of daily OHLCV records, each containing:
            date, open, high, low, close, volume
        Sorted oldest-first.
    """
    logger.info(f"yFinance: fetching {period} price history for {ticker}")

    try:
        stock = yf.Ticker(ticker)
        hist  = stock.history(period=period)

        if hist.empty:
            logger.warning(f"No price history returned for {ticker}")
            return []

        records = []
        for date, row in hist.iterrows():
            records.append({
                "date":   date.strftime("%Y-%m-%d"),
                "open":   round(float(row["Open"]),   2),
                "high":   round(float(row["High"]),   2),
                "low":    round(float(row["Low"]),    2),
                "close":  round(float(row["Close"]),  2),
                "volume": int(row["Volume"]),
            })

        return records

    except Exception as e:
        logger.error(f"yFinance price history failed for {ticker}: {e}")
        return [{"error": str(e)}]


@tool
def get_earnings_history(ticker: str) -> list[dict]:
    """
    Returns historical earnings results — actual EPS vs analyst estimates.

    Use this tool for the credibility scorer — it shows whether management's
    guidance translated into beats or misses against market expectations.
    Consistent guidance misses reduce the credibility score significantly.

    Args:
        ticker: Stock ticker.

    Returns:
        List of quarterly earnings records, each containing:
            date, eps_actual, eps_estimate, surprise_percent
        Sorted newest-first.
    """
    logger.info(f"yFinance: fetching earnings history for {ticker}")

    try:
        stock    = yf.Ticker(ticker)
        earnings = stock.earnings_history

        if earnings is None or earnings.empty:
            logger.warning(f"No earnings history returned for {ticker}")
            return []

        records = []
        for _, row in earnings.iterrows():
            eps_actual   = row.get("epsActual")
            eps_estimate = row.get("epsEstimate")

            # Compute surprise percentage — how far actual diverged from estimate.
            # Positive = beat, negative = miss. This is what the credibility
            # scorer uses to penalise or reward management guidance accuracy.
            if eps_estimate and eps_estimate != 0:
                surprise_pct = round(
                    ((eps_actual - eps_estimate) / abs(eps_estimate)) * 100, 2
                )
            else:
                surprise_pct = None

            records.append({
                "date":             str(row.get("quarterDate", "")),
                "eps_actual":       eps_actual,
                "eps_estimate":     eps_estimate,
                "surprise_percent": surprise_pct,
            })

        return records

    except Exception as e:
        logger.error(f"yFinance earnings history failed for {ticker}: {e}")
        return [{"error": str(e)}]


@tool
def get_company_info(ticker: str) -> dict:
    """
    Returns general company information — sector, industry, market cap,
    employee count, and business summary.

    Use this tool when an agent needs context about a company before
    analysing its filings — particularly useful for the supply chain
    agent when assessing how exposed a company is to upstream suppliers.

    Args:
        ticker: Stock ticker.

    Returns:
        Dict with company metadata fields.
    """
    logger.info(f"yFinance: fetching company info for {ticker}")

    try:
        stock = yf.Ticker(ticker)
        info  = stock.info

        # Extract only the fields relevant to our agents —
        # yFinance info contains 80+ fields most of which we do not need.
        return {
            "ticker":           ticker,
            "name":             info.get("longName", ""),
            "sector":           info.get("sector", ""),
            "industry":         info.get("industry", ""),
            "market_cap":       info.get("marketCap"),
            "employees":        info.get("fullTimeEmployees"),
            "description":      info.get("longBusinessSummary", "")[:500],
            "revenue_ttm":      info.get("totalRevenue"),
            "gross_margins":    info.get("grossMargins"),
            "operating_margins": info.get("operatingMargins"),
            "profit_margins":   info.get("profitMargins"),
            "pe_ratio":         info.get("trailingPE"),
            "52w_high":         info.get("fiftyTwoWeekHigh"),
            "52w_low":          info.get("fiftyTwoWeekLow"),
        }

    except Exception as e:
        logger.error(f"yFinance company info failed for {ticker}: {e}")
        return {"error": str(e)}


@tool
def get_price_on_date(ticker: str, date: str) -> dict:
    """
    Returns the closing stock price on or near a specific date.

    Use this tool when the insider signal agent needs to know the stock
    price on the date an insider transaction was filed — to assess whether
    the insider sold near a local high or bought near a local low relative
    to subsequent price movement.

    Args:
        ticker: Stock ticker.
        date:   Date string in "YYYY-MM-DD" format.

    Returns:
        Dict with date and closing price, or error if not found.
    """
    logger.info(f"yFinance: fetching price for {ticker} on {date}")

    try:
        stock = yf.Ticker(ticker)

        # Fetch a small window around the date — yFinance does not always
        # have data for exact dates (weekends, holidays). A 5-day window
        # ensures we get the nearest trading day.
        from datetime import datetime, timedelta
        target   = datetime.strptime(date, "%Y-%m-%d")
        start    = (target - timedelta(days=3)).strftime("%Y-%m-%d")
        end      = (target + timedelta(days=3)).strftime("%Y-%m-%d")

        hist = stock.history(start=start, end=end)

        if hist.empty:
            return {"error": f"No price data found for {ticker} around {date}"}

        # Return the row closest to the requested date
        closest_date = min(hist.index, key=lambda d: abs((d.date() - target.date()).days))
        row          = hist.loc[closest_date]

        return {
            "ticker":       ticker,
            "requested_date": date,
            "actual_date":  closest_date.strftime("%Y-%m-%d"),
            "close":        round(float(row["Close"]), 2),
            "volume":       int(row["Volume"]),
        }

    except Exception as e:
        logger.error(f"yFinance price on date failed for {ticker} {date}: {e}")
        return {"error": str(e)}