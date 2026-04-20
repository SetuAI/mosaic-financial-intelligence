"""
MOSAIC — Multi-Agent Orchestration System for Analyst Intelligence and Cognition
ingestion/edgar_client.py

Handles every interaction with the SEC EDGAR API.

This module's only job is to talk to EDGAR and return raw data.
It does not parse, clean, or interpret anything — that is document_parser.py's job.
Keeping the boundary clean means if EDGAR changes their API format, we fix
it here and nowhere else touches it.

Lessons learned during development:
    - EDGAR submissions endpoint uses zero-padded CIKs (e.g. "0000789019")
    - EDGAR archive URLs use unpadded integer CIKs (e.g. 789019) — different,
      and critical. Sending a padded CIK to the archive causes a 301 redirect
      that httpx does not follow by default, breaking the download silently.
    - The -index.json endpoint returns 404 for many filings — unreliable.
      The HTML directory listing is the only consistently available index.
    - Exhibit filenames follow loose conventions — ex99_1, ex99-1, ex991
      all mean EX-99.1. We handle all known variants in _classify_document().

EDGAR API reference: https://www.sec.gov/developer
Rate limit: 10 requests/second — we stay well under at one per 0.5s.
"""

import time
import httpx

from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from typing import Optional

from config.settings import settings
from config.logging_config import get_logger

logger = get_logger(__name__)


# ── Constants ──────────────────────────────────────────────────────────────────

# Two base URLs serving different purposes — mixing them causes failures.
# DATA_URL:    structured JSON API for filing metadata and company facts
# ARCHIVE_URL: raw filing documents — HTML, text, exhibits
DATA_URL    = "https://data.sec.gov"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data"

# EDGAR validates the Host header and will reject requests where it does
# not match the domain being called. Keep these in sync with their URLs.
DATA_HEADERS = {
    "User-Agent":      settings.sec_user_agent,
    "Accept-Encoding": "gzip, deflate",
    "Host":            "data.sec.gov",
}

ARCHIVE_HEADERS = {
    "User-Agent":      settings.sec_user_agent,
    "Accept-Encoding": "gzip, deflate",
    "Host":            "www.sec.gov",
}


# ── Core HTTP helper ───────────────────────────────────────────────────────────

def _get(url: str, headers: dict, retries: int = settings.sec_max_retries) -> httpx.Response:
    """
    Makes a single GET request to EDGAR with retry logic and rate limiting.

    Every EDGAR call in this module goes through here — centralising
    the retry and rate-limit logic means we fix it in one place if
    EDGAR's behaviour changes.

    The sleep before every request is not optional — EDGAR has blocked
    IPs that ignore their rate limit, and recovering from a block mid-run
    is painful. 0.5s keeps us comfortably under their 10 req/s cap.

    Args:
        url:     Full URL to fetch.
        headers: Request headers — use DATA_HEADERS or ARCHIVE_HEADERS above.
        retries: How many times to retry on transient failures before giving up.

    Returns:
        httpx.Response with the raw response.

    Raises:
        httpx.HTTPStatusError: If the request fails after all retries.
    """
    time.sleep(settings.sec_request_delay)

    last_exception = None

    for attempt in range(1, retries + 1):
        try:
            response = httpx.get(
                url,
                headers=headers,
                timeout=30.0,
                follow_redirects=True,   # handles any remaining redirects gracefully
            )
            response.raise_for_status()
            return response

        except httpx.HTTPStatusError as e:
            status = e.response.status_code

            if status == 429:
                # Rate limited — back off exponentially before retrying.
                wait = 2 ** attempt
                logger.warning(
                    f"Rate limited by EDGAR. Waiting {wait}s "
                    f"before retry {attempt}/{retries}"
                )
                time.sleep(wait)

            elif status == 404:
                # 404 is definitive — retrying will not help.
                logger.error(f"EDGAR returned 404 for {url} — not retrying")
                raise

            elif 400 <= status < 500:
                # Other 4xx client errors will not resolve with a retry.
                logger.error(f"EDGAR returned {status} for {url} — not retrying")
                raise

            last_exception = e

        except httpx.RequestError as e:
            # Network-level errors — DNS failure, timeout, connection reset.
            # Worth retrying with exponential backoff.
            logger.warning(f"Network error on attempt {attempt}/{retries}: {e}")
            last_exception = e
            time.sleep(2 ** attempt)

    logger.error(f"All {retries} attempts failed for {url}")
    raise last_exception


# ── Company lookup ─────────────────────────────────────────────────────────────

def get_cik(ticker: str) -> str:
    """
    Returns the zero-padded CIK for a ticker from our configured map.

    We use a local map rather than hitting EDGAR's ticker-to-CIK lookup
    endpoint because our coverage universe is fixed — no point making a
    network call for something that never changes for these 6 companies.

    IMPORTANT: This returns the zero-padded CIK string (e.g. "0000789019").
    - Use as-is for the submissions API and company facts endpoint.
    - For archive URLs, always wrap with int() to strip leading zeros:
      int(get_cik("MSFT")) → 789019
    Sending a padded CIK to the archive causes a 301 redirect.

    Args:
        ticker: Stock ticker symbol — e.g. "AMZN", "MSFT".

    Returns:
        Zero-padded 10-digit CIK string — e.g. "0000789019".

    Raises:
        ValueError: If the ticker is not in our coverage universe.
    """
    ticker = ticker.upper().strip()

    if ticker not in settings.ticker_cik_map:
        raise ValueError(
            f"'{ticker}' is not in our coverage universe. "
            f"Covered tickers: {list(settings.ticker_cik_map.keys())}"
        )

    return settings.ticker_cik_map[ticker]


# ── Filing metadata ────────────────────────────────────────────────────────────

def get_filing_metadata(ticker: str) -> dict:
    """
    Fetches the full submission history for a company from EDGAR.

    EDGAR returns all filings in a single JSON response — it is a large
    payload but the structure is consistent and well-documented.

    NOTE: EDGAR returns the 1000 most recent filings in the main payload.
    For companies like MSFT with 30+ years of filing history that is fine
    for our 2-year window. Older filings live in response["filings"]["files"]
    as paginated references — not needed for MOSAIC's scope.

    Args:
        ticker: Stock ticker — e.g. "AMZN".

    Returns:
        The full EDGAR submissions JSON for the company.
    """
    cik = get_cik(ticker)

    # Submissions endpoint uses the zero-padded CIK — this is intentional.
    url = f"{DATA_URL}/submissions/CIK{cik}.json"

    logger.info(f"Fetching filing metadata for {ticker} (CIK: {cik})")
    response = _get(url, headers=DATA_HEADERS)

    return response.json()


def get_recent_filings(ticker: str, form_type: str) -> list[dict]:
    """
    Returns filings of a specific type for a ticker within our lookback window.

    EDGAR's API returns each field as a parallel array rather than a list
    of objects — an unusual design choice. We zip those arrays into proper
    dicts here so the rest of the codebase never has to think about it.

    Args:
        ticker:    Stock ticker — e.g. "GOOG".
        form_type: SEC form type — "8-K", "10-K", "10-Q", or "4".

    Returns:
        List of filing dicts, each containing:
            accessionNumber  — unique filing ID used to build download URLs
            filingDate       — date filed with the SEC
            form             — the form type
            primaryDocument  — filename of the main document in the package
    """
    metadata = get_filing_metadata(ticker)
    recent   = metadata["filings"]["recent"]

    # Zip EDGAR's parallel arrays into a list of per-filing dicts.
    all_filings = [
        dict(zip(recent.keys(), values))
        for values in zip(*recent.values())
    ]

    # Keep only the form type we care about
    typed_filings = [f for f in all_filings if f["form"] == form_type]

    # Keep only filings within our configured lookback window
    cutoff_date   = datetime.now() - timedelta(days=365 * settings.lookback_years)
    recent_filings = [
        f for f in typed_filings
        if datetime.strptime(f["filingDate"], "%Y-%m-%d") >= cutoff_date
    ]

    logger.info(
        f"{ticker} — found {len(recent_filings)} {form_type} filings "
        f"in the last {settings.lookback_years} years"
    )

    return recent_filings


# ── Filing index ───────────────────────────────────────────────────────────────

def get_filing_index(ticker: str, accession_number: str) -> list[dict]:
    """
    Fetches the list of documents in a filing by parsing the HTML directory.

    We discovered during development that EDGAR's -index.json endpoint
    returns 404 for a large proportion of filings — it is simply not
    reliable across all filers and filing types. The HTML directory
    listing at the accession folder URL is always present and contains
    the same file information, so we parse that instead.

    Args:
        ticker:           Stock ticker.
        accession_number: EDGAR accession number — e.g. "0001193125-26-027198".

    Returns:
        List of dicts: [{'filename': 'msft-ex99_1.htm', 'type': 'EX-99.1'}, ...]
    """
    # Archive URLs need the unpadded integer CIK — no leading zeros.
    cik_int         = int(get_cik(ticker))
    accession_clean = accession_number.replace("-", "")

    url = f"{ARCHIVE_URL}/{cik_int}/{accession_clean}/"

    logger.debug(f"Fetching filing directory for {ticker} / {accession_number}")
    response = _get(url, headers=ARCHIVE_HEADERS)

    return _parse_filing_directory(response.text)


def _parse_filing_directory(html: str) -> list[dict]:
    """
    Parses the EDGAR filing directory HTML page to extract document filenames.

    The directory page is a plain HTML file listing. We pull all links
    that point into the Archives path and classify each one by its filename
    pattern so find_exhibit() can match them by type.

    Filters out non-document files — XBRL schemas, ZIP archives, CSS, JS,
    JSON metadata — since we only need the human-readable HTML documents.

    Args:
        html: Raw HTML of the EDGAR filing directory page.

    Returns:
        List of dicts with 'filename' and 'type' for each document.
    """
    soup      = BeautifulSoup(html, "lxml")
    documents = []

    for link in soup.find_all("a", href=True):
        href     = link["href"]
        filename = href.split("/")[-1]

        # Skip navigation links and empty filenames
        if not filename or not href.startswith("/Archives/"):
            continue

        # Skip non-document files — only HTML and text files are useful
        if filename.endswith((".zip", ".xsd", ".xml", ".css", ".js", ".json")):
            continue

        # Skip index navigation files — they are not content
        if "index" in filename.lower():
            continue

        doc_type = _classify_document(filename)
        documents.append({"filename": filename, "type": doc_type})

    logger.debug(f"Found {len(documents)} documents in filing directory")
    return documents


def _classify_document(filename: str) -> str:
    """
    Infers the exhibit type from the filename.

    EDGAR has no strict filename convention for exhibits — companies
    name them inconsistently across filers and even across quarters
    for the same company. These patterns cover the vast majority of
    naming conventions observed across our 6 tickers.

    Observed patterns in the wild:
        msft-ex99_1.htm    →  EX-99.1  (Microsoft)
        amzn-ex991.htm     →  EX-99.1  (Amazon)
        ex99-1.htm         →  EX-99.1  (generic)
        ex991.htm          →  EX-99.1  (Oracle style)
        goog-ex991.htm     →  EX-99.1  (Alphabet)

    Args:
        filename: Document filename — e.g. "msft-ex99_1.htm".

    Returns:
        "EX-99.1", "EX-99.2", or "PRIMARY".
    """
    fname = filename.lower()

    # EX-99.1 — primary earnings exhibit, transcript or press release
    if any(p in fname for p in ["ex99_1", "ex99-1", "ex991", "ex-99_1", "ex-991"]):
        return "EX-99.1"

    # EX-99.2 — secondary exhibit. AAPL splits prepared remarks (EX-99.1)
    # and Q&A (EX-99.2) into separate files — handle both.
    if any(p in fname for p in ["ex99_2", "ex99-2", "ex992", "ex-99_2", "ex-992"]):
        return "EX-99.2"

    # Everything else is the primary 8-K cover page — useful as fallback
    # but not the content we are after for transcript analysis.
    return "PRIMARY"


def find_exhibit(filing_documents: list[dict], exhibit_type: str) -> Optional[str]:
    """
    Searches a filing document list for a specific exhibit type.

    Args:
        filing_documents: List of dicts from get_filing_index().
        exhibit_type:     Exhibit type to find — e.g. "EX-99.1".

    Returns:
        Filename of the matching exhibit, or None if not found.
    """
    for doc in filing_documents:
        if doc.get("type", "").upper() == exhibit_type.upper():
            return doc["filename"]

    return None


# ── Document fetching ──────────────────────────────────────────────────────────

def get_document_text(ticker: str, accession_number: str, filename: str) -> str:
    """
    Downloads the raw text or HTML of a specific document within a filing.

    This is the final step in the fetch chain — we have identified the filing,
    found the right exhibit in its directory, and now pull the actual content.
    document_parser.py handles everything after this point.

    Args:
        ticker:           Stock ticker.
        accession_number: Filing's unique ID.
        filename:         Specific file to fetch — e.g. "msft-ex99_1.htm".

    Returns:
        Raw document content as a string — HTML or plain text, unparsed.
    """
    # Unpadded integer CIK for all archive URLs — this is the fix that
    # resolved the 301 redirect issue from our initial development run.
    cik_int         = int(get_cik(ticker))
    accession_clean = accession_number.replace("-", "")

    url = f"{ARCHIVE_URL}/{cik_int}/{accession_clean}/{filename}"

    logger.debug(f"Downloading: {filename} from {ticker} / {accession_number}")
    response = _get(url, headers=ARCHIVE_HEADERS)

    return response.text


# ── Company financial facts ────────────────────────────────────────────────────

def get_company_facts(ticker: str) -> dict:
    """
    Fetches all structured financial data EDGAR has for a company.

    Returns every financial metric the company has ever reported in XBRL —
    revenue, EPS, margins, share counts, and more. The credibility scorer
    uses this to verify what management actually delivered vs. what they
    guided on their earnings calls.

    NOTE: This payload is large — MSFT's facts JSON is ~15MB.
    Fetch once and let document_store.py cache it locally.
    Do not call this in a loop.

    Args:
        ticker: Stock ticker.

    Returns:
        Company facts JSON — structured financial data by metric and period.
    """
    cik = get_cik(ticker)

    # Company facts uses the zero-padded CIK — this is the data API endpoint.
    url = f"{DATA_URL}/api/xbrl/companyfacts/CIK{cik}.json"

    logger.info(f"Fetching company financial facts for {ticker}")

    # Longer timeout for this one — the payload can be 15MB+ and EDGAR's
    # XBRL endpoint is occasionally slow to respond.
    time.sleep(settings.sec_request_delay)
    response = httpx.get(
        url,
        headers=DATA_HEADERS,
        timeout=60.0,
        follow_redirects=True,
    )
    response.raise_for_status()

    return response.json()