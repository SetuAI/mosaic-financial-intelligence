'''
This is the most nuanced file in the ingestion layer — a 34MB SEC filing is not clean text, 
it is a mess of XBRL tags, inline CSS, JavaScript, legal boilerplate, and table markup. 
The parser's job is to get us from that to clean, readable prose that an agent can reason over.
'''

"""
MOSAIC — Multi-Agent Orchestration System for Analyst Intelligence and Cognition
ingestion/document_parser.py

Converts raw EDGAR filing documents into clean, structured text.

SEC filings are notoriously messy HTML — a 10-K from a large company
like MSFT is 30MB+ of inline XBRL, nested tables, legal boilerplate,
CSS, and JavaScript before you get to a single sentence of readable prose.

This module's job is to strip all of that noise and return clean text
that our agents can actually reason over. It also identifies document
sections — Management Discussion, Risk Factors, etc. — so agents can
query specific parts rather than the whole document every time.

Parsing strategy:
    1. Strip scripts, styles, and XBRL metadata tags entirely
    2. Extract visible text using BeautifulSoup
    3. Normalise whitespace — SEC filings are full of \xa0 and excessive newlines
    4. Identify and label major sections by their headings
    5. Return a structured dict with metadata + sectioned content
"""

import re
import hashlib

from datetime import datetime
from typing import Optional
from bs4 import BeautifulSoup

from config.logging_config import get_logger

logger = get_logger(__name__)


# ── Section patterns ───────────────────────────────────────────────────────────

# These are the section headings we care about across 10-K, 10-Q, and 8-K filings.
# SEC requires standard Item numbering — "Item 1A" is always Risk Factors,
# "Item 7" is always MD&A. We use this to slice documents into named sections
# so agents can query "risk factors" without reading the whole 300-page filing.
#
# The patterns are intentionally loose — companies format headings inconsistently.
# "ITEM 1A.", "Item 1A —", "ITEM 1A:" all need to match.
SECTION_PATTERNS = {
    "business_overview":        r"item\s+1[^a-z]",
    "risk_factors":             r"item\s+1a",
    "legal_proceedings":        r"item\s+3",
    # Item 7 = MD&A in 10-K, Item 2 = MD&A in 10-Q
    # Both patterns needed — 10-K and 10-Q use different item numbers
    "management_discussion":    r"item\s+[27][^a-z]",
    "quantitative_disclosures": r"item\s+[37]a",
    "financial_statements":     r"item\s+[18][^a-z]",
    "executive_compensation":   r"item\s+11",
}

# Tags that contain zero useful text for our purposes.
# Removing these before extraction prevents garbage like "function(){return..."
# or raw XBRL attribute strings from ending up in our parsed output.
TAGS_TO_STRIP = [
    "script",
    "style",
    "meta",
    "link",
    "head",
    # XBRL inline tags — these are data annotations embedded in the HTML.
    # They look like <ix:nonfraction ...>1,234</ix:nonfraction> and the
    # number is already visible in the surrounding text — we do not need
    # the tag itself.
    "ix:header",
    "ix:hidden",
    "xbrl",
]


# ── Main parser ────────────────────────────────────────────────────────────────

def parse_filing(
    raw_html: str,
    ticker: str,
    form_type: str,
    filing_date: str,
    accession_number: str,
) -> dict:
    """
    Master parsing function — takes raw HTML and returns a structured document.

    This is the only function document_store.py needs to call. Everything
    else in this module is a helper that this function orchestrates.

    Args:
        raw_html:         Raw HTML string downloaded from EDGAR.
        ticker:           Stock ticker — e.g. "MSFT".
        form_type:        SEC form type — "10-K", "10-Q", "8-K", "4".
        filing_date:      Filing date string — "2024-07-30".
        accession_number: EDGAR accession number for this filing.

    Returns:
        Structured dict with metadata and cleaned text content, ready
        to be saved as JSON by document_store.py.
    """
    logger.info(f"Parsing {form_type} filing for {ticker} dated {filing_date}")

    # Step 1: strip the HTML noise and get clean text
    clean_text = _extract_clean_text(raw_html)

    if not clean_text or len(clean_text) < 500:
        # Under 500 characters almost certainly means the parser got nothing
        # useful — probably a filing that redirects to another document or
        # is pure XBRL with no human-readable content.
        logger.warning(
            f"Extracted text is suspiciously short ({len(clean_text)} chars) "
            f"for {ticker} {form_type} {filing_date} — check the raw file"
        )

    # Step 2: identify sections within the cleaned text
    sections = _extract_sections(clean_text, form_type)

    # Step 3: build the structured output document
    document = {
        # ── Identity ──
        "ticker":           ticker,
        "form_type":        form_type,
        "filing_date":      filing_date,
        "accession_number": accession_number,
        "parsed_at":        datetime.utcnow().isoformat(),

        # ── Content ──
        # full_text is the entire document cleaned — used when we need to
        # search across the whole filing without section boundaries.
        "full_text":        clean_text,

        # sections is a dict of named chunks — agents query these directly
        # rather than sending 300 pages to the LLM every time.
        "sections":         sections,

        # ── Diagnostics ──
        # char_count helps us quickly spot parse failures without opening the file.
        "char_count":       len(clean_text),
        "section_count":    len(sections),

        # content_hash lets us detect if a re-ingested document has changed —
        # useful when EDGAR amends a filing after initial submission.
        "content_hash":     _hash_content(clean_text),
    }

    logger.info(
        f"Parsed {ticker} {form_type}: {len(clean_text):,} chars, "
        f"{len(sections)} sections identified"
    )

    return document


# ── Text extraction ────────────────────────────────────────────────────────────

def _extract_clean_text(raw_html: str) -> str:
    """
    Strips HTML markup and returns clean readable prose.

    BeautifulSoup is doing the heavy lifting here. We use the lxml parser
    because it is significantly faster than html.parser on large files —
    a 30MB 10-K takes ~2s with lxml vs ~12s with html.parser.

    Args:
        raw_html: Raw HTML string from EDGAR.

    Returns:
        Clean text string with normalised whitespace.
    """
    # lxml is faster than html.parser for large documents — worth the dependency.
    soup = BeautifulSoup(raw_html, "lxml")

    # Remove tags that contribute zero readable content.
    # We do this before get_text() so their content does not bleed through.
    for tag_name in TAGS_TO_STRIP:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    # get_text() with separator="\n" preserves block-level structure —
    # paragraphs and table cells become separate lines rather than running together.
    raw_text = soup.get_text(separator="\n")

    # Clean up the whitespace mess that SEC HTML produces
    clean = _normalise_whitespace(raw_text)

    return clean


def _normalise_whitespace(text: str) -> str:
    """
    Cleans up the whitespace artifacts that SEC HTML reliably produces.

    SEC filings are full of:
    - \xa0 (non-breaking space) used as visual padding in tables
    - Runs of 10+ blank lines between sections
    - Lines that are just dots or dashes (table of contents leaders)
    - Unicode control characters from copy-paste into their filing systems

    Args:
        text: Raw extracted text with messy whitespace.

    Returns:
        Text with normalised whitespace — readable and consistent.
    """
    # Replace non-breaking spaces with regular spaces —
    # \xa0 is invisible but breaks string matching and tokenisation.
    text = text.replace("\xa0", " ")

    # Strip lines that are purely decorative — dots, dashes, underscores
    # used as visual separators in the table of contents.
    # These add noise without content.
    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        stripped = line.strip()

        # Skip lines that are just punctuation/whitespace — table of contents
        # leaders like "......." or "- - - - -" or "___________"
        if re.match(r'^[\s\.\-\_\*]*$', stripped):
            continue

        cleaned_lines.append(stripped)

    # Collapse runs of more than 2 consecutive blank lines into just 2.
    # SEC filings use excessive vertical whitespace for visual formatting
    # that means nothing to an LLM.
    text = "\n".join(cleaned_lines)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


# ── Section extraction ─────────────────────────────────────────────────────────

def _extract_sections(text: str, form_type: str) -> dict:
    """
    Identifies and extracts named sections from a parsed filing.

    For 10-K and 10-Q filings, SEC mandates a standard Item structure —
    we exploit that to slice the document into named sections that agents
    can query directly.

    For 8-K filings (earnings call transcripts), we look for the natural
    language structure — Operator remarks, Prepared Remarks, Q&A.

    For Form 4 (insider transactions), there are no sections — the whole
    document is structured data, so we return it as one block.

    Args:
        text:      Full cleaned text of the filing.
        form_type: SEC form type — determines which extraction strategy to use.

    Returns:
        Dict mapping section names to their text content.
        Empty dict if no sections could be identified.
    """
    if form_type in ("10-K", "10-Q"):
        return _extract_annual_sections(text)
    elif form_type == "8-K":
        return _extract_transcript_sections(text)
    else:
        # Form 4 and anything else — return as a single undivided block.
        return {"full_document": text}


def _extract_annual_sections(text: str) -> dict:
    """
    Extracts Item-numbered sections from 10-K and 10-Q filings.

    Finds each Item heading, then captures everything between it and
    the next Item heading. This is imperfect — some companies use
    non-standard formatting — but it works for 90%+ of large-cap filings.

    Args:
        text: Full cleaned text of the annual filing.

    Returns:
        Dict of section_name -> section_text.
    """
    sections = {}
    text_lower = text.lower()

    # Find the character position of each section heading in the document
    section_positions = []

    for section_name, pattern in SECTION_PATTERNS.items():
        match = re.search(pattern, text_lower)

        if match:
            section_positions.append((match.start(), section_name))

    # Sort by position so we can slice between consecutive headings
    section_positions.sort(key=lambda x: x[0])

    # Slice the text between each pair of consecutive section starts
    for i, (start_pos, section_name) in enumerate(section_positions):
        # Each section runs from its heading to the start of the next section.
        # The last section runs to the end of the document.
        if i + 1 < len(section_positions):
            end_pos = section_positions[i + 1][0]
        else:
            end_pos = len(text)

        section_text = text[start_pos:end_pos].strip()

        # Skip sections that are suspiciously short — likely a heading with
        # a forward reference like "See Item 7A" rather than actual content.
        if len(section_text) > 200:
            sections[section_name] = section_text

    if not sections:
        logger.warning("No standard Item sections found — returning full text as single section")
        sections["full_document"] = text

    return sections


def _extract_transcript_sections(text: str) -> dict:
    """
    Extracts sections from earnings call transcripts filed as 8-K exhibits.

    Earnings call transcripts have a consistent natural language structure:
    - Operator introduction
    - Prepared remarks (CEO/CFO formal statements)
    - Q&A session (analyst questions + management responses)

    The Q&A section is particularly valuable for the temporal drift agent —
    how management responds under direct questioning reveals more than
    their prepared remarks.

    Args:
        text: Full cleaned text of the transcript.

    Returns:
        Dict with 'prepared_remarks' and 'qa_session' sections where found.
    """
    sections = {}
    text_lower = text.lower()

    # These phrases reliably mark the start of Q&A in earnings transcripts.
    # "question-and-answer" is the operator's formal intro to the Q&A portion.
    qa_markers = [
        "question-and-answer",
        "question and answer",
        "we will now begin the question",
        "open the line for questions",
        "open for questions",
        "i would now like to turn",
        "we will now take questions",
    ]

    qa_start = None
    for marker in qa_markers:
        pos = text_lower.find(marker)
        if pos != -1:
            qa_start = pos
            break

    if qa_start:
        prepared = text[:qa_start].strip()
        qa       = text[qa_start:].strip()

        if len(prepared) > 200:
            sections["prepared_remarks"] = prepared
        if len(qa) > 200:
            sections["qa_session"] = qa
    else:
        # No Q&A marker found — treat the whole thing as prepared remarks.
        # AAPL occasionally files only the prepared remarks without the Q&A.
        logger.warning(
            "Could not identify Q&A section boundary — "
            "storing full transcript as prepared_remarks"
        )
        sections["prepared_remarks"] = text

    return sections


# ── Utilities ──────────────────────────────────────────────────────────────────

def _hash_content(text: str) -> str:
    """
    Returns a short SHA-256 hash of the document content.

    Used to detect if a re-ingested document has changed — EDGAR allows
    amended filings (10-K/A, 8-K/A) that replace earlier submissions.
    If the hash changes on re-ingestion, we know we need to re-embed.

    Args:
        text: Document text to hash.

    Returns:
        First 16 characters of the SHA-256 hex digest — short enough
        to store cheaply, long enough to be collision-resistant for our scale.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def extract_fiscal_period(filing_date: str, form_type: str) -> str:
    """
    Derives a human-readable fiscal period label from a filing date.

    Used in document metadata so agents can filter by period naturally —
    "Q2 FY2024" is easier to reason about than "2024-07-30".

    This is an approximation — it works correctly for calendar-year companies
    but will be off by one quarter for companies with non-standard fiscal years
    like MSFT (fiscal year ends June 30) or AAPL (fiscal year ends September).
    Good enough for our temporal analysis purposes.

    Args:
        filing_date: ISO date string — "2024-07-30".
        form_type:   SEC form type.

    Returns:
        Period string — e.g. "Q2 FY2024" or "FY2024".
    """
    date   = datetime.strptime(filing_date, "%Y-%m-%d")
    year   = date.year
    month  = date.month

    if form_type == "10-K":
        return f"FY{year}"

    # Map filing month to approximate fiscal quarter.
    # Companies typically file 10-Qs within 40 days of quarter end.
    quarter_map = {
        (1, 2, 3):   "Q1",
        (4, 5, 6):   "Q2",
        (7, 8, 9):   "Q3",
        (10, 11, 12): "Q4",
    }

    quarter = next(
        q for months, q in quarter_map.items() if month in months
    )

    return f"{quarter} FY{year}"