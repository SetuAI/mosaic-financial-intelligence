"""
MOSAIC — Multi-Agent Orchestration System for Analyst Intelligence and Cognition
processing/chunker.py

Splits large parsed documents into smaller overlapping chunks suitable
for embedding and semantic search.

Why chunking is necessary:
    A MSFT 10-K is ~354,000 characters — roughly 88,000 tokens. GPT-4o's
    context window is 128,000 tokens, and even within that limit, attention
    quality degrades on very long inputs. The model misses things buried
    in the middle of a 300-page document.

    Chunking breaks documents into ~1,000 token pieces that fit cleanly
    in context. The overlap between consecutive chunks ensures that a key
    sentence falling at a chunk boundary appears in both adjacent chunks
    and is never lost during retrieval.

Chunking strategy:
    We chunk at the section level, not the document level. A 10-K has
    distinct sections — Risk Factors, MD&A, Legal Proceedings. Chunking
    within sections preserves semantic coherence and lets agents query
    "risk factors" without getting chunks from financial statements mixed in.

    For documents with no identified sections (Form 4, some 8-Ks), we
    chunk the full text directly.
"""

import tiktoken

from typing import Generator
from config.settings import settings
from config.logging_config import get_logger

logger = get_logger(__name__)


# ── Tokeniser setup ────────────────────────────────────────────────────────────

# cl100k_base is the tokeniser used by both GPT-4o and text-embedding-3-small.
# Using the same tokeniser for chunking and embedding ensures our chunk size
# estimates are accurate — no surprises when the embedding API sees the text.
_tokeniser = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """
    Returns the number of tokens in a text string.

    Used throughout this module to measure chunk sizes accurately.
    Character count is a poor proxy for token count — a single Chinese
    character is one token, but "don't" is two tokens. Token counting
    is the only reliable measure.

    Args:
        text: Any string.

    Returns:
        Token count as an integer.
    """
    return len(_tokeniser.encode(text))


# ── Core chunking logic ────────────────────────────────────────────────────────

def chunk_text(
    text: str,
    chunk_size: int  = settings.chunk_size,
    chunk_overlap: int = settings.chunk_overlap,
) -> list[str]:
    """
    Splits a text string into overlapping chunks by token count.

    The overlap is what makes retrieval robust. Without it, a sentence
    like "revenue grew 23% year-over-year, driven primarily by Azure" that
    falls across a chunk boundary gets split — the first chunk has "revenue
    grew 23% year-over-year" and the second has "driven primarily by Azure".
    A query about Azure revenue growth might miss both chunks. With overlap,
    the full sentence appears in at least one chunk and is always retrievable.

    Strategy:
        1. Encode the full text to tokens once
        2. Slide a window of chunk_size tokens across the token list
        3. Step forward by (chunk_size - chunk_overlap) tokens each time
        4. Decode each window back to a string

    This token-level approach is more accurate than character-based splitting,
    which can create chunks that are either too large or too small for the
    embedding model depending on the text's character-to-token ratio.

    Args:
        text:          Text to split — should be clean prose, not raw HTML.
        chunk_size:    Target size of each chunk in tokens. Default from settings.
        chunk_overlap: Number of tokens to overlap between consecutive chunks.
                       Must be less than chunk_size — overlap larger than the
                       chunk itself makes no sense and this is validated below.

    Returns:
        List of text strings, each approximately chunk_size tokens long.
        The last chunk may be shorter if the text does not divide evenly.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be less than "
            f"chunk_size ({chunk_size}). Overlapping more than the "
            f"chunk size produces infinite loops."
        )

    if not text or not text.strip():
        return []

    # Encode once — decoding is cheap, encoding the same text repeatedly is not
    tokens = _tokeniser.encode(text)

    if len(tokens) <= chunk_size:
        # Text fits in a single chunk — no splitting needed
        return [text]

    chunks = []
    step   = chunk_size - chunk_overlap
    start  = 0

    while start < len(tokens):
        end        = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]

        # Decode back to string — errors="replace" handles the rare case
        # where a token boundary splits a multi-byte unicode character
        chunk_text_str = _tokeniser.decode(chunk_tokens)
        chunks.append(chunk_text_str)

        # If we just encoded the final tokens, we are done
        if end == len(tokens):
            break

        start += step

    return chunks


# ── Document-level chunking ────────────────────────────────────────────────────

def chunk_document(document: dict) -> list[dict]:
    """
    Converts a parsed document dict into a list of chunk dicts.

    This is the main entry point for the processing pipeline. It reads
    a document from document_store.py, chunks each section separately,
    and returns a flat list of chunk dicts ready for embedding.

    Chunking at the section level rather than the full document means:
    - Chunks stay semantically coherent — risk factor text does not bleed
      into financial statement text mid-chunk
    - Agents can filter chunks by section_name during retrieval
    - Smaller sections (Legal Proceedings) produce fewer, larger chunks
      rather than being padded with content from adjacent sections

    Args:
        document: Parsed document dict from document_store.load_processed_document().
                  Must contain: ticker, form_type, filing_date, accession_number,
                  sections (dict), full_text (str).

    Returns:
        List of chunk dicts, each containing all the metadata needed for
        the vector store and for agents to trace a chunk back to its source.
    """
    ticker           = document["ticker"]
    form_type        = document["form_type"]
    filing_date      = document["filing_date"]
    accession_number = document["accession_number"]
    sections         = document.get("sections", {})

    all_chunks  = []
    chunk_index = 0   # global index across all sections in this document

    if sections:
        # Chunk each section independently — preserves semantic boundaries
        for section_name, section_text in sections.items():
            if not section_text or not section_text.strip():
                continue

            section_chunks = chunk_text(section_text)

            for position_in_section, chunk_str in enumerate(section_chunks):
                chunk = _build_chunk_dict(
                    text             = chunk_str,
                    ticker           = ticker,
                    form_type        = form_type,
                    filing_date      = filing_date,
                    accession_number = accession_number,
                    section_name     = section_name,
                    chunk_index      = chunk_index,
                    position_in_section = position_in_section,
                    total_in_section    = len(section_chunks),
                )
                all_chunks.append(chunk)
                chunk_index += 1

    else:
        # No sections identified — fall back to chunking the full text.
        # This happens for Form 4 filings and some short 8-Ks.
        full_text = document.get("full_text", "")

        if full_text.strip():
            text_chunks = chunk_text(full_text)

            for position, chunk_str in enumerate(text_chunks):
                chunk = _build_chunk_dict(
                    text             = chunk_str,
                    ticker           = ticker,
                    form_type        = form_type,
                    filing_date      = filing_date,
                    accession_number = accession_number,
                    section_name     = "full_document",
                    chunk_index      = chunk_index,
                    position_in_section = position,
                    total_in_section    = len(text_chunks),
                )
                all_chunks.append(chunk)
                chunk_index += 1

    logger.debug(
        f"Chunked {ticker} {form_type} {filing_date}: "
        f"{len(all_chunks)} chunks from {len(sections)} sections"
    )

    return all_chunks


def _build_chunk_dict(
    text: str,
    ticker: str,
    form_type: str,
    filing_date: str,
    accession_number: str,
    section_name: str,
    chunk_index: int,
    position_in_section: int,
    total_in_section: int,
) -> dict:
    """
    Constructs a single chunk dict with full metadata.

    Every chunk carries enough metadata that an agent receiving it can
    answer: what company, what filing, what section, when was it filed,
    and where does this chunk sit within its section. This provenance
    is what allows agents to cite their sources rather than hallucinating.

    Args:
        text:                The chunk text content.
        ticker:              Stock ticker.
        form_type:           SEC form type.
        filing_date:         Filing date string.
        accession_number:    EDGAR accession number.
        section_name:        Name of the section this chunk came from.
        chunk_index:         Global chunk index within this document.
        position_in_section: Position of this chunk within its section (0-indexed).
        total_in_section:    Total chunks in this section — helps agents know
                             if they are reading the start, middle, or end.

    Returns:
        Chunk dict ready for embedding and vector store insertion.
    """
    # chunk_id is the stable identifier for this chunk — used by the vector
    # store to avoid re-embedding chunks that have not changed between runs.
    chunk_id = (
        f"{ticker}_{form_type}_{filing_date}_"
        f"{accession_number}_{chunk_index}"
    ).replace("-", "_")

    return {
        # ── Identity ──────────────────────────────────────────────────────
        "chunk_id":          chunk_id,
        "chunk_index":       chunk_index,

        # ── Content ───────────────────────────────────────────────────────
        "text":              text,
        "token_count":       count_tokens(text),

        # ── Source provenance — agents cite these when presenting signals ──
        "ticker":            ticker,
        "form_type":         form_type,
        "filing_date":       filing_date,
        "accession_number":  accession_number,

        # ── Section context ───────────────────────────────────────────────
        "section_name":         section_name,
        "position_in_section":  position_in_section,
        "total_in_section":     total_in_section,

        # is_first / is_last help the temporal drift agent decide whether
        # it is reading the opening or closing language of a section —
        # executives tend to put the most carefully worded statements first.
        "is_first_in_section":  position_in_section == 0,
        "is_last_in_section":   position_in_section == total_in_section - 1,
    }


# ── Batch processing ───────────────────────────────────────────────────────────

def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunks a list of documents and returns all chunks as a flat list.

    Used by the processing pipeline to batch-chunk an entire ticker's
    document set in one call.

    Args:
        documents: List of parsed document dicts from document_store.

    Returns:
        Flat list of all chunk dicts across all input documents.
    """
    all_chunks = []
    total_docs = len(documents)

    for i, document in enumerate(documents, 1):
        ticker      = document.get("ticker", "?")
        form_type   = document.get("form_type", "?")
        filing_date = document.get("filing_date", "?")

        logger.info(
            f"Chunking [{i}/{total_docs}] "
            f"{ticker} {form_type} {filing_date}"
        )

        doc_chunks = chunk_document(document)
        all_chunks.extend(doc_chunks)

    logger.info(
        f"Chunking complete — {len(all_chunks)} total chunks "
        f"from {total_docs} documents"
    )

    return all_chunks


def chunk_documents_lazy(documents: list[dict]) -> Generator[dict, None, None]:
    """
    Generator version of chunk_documents — yields chunks one at a time.

    Use this instead of chunk_documents() when processing a large corpus
    and you want to embed and store chunks as they are produced rather
    than holding the entire chunk list in memory.

    For our 121-document corpus this does not matter much, but at 500+
    documents the memory difference becomes meaningful.

    Args:
        documents: List of parsed document dicts.

    Yields:
        Individual chunk dicts, one at a time.
    """
    for document in documents:
        doc_chunks = chunk_document(document)
        yield from doc_chunks