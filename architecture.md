# MOSAIC — Architecture Documentation

## Overview

MOSAIC is built on three foundational principles:

1. **Separation of concerns** — ingestion, processing, agents, and infrastructure are completely independent layers
2. **Local-first development** — everything runs locally before being lifted to AWS unchanged
3. **Human in the loop** — no high-stakes signal ever auto-publishes without human approval

---

## System Layers

### Layer 1 — Ingestion

**Files:** `ingestion/`

The ingestion layer talks to SEC EDGAR and nothing else. It has no knowledge of agents, embeddings, or the vector store. Its only job is to produce clean structured JSON documents on disk.

```
edgar_client.py      → EDGAR API calls, rate limiting, retry logic
document_parser.py   → HTML cleaning, section extraction
document_store.py    → Save/load, idempotent ingestion
run_ingestion.py     → Entry point, orchestrates the three above
```

**Key design decision:** `document_exists()` in `document_store.py` makes ingestion idempotent. Re-running never duplicates work. Crashes mid-run are safe to recover from.

---

### Layer 2 — Processing

**Files:** `processing/`

The processing layer converts documents into a searchable vector index. It has no knowledge of agents or the ingestion source — it reads whatever JSON is in `data/processed/`.

```
chunker.py          → Token-level sliding window, section-aware
embedder.py         → OpenAI API batching, in-memory cache
vector_store.py     → FAISS IndexFlatL2, save/load, filtered search
run_processing.py   → Entry point, chains all three
```

**Key design decision:** Chunking happens at the section level, not document level. A 10-K Risk Factors section never bleeds into a Financial Statements chunk. Agents query sections, not documents.

---

### Layer 3 — Tools

**Files:** `tools/`

LangGraph-compatible tool wrappers. Agents call these tools during their reasoning loop rather than importing functions directly. This gives us automatic LangSmith tracing on every tool call.

```
vector_search_tools.py   → search_filings, search_temporal, search_across_companies
edgar_tools.py           → get_filing_list, get_financial_facts, get_insider_transactions
yfinance_tools.py        → get_earnings_history, get_stock_price_history, get_price_on_date
```

---

### Layer 4 — Agents

**Files:** `agents/`

Each agent is a LangGraph node — a Python function that reads from `MosaicState` and returns a partial state update. Agents never call each other directly. They communicate through the shared state.

Every agent follows the same pattern:

1. Load vector store if not ready
2. Search for relevant chunks
3. Call GPT-4o with structured prompt
4. Parse JSON response
5. Call `build_signal()` if signal detected
6. Return partial state update

---

### Layer 5 — Graph

**Files:** `graph/`

```
state.py          → MosaicState TypedDict, SignalDict, build_signal(), create_initial_state()
graph_builder.py  → StateGraph definition, node wiring, run_mosaic() entry point
hitl.py           → Interactive review, audit logging, rejection pattern analysis
```

**Key design decision:** `operator.add` annotation on list fields in `MosaicState` means agents append to shared lists rather than overwriting them. Five agents writing signals simultaneously produces one merged list, not five separate lists.

---

### Layer 6 — Evaluation

**Files:** `evaluation/`

```
scorers.py      → 4 scoring functions + composite scorer
eval_runner.py  → 5 test cases, regression checking, LangSmith experiment logging
```

Scorers are pure functions — they take agent output and return a score. No LLM calls. No side effects. Fast and deterministic.

---

### Layer 7 — AWS

**Files:** `aws/`

```
infrastructure.py   → CDK stack defining all resources
lambda_handlers.py  → Thin wrappers around local pipeline functions
deploy.sh           → End-to-end deployment in one command
```

**Key design decision:** Lambda handlers do only AWS plumbing. Business logic stays in the main codebase. The gap between local and production is minimal.

---

## State Architecture

`MosaicState` is a TypedDict that every agent reads from and writes to. The critical fields:

```python
signals:          Annotated[list[SignalDict], operator.add]   # accumulates
retrieved_chunks: Annotated[list[dict], operator.add]          # accumulates
approved_signals: Annotated[list[SignalDict], operator.add]    # accumulates
rejected_signals: Annotated[list[SignalDict], operator.add]    # accumulates

hitl_required:    bool          # overwritten
final_brief:      Optional[str] # overwritten
run_complete:     bool          # overwritten
```

List fields use `operator.add` — LangGraph merges rather than replaces on partial state updates. Scalar fields are last-write-wins.

---

## Signal Architecture

Every signal produced by any agent conforms to `SignalDict`:

```python
signal_id        # TICKER_TYPE_UUID — unique, traceable
signal_type      # temporal_drift | contradiction | credibility | supply_chain | insider_alignment
ticker           # primary company
headline         # one sentence — shown in HITL review
detail           # full explanation with evidence
confidence_score # 0.0-1.0 — drives HITL routing
severity         # low | medium | high | critical
requires_hitl    # auto-set by build_signal() based on thresholds
evidence         # list of source document excerpts
```

HITL is automatically triggered when:

* `confidence_score < 0.75` (settings.hitl_confidence_threshold)
* `severity == "critical"`
* `signal_type == "insider_alignment"` (always, regardless of confidence)

---

## HITL Architecture

The HITL gate is deliberately strict:

| Condition                                           | Routing                                        |
| --------------------------------------------------- | ---------------------------------------------- |
| confidence >= 0.75 AND not critical AND not insider | Auto-approved                                  |
| confidence < 0.75                                   | HITL queue                                     |
| severity = critical                                 | HITL queue (always)                            |
| signal_type = insider_alignment                     | HITL queue (always, confidence capped at 0.70) |

Every review decision — approve, reject, edit, escalate — is written to the audit log permanently. Rejection reasons feed back to prompt improvement.

---

## Cost Architecture

### Local Development

* OpenAI API: ~$0.30 for full corpus embedding (one-time)
* OpenAI API: ~$0.50-1.00 per full pipeline run (6 tickers, all agents)
* LangSmith: Free tier sufficient

### AWS Production

* S3: ~$2-5/month
* Lambda: ~$1-3/month
* EventBridge: ~$0
* RDS t3.micro: ~$15-25/month
* DynamoDB: ~$0
* Secrets Manager: ~$2/month
* CloudWatch: ~$3-8/month
* **Total: ~$25-45/month**
