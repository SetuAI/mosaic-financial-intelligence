# MOSAIC — Agents Documentation

## Overview

MOSAIC runs 5 specialist agents sequentially, followed by a Supervisor that synthesises their outputs. Each agent is a LangGraph node that reads from and writes to the shared `MosaicState`.

---

## Agent 1 — Temporal Drift Agent

**File:** `agents/temporal_drift_agent.py`

### What It Detects

Language shift across 8+ consecutive filings. Management teams gradually get more cautious 2-3 quarters before bad news becomes undeniable. This agent catches that drift early.

### How It Works

1. Searches for management discussion chunks for the ticker using `search_temporal` — results sorted chronologically oldest to newest
2. Groups chunks by filing date into a timeline
3. Sends the timeline to GPT-4o with a structured analytical prompt
4. GPT-4o identifies drift direction, severity, and specific phrases that evidence the shift

### Signal Output

```python
{
    "signal_type": "temporal_drift",
    "drift_direction": "more_cautious",
    "severity": "medium",
    "confidence_score": 0.80,
    "key_phrases": ["may", "subject to", "we believe"],
    "earliest_signal_quarter": "2024-07-30"
}
```

### HITL Threshold

Auto-approved if confidence >= 0.75 and severity != critical.

---

## Agent 2 — Contradiction Agent

**File:** `agents/contradiction_agent.py`

### What It Detects

Gaps between what management said on the earnings call (8-K) versus what they filed legally in the 10-K. CEOs can be optimistic on calls. They cannot be inaccurate in SEC filings.

### How It Works

1. For each of 5 topics (revenue guidance, litigation, competitive position, operational risk, AI investment), fetches relevant passages from both 8-K and 10-K
2. Asks GPT-4o to compare the two documents per topic
3. Surfaces the most severe contradiction found across all topics

### Topics Compared

* `revenue_guidance` — what they said vs what they filed
* `litigation_legal` — legal risk disclosed on call vs in 10-K
* `competitive_position` — confident call vs defensive filings
* `operational_risk` — supply chain optimism vs risk factors
* `ai_investment` — bold AI claims vs conservative capex

### Signal Output

```python
{
    "signal_type": "contradiction",
    "contradiction_type": "factual_inconsistency",
    "severity": "critical",
    "call_excerpt": "...",
    "filing_excerpt": "..."
}
```

### HITL Threshold

Critical severity always triggers HITL. Other severities follow standard threshold.

---

## Agent 3 — Credibility Agent

**File:** `agents/credibility_agent.py`

### What It Detects

Management's track record of delivering on stated guidance. Extracts every forward guidance statement, cross-references against actual reported financials, and scores credibility 0-100.

### How It Works

1. Fetches earnings call guidance passages using `search_temporal`
2. Pulls actual reported financials from EDGAR via `get_financial_facts`
3. Fetches EPS beat/miss history from yFinance
4. Asks GPT-4o to extract guidance statements, match to actuals, and score

### Scoring Formula

```
score = (DELIVERED + 0.5 × PARTIAL) / total_verifiable × 100
```

### Credibility Tiers

| Score  | Tier     | Analyst Action               |
| ------ | -------- | ---------------------------- |
| 80-100 | High     | Model guidance at face value |
| 60-79  | Moderate | Minor haircut on guidance    |
| 40-59  | Low      | Significant haircut          |
| 0-39   | Very Low | Guidance unreliable          |

### Signal Output

Only generated for scores below 80. High credibility is noted in the brief but does not generate a risk signal.

---

## Agent 4 — Supply Chain Agent

**File:** `agents/supply_chain_agent.py`

### What It Detects

Supply chain stress originating in any company in the coverage universe, and propagates that signal to all dependent downstream companies.

### How It Works

1. Searches all 6 tickers simultaneously using `search_across_companies` for 5 supply chain topics
2. For each topic, identifies which company shows the strongest stress signal
3. Asks GPT-4o to assess which other companies are exposed and how severely

### Topics Searched

* Semiconductor supply constraints
* Data centre infrastructure lead times
* Manufacturing and logistics disruptions
* Geopolitical trade restrictions
* Input cost and margin pressure

### Why It Is Unique

Every other agent analyses one ticker at a time. This agent searches all 6 companies simultaneously — the supply chain does not respect company boundaries.

---

## Agent 5 — Insider Signal Agent

**File:** `agents/insider_signal_agent.py`

### What It Detects

Meaningful divergence between what executives say publicly on earnings calls and what they do privately with their own company shares.

### How It Works

1. Fetches bullish and cautious language from earnings calls
2. Fetches Form 4 transaction chunks from the vector store
3. Gets Form 4 filing metadata (dates, volume) from EDGAR
4. Asks GPT-4o to identify divergence patterns

### Signal Patterns

* **High significance:** Large sales (>$10M) within 30 days of bullish call
* **High significance:** Cluster selling — 3+ executives in same 2-week window
* **Medium significance:** Repeated pattern of selling before guidance reductions
* **Mitigating:** 10b5-1 pre-scheduled plan (noted but cannot confirm from Form 4)

### Important: Always HITL

Insider signal agents always set `requires_hitl=True` and cap confidence at 0.70. Insider trading implications are legally serious — no insider signal ever auto-publishes.

---

## Supervisor Agent

**File:** `agents/supervisor.py`

### What It Does

Reads all signals from state after every other agent has run. Builds the HITL review queue sorted by urgency. Asks GPT-4o to compose a professional intelligence brief.

### Brief Structure

1. Executive summary — overall risk posture
2. Per-company sections — findings, positive indicators, recommended actions
3. Cross-company themes — patterns across multiple companies
4. HITL review items — signals pending human approval
5. Data limitations — honest acknowledgment of gaps

### Output

The brief is written as professional analyst prose — not a list of AI outputs. The Supervisor prompt explicitly instructs GPT-4o to write as a senior analyst, not as an AI system.
