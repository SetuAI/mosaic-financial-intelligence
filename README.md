<div align="center">


# 🧩 MOSAIC

### Multi-Agent Orchestration System for Analyst Intelligence and Cognition

[![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Latest-FF6B6B?style=for-the-badge&logo=chainlink&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![GPT-4o](https://img.shields.io/badge/GPT--4o-OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white)](https://openai.com/)
[![LangSmith](https://img.shields.io/badge/LangSmith-Tracing-F5A623?style=for-the-badge&logo=chainlink&logoColor=white)](https://smith.langchain.com/)
[![AWS](https://img.shields.io/badge/AWS-Deployed-FF9900?style=for-the-badge&logo=amazonaws&logoColor=white)](https://aws.amazon.com/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_Store-00A98F?style=for-the-badge&logo=meta&logoColor=white)](https://faiss.ai/)
[![SEC EDGAR](https://img.shields.io/badge/SEC_EDGAR-Free_API-003366?style=for-the-badge&logoColor=white)](https://www.sec.gov/developer)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](https://claude.ai/chat/LICENSE)
[![Stars](https://img.shields.io/github/stars/SetuAI/mosaic-financial-intelligence?style=for-the-badge&color=yellow)](https://github.com/SetuAI/mosaic-financial-intelligence/stargazers)

<br/>
>
> **Reads between the lines. Across every filing. Across every quarter.**
>

A production-grade multi-agent AI system that ingests SEC filings, detects signals invisible to any single analyst, and composes professional intelligence briefs — in under 5 minutes.

<br/>
[View Demo](https://claude.ai/chat/21022376-4157-4eb4-8fc9-9fde56ee4eac#-demo) • [Quick Start](https://claude.ai/chat/21022376-4157-4eb4-8fc9-9fde56ee4eac#-getting-started) • [Architecture](https://claude.ai/chat/21022376-4157-4eb4-8fc9-9fde56ee4eac#-architecture) • [Docs](https://claude.ai/chat/docs/) • [Report Bug](https://github.com/SetuAI/mosaic-financial-intelligence/issues)

</div>
---


---

## 📋 Table of Contents

* [📖 Project Description](https://claude.ai/chat/21022376-4157-4eb4-8fc9-9fde56ee4eac#-project-description)
* [🎬 Demo](https://claude.ai/chat/21022376-4157-4eb4-8fc9-9fde56ee4eac#-demo)
* [✨ Features](https://claude.ai/chat/21022376-4157-4eb4-8fc9-9fde56ee4eac#-features)
* [🛠 Tech Stack](https://claude.ai/chat/21022376-4157-4eb4-8fc9-9fde56ee4eac#-tech-stack)
* [🏗 Architecture](https://claude.ai/chat/21022376-4157-4eb4-8fc9-9fde56ee4eac#-architecture)
* [📁 Project Structure](https://claude.ai/chat/21022376-4157-4eb4-8fc9-9fde56ee4eac#-project-structure)
* [🚀 Getting Started](https://claude.ai/chat/21022376-4157-4eb4-8fc9-9fde56ee4eac#-getting-started)
* [🔐 Environment Variables](https://claude.ai/chat/21022376-4157-4eb4-8fc9-9fde56ee4eac#-environment-variables)
* [💻 Usage](https://claude.ai/chat/21022376-4157-4eb4-8fc9-9fde56ee4eac#-usage)
* [📡 API Documentation](https://claude.ai/chat/21022376-4157-4eb4-8fc9-9fde56ee4eac#-api-documentation)
* [🔄 Workflows](https://claude.ai/chat/21022376-4157-4eb4-8fc9-9fde56ee4eac#-workflows)
* [🧪 Testing](https://claude.ai/chat/21022376-4157-4eb4-8fc9-9fde56ee4eac#-testing)
* [☁️ Deployment](https://claude.ai/chat/21022376-4157-4eb4-8fc9-9fde56ee4eac#%EF%B8%8F-deployment)
* [🔧 Troubleshooting](https://claude.ai/chat/21022376-4157-4eb4-8fc9-9fde56ee4eac#-troubleshooting)
* [🗺 Roadmap](https://claude.ai/chat/21022376-4157-4eb4-8fc9-9fde56ee4eac#-roadmap)
* [❓ FAQ](https://claude.ai/chat/21022376-4157-4eb4-8fc9-9fde56ee4eac#-faq)
* [🤝 Contributing](https://claude.ai/chat/21022376-4157-4eb4-8fc9-9fde56ee4eac#-contributing)
* [📄 License](https://claude.ai/chat/21022376-4157-4eb4-8fc9-9fde56ee4eac#-license)
* [🙏 Acknowledgements](https://claude.ai/chat/21022376-4157-4eb4-8fc9-9fde56ee4eac#-acknowledgements)
* [📬 Contact](https://claude.ai/chat/21022376-4157-4eb4-8fc9-9fde56ee4eac#-contact)

---

## 📖 Project Description

MOSAIC is a **production-grade multi-agent financial intelligence system** that does what no single analyst — and no single LLM prompt — can do.

Every quarter, companies file earnings call transcripts, annual reports, quarterly filings, and insider transaction records with the SEC. Individually, each document is a needle in a haystack. Together, across time and across companies, they tell a story that experienced analysts spend weeks trying to read.

**MOSAIC reads that story in 5 minutes.**

### The Core Problem

| Challenge                                              | Why It Is Hard                                             |
| ------------------------------------------------------ | ---------------------------------------------------------- |
| A single LLM has no memory of prior quarters           | Cannot detect language drift over time                     |
| No analyst can hold 40 companies in their head         | Cannot propagate supply chain signals across companies     |
| Earnings calls and 10-K filings are separate documents | Cannot detect contradictions between them at scale         |
| Guidance is scattered across 8 quarters of calls       | Cannot score management credibility systematically         |
| Form 4 filings and call transcripts are unconnected    | Cannot cross-reference insider transactions with sentiment |

**MOSAIC solves all five problems simultaneously.**

### Coverage Universe

| Ticker | Company   | Filings Ingested        |
| ------ | --------- | ----------------------- |
| AMZN   | Amazon    | 10-K, 10-Q, 8-K, Form 4 |
| MSFT   | Microsoft | 10-K, 10-Q, 8-K, Form 4 |
| GOOG   | Alphabet  | 10-K, 10-Q, 8-K, Form 4 |
| ORCL   | Oracle    | 10-K, 10-Q, 8-K, Form 4 |
| AAPL   | Apple     | 10-K, 10-Q, 8-K, Form 4 |
| NVDA   | NVIDIA    | 10-K, 10-Q, 8-K, Form 4 |

**121 documents. 5,617 indexed passages. 2 years of history. Updated daily.**

---

## 🎬 Demo

<div align="center">
  <img src="assets/mosaic_demo.gif" alt="MOSAIC Demo — Full Pipeline Run" width="90%"/>
  <br/><br/>
  <em>Full pipeline run — MSFT + NVDA — 10 signals generated, 3 HITL items, intelligence brief composed in under 5 minutes</em>
</div>
<br/>
### Sample Output

```
MOSAIC INTELLIGENCE BRIEF — Run 7152b2ef
============================================================

EXECUTIVE SUMMARY
The overall risk posture for Microsoft (MSFT) and NVIDIA (NVDA) is assessed
as medium, with notable concerns around insider selling and supply chain
constraints. Both companies exhibit increased caution in their communications.

MSFT — Risk: MEDIUM
A critical contradiction was detected between revenue growth guidance on an
earnings call and figures reported in the 10-K, warranting further investigation.
High severity insider selling activity was observed following a bullish earnings
call, raising questions about executive confidence.

NVDA — Risk: MEDIUM
NVIDIA is experiencing high severity supply chain constraints, including extended
lead times exceeding 12 months. Geopolitical risks from AI Diffusion IFR add
further complexity.

PENDING HUMAN REVIEW (3 items)
  [URGENT]   MSFT — Revenue guidance diverges from 10-K reported figures
  [ELEVATED] MSFT — Cluster selling after bullish earnings call
  [ELEVATED] NVDA — Cluster selling after bullish earnings call
```

---

## ✨ Features

### 🤖 5 Specialist AI Agents

| Agent                    | What It Finds                                                                                    |
| ------------------------ | ------------------------------------------------------------------------------------------------ |
| **Temporal Drift** | Language getting more cautious across 8+ quarters — 2-3 quarters before bad news is confirmed   |
| **Contradiction**  | What management said on the call vs what they filed legally — the gap is where signals live     |
| **Credibility**    | Every forward guidance statement tracked against actuals — score 0-100 with full evidence trail |
| **Supply Chain**   | Upstream stress propagated to all downstream companies simultaneously                            |
| **Insider Signal** | Cluster selling after bullish calls — always routed to human review                             |

### 🛡 Human-in-the-Loop (HITL) Gate

* Every high-stakes signal reviewed by a human before publishing
* Automatic routing based on confidence score and severity
* Interactive CLI review with approve / reject / edit / escalate
* Permanent audit log with feedback loop to improve agents

### 🔍 Semantic Search Over 5,617 Passages

* Chunk-level semantic search via FAISS + OpenAI embeddings
* Section-aware chunking — risk factors never bleed into financial statements
* Filtered search by ticker, form type, and section name
* Millisecond retrieval across 2 years of filing history

### 📊 Built-in Evaluation Framework

* 5 eval test cases — one per agent
* 4 scoring dimensions: completeness, evidence quality, confidence calibration, HITL routing
* Regression detection — catches prompt degradation before it reaches production
* LangSmith experiment logging with run comparison

### ☁️ Production AWS Architecture

* Fully automated daily ingestion via EventBridge + Lambda
* pgvector on RDS for concurrent-safe semantic search
* Secrets Manager for all credentials — nothing hardcoded
* CloudWatch alarms for pipeline failures
* One-command deployment with CDK

---

## 🛠 Tech Stack

### Core AI

| Component       | Technology                      |
| --------------- | ------------------------------- |
| LLM             | GPT-4o (OpenAI)                 |
| Agent Framework | LangGraph                       |
| Embeddings      | text-embedding-3-small (OpenAI) |
| Observability   | LangSmith                       |

### Data & Storage

| Component                 | Technology                   |
| ------------------------- | ---------------------------- |
| Vector Store (local)      | FAISS                        |
| Vector Store (production) | pgvector on RDS PostgreSQL   |
| Document Storage          | Local filesystem / Amazon S3 |
| Audit Log                 | JSONL / Amazon DynamoDB      |

### Data Sources

| Source            | Data                       | Cost            |
| ----------------- | -------------------------- | --------------- |
| SEC EDGAR API     | 10-K, 10-Q, 8-K, Form 4    | Free            |
| yFinance          | Price history, EPS history | Free            |
| OpenAI Embeddings | Vector representations     | ~$0.30 one-time |

### Infrastructure (AWS)

| Service         | Role                    |
| --------------- | ----------------------- |
| S3              | Document store          |
| Lambda          | Serverless compute      |
| EventBridge     | Daily scheduling        |
| RDS + pgvector  | Production vector store |
| DynamoDB        | HITL audit log          |
| Secrets Manager | API key storage         |
| CloudWatch      | Monitoring and alerts   |
| CDK             | Infrastructure as code  |

### Application

| Component       | Technology                   |
| --------------- | ---------------------------- |
| Language        | Python 3.13                  |
| Package Manager | uv                           |
| HTTP Client     | httpx                        |
| HTML Parser     | BeautifulSoup4 + lxml        |
| Tokeniser       | tiktoken                     |
| Config          | Pydantic + pydantic-settings |

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      MOSAIC PIPELINE                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  SEC EDGAR API ──► Ingestion Layer ──► Processing Layer     │
│                    (121 documents)     (5,617 vectors)      │
│                                              │              │
│                                    ┌─────────▼──────────┐  │
│                                    │  LangGraph Graph   │  │
│                                    │                    │  │
│                                    │  ┌──────────────┐  │  │
│                                    │  │Temporal Drift│  │  │
│                                    │  │Contradiction │  │  │
│                                    │  │Credibility   │  │  │
│                                    │  │Supply Chain  │  │  │
│                                    │  │Insider Signal│  │  │
│                                    │  └──────┬───────┘  │  │
│                                    │         │          │  │
│                                    │    Supervisor      │  │
│                                    └─────────┬──────────┘  │
│                                              │              │
│                          ┌───────────────────┼───────────┐  │
│                          ▼                   ▼           │  │
│                    Auto-Approved          HITL Queue     │  │
│                    (7 signals)            (3 signals)    │  │
│                          │                   │           │  │
│                          └───────────────────┘           │  │
│                                    │                      │  │
│                          Intelligence Brief               │  │
└─────────────────────────────────────────────────────────────┘
```

See [`docs/ARCHITECTURE.md`](https://claude.ai/chat/docs/ARCHITECTURE.md) for the full technical deep-dive.

---

## 📁 Project Structure

```
mosaic-financial-intelligence/
│
├── 📂 config/
│   ├── settings.py              # All env vars via pydantic-settings
│   └── logging_config.py        # Centralised logging
│
├── 📂 ingestion/
│   ├── edgar_client.py          # EDGAR API — fetch, rate limit, retry
│   ├── document_parser.py       # HTML → clean structured text
│   ├── document_store.py        # Save/load, idempotent ingestion
│   └── run_ingestion.py         # Entry point
│
├── 📂 processing/
│   ├── chunker.py               # Token-level section-aware chunking
│   ├── embedder.py              # OpenAI embeddings with cache
│   ├── vector_store.py          # FAISS index, save/load, search
│   └── run_processing.py        # Entry point
│
├── 📂 agents/
│   ├── temporal_drift_agent.py  # Language shift across quarters
│   ├── contradiction_agent.py   # 10-K vs earnings call diff
│   ├── credibility_agent.py     # Promise vs delivery scoring
│   ├── supply_chain_agent.py    # Cross-company propagation
│   ├── insider_signal_agent.py  # Form 4 vs sentiment
│   └── supervisor.py            # Synthesis + brief composition
│
├── 📂 tools/
│   ├── vector_search_tools.py   # Semantic search LangGraph tools
│   ├── edgar_tools.py           # EDGAR API LangGraph tools
│   └── yfinance_tools.py        # Market data LangGraph tools
│
├── 📂 graph/
│   ├── state.py                 # MosaicState TypedDict + build_signal()
│   ├── graph_builder.py         # Graph wiring + run_mosaic()
│   └── hitl.py                  # Human review + audit log
│
├── 📂 evaluation/
│   ├── scorers.py               # 4 quality scoring functions
│   └── eval_runner.py           # 5-case test suite
│
├── 📂 aws/
│   ├── infrastructure.py        # CDK stack — all AWS resources
│   ├── lambda_handlers.py       # Lambda function handlers
│   └── deploy.sh                # One-command deployment
│
├── 📂 docs/                     # Technical documentation
│   ├── ARCHITECTURE.md
│   ├── SETUP.md
│   └── AGENTS.md
│
├── 📂 workflows/                # Mermaid workflow diagrams
│   └── WORKFLOWS.md
│
├── 📂 assets/                   # Demo GIF and images
│   └── mosaic_demo.gif
│
├── 📂 data/                     # Local data (gitignored)
│   ├── raw/                     # Raw EDGAR HTML filings
│   ├── processed/               # Parsed JSON documents
│   └── vector_store/            # FAISS index files
│
├── main.py                      # Local CLI control panel
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

* Python 3.13+
* OpenAI API key — [platform.openai.com](https://platform.openai.com/api-keys)
* LangSmith API key — [smith.langchain.com](https://smith.langchain.com/) (free)
* SEC EDGAR user agent — your name + email (no registration)

### Installation

**1. Clone the repository**

```bash
git clone https://github.com/SetuAI/mosaic-financial-intelligence.git
cd mosaic-financial-intelligence
```

**2. Install uv**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**3. Create virtual environment**

```bash
uv venv
source .venv/bin/activate     # Mac/Linux
# .venv\Scripts\activate      # Windows
```

**4. Install dependencies**

```bash
uv pip install -r requirements.txt
```

**5. Configure environment**

```bash
cp .env.example .env
# edit .env with your keys — see Environment Variables below
```

**6. Verify setup**

```bash
python main.py status
```

**7. Ingest documents**

```bash
python main.py ingest
# takes 8-12 minutes, downloads 121 SEC filings
```

**8. Build vector store**

```bash
python main.py process
# chunks, embeds, and indexes all documents — ~$0.30
```

**9. Run the pipeline**

```bash
python main.py pipeline --tickers MSFT NVDA
```

---

## 🔐 Environment Variables

Copy `.env.example` to `.env` and fill in the following:

```bash
# ── LLM ──────────────────────────────────────────────
OPENAI_API_KEY=sk-proj-...          # Required — platform.openai.com
OPENAI_MODEL=gpt-4o                 # Default
OPENAI_MAX_TOKENS=4096              # Default
OPENAI_TEMPERATURE=0.0              # Default — deterministic outputs

# ── LangSmith Tracing ─────────────────────────────────
LANGCHAIN_API_KEY=ls__...           # Required — smith.langchain.com
LANGCHAIN_PROJECT=mosaic-prod       # Default
LANGCHAIN_TRACING_V2=true           # Default

# ── SEC EDGAR ─────────────────────────────────────────
SEC_USER_AGENT=Jane Smith jane@example.com   # Required — your name + email

# ── Data Storage ──────────────────────────────────────
RAW_DATA_PATH=data/raw              # Default
PROCESSED_DATA_PATH=data/processed  # Default

# ── Vector Store ──────────────────────────────────────
EMBEDDING_MODEL=text-embedding-3-small    # Default
EMBEDDING_DIMENSION=1536                  # Default
CHUNK_SIZE=1000                           # Default
CHUNK_OVERLAP=200                         # Default

# ── HITL ──────────────────────────────────────────────
HITL_CONFIDENCE_THRESHOLD=0.75      # Signals below this go to human review
```

> **Note:** Never commit `.env` to git. It is already in `.gitignore`.

---

## 💻 Usage

### Command Reference

```bash
# Check system status
python main.py status

# Ingest SEC filings from EDGAR
python main.py ingest
python main.py ingest --ticker MSFT --form-type 10-K

# Build the vector store
python main.py process
python main.py process --ticker MSFT

# Run the full agent pipeline
python main.py pipeline --tickers MSFT NVDA
python main.py pipeline --tickers MSFT --query "Supply chain risk analysis"
python main.py pipeline                           # all 6 tickers

# Review HITL signals interactively
python main.py hitl --reviewer "Your Name"
python main.py hitl --audit                       # view audit log

# Run evaluation suite
python main.py eval --no-langsmith
python main.py eval --case msft_contradiction     # single test case
```

### Pipeline Output

After running `python main.py pipeline`, you will see:

* **Run statistics** — total signals, auto-approved, HITL queue size
* **All signals** — type, severity, confidence, headline for each
* **HITL queue** — signals awaiting human review sorted by urgency
* **Intelligence brief** — professional analyst document

The HITL queue is saved to `data/hitl_queue_latest.json` automatically for review.

---

## 📡 API Documentation

### `run_mosaic()` — Main Entry Point

```python
from graph.graph_builder import run_mosaic

result = run_mosaic(
    query          = "Analyse MSFT and NVDA for risk signals",
    tickers        = ["MSFT", "NVDA"],
    primary_ticker = "MSFT",
    triggered_by   = "analyst_request",  # or "lambda_invocation"
)

# Access results
print(result["final_brief"])
print(f"Signals: {len(result['signals'])}")
print(f"HITL queue: {len(result['hitl_queue'])}")
print(f"Overall risk: {result['run_summary']['overall_risk']}")
```

### `SignalDict` — Signal Structure

```python
{
    "signal_id":        "MSFT_temporal_drift_abc123",
    "signal_type":      "temporal_drift",
    "ticker":           "MSFT",
    "generated_by":     "temporal_drift_agent",
    "headline":         "Hedge language frequency rising 3 consecutive quarters",
    "detail":           "Full explanation...",
    "evidence":         [...],
    "confidence_score": 0.82,
    "severity":         "medium",
    "requires_hitl":    False,
    "review_status":    "auto_approved"
}
```

### Vector Search Tools

```python
from tools.vector_search_tools import search_filings, search_temporal

# General semantic search
results = search_filings.invoke({
    "query":     "supply chain risk semiconductor",
    "ticker":    "NVDA",
    "form_type": "10-K",
    "k":         5,
})

# Chronological search for temporal analysis
results = search_temporal.invoke({
    "query":     "revenue guidance outlook",
    "ticker":    "MSFT",
    "form_type": "10-K",
    "k":         8,
})
```

For full API documentation see [`docs/ARCHITECTURE.md`](https://claude.ai/chat/docs/ARCHITECTURE.md).

---

## 🔄 Workflows

All system workflows are documented as Mermaid diagrams in [`workflows/WORKFLOWS.md`](https://claude.ai/chat/workflows/WORKFLOWS.md).

| Workflow                    | Description                                      |
| --------------------------- | ------------------------------------------------ |
| Full Pipeline Overview      | End-to-end flow from EDGAR to intelligence brief |
| Ingestion Pipeline          | How SEC filings are downloaded and parsed        |
| Processing Pipeline         | How documents are chunked, embedded, and indexed |
| LangGraph Agent Graph       | State diagram of the 6-node agent graph          |
| HITL Review Flow            | Signal routing and human review decision tree    |
| AWS Deployment Architecture | Complete cloud infrastructure diagram            |
| State Flow                  | How MosaicState evolves through the graph        |
| Signal Lifecycle            | Sequence diagram from generation to brief        |

---

## 🧪 Testing

### Run The Evaluation Suite

```bash
# all 5 test cases, no LangSmith
python main.py eval --no-langsmith

# with LangSmith tracing
python main.py eval

# single test case
python main.py eval --case msft_contradiction
```

### Current Baseline

```
✓ PASS | msft_temporal_drift   | signals=1 | overall=1.00
✓ PASS | msft_contradiction    | signals=1 | overall=0.93
✓ PASS | nvda_supply_chain     | signals=5 | overall=0.90
✓ PASS | msft_credibility      | signals=0 | overall=1.00
✓ PASS | insider_alignment     | signals=1 | overall=0.90
All evals passed — no regressions detected.
```

### Scoring Dimensions

| Dimension              | What It Measures                   | Weight |
| ---------------------- | ---------------------------------- | ------ |
| Completeness           | All required fields populated      | 25%    |
| Evidence Quality       | Grounded in actual document quotes | 35%    |
| Confidence Calibration | Score appropriate for the finding  | 20%    |
| HITL Appropriateness   | Correctly routed to human review   | 20%    |

### Individual Agent Tests

```bash
python test_temporal_agent.py
python test_contradiction_agent.py
python test_credibility_agent.py
python test_supply_chain_agent.py
python test_insider_agent.py
python test_full_pipeline.py
```

---

## ☁️ Deployment

### Prerequisites

```bash
# install AWS CLI
curl "https://awscli.amazonaws.com/AWSCLIV2.pkg" -o "AWSCLIV2.pkg"
sudo installer -pkg AWSCLIV2.pkg -target /

# configure credentials
aws configure

# install CDK
npm install -g aws-cdk
```

### Deploy

```bash
bash aws/deploy.sh
```

This provisions all AWS resources and syncs local data to S3 in a single command.

### What Gets Created

| Resource        | Purpose              | Est. Cost            |
| --------------- | -------------------- | -------------------- |
| S3 Bucket       | Document store       | ~$2-5/mo             |
| Lambda (×2)    | Ingestion + pipeline | ~$1-3/mo             |
| EventBridge     | Daily scheduling     | ~$0                  |
| RDS t3.micro    | pgvector database    | ~$15-25/mo           |
| DynamoDB        | HITL audit log       | ~$0                  |
| Secrets Manager | API key storage      | ~$2/mo               |
| CloudWatch      | Monitoring           | ~$3-8/mo             |
| **Total** |                      | **~$25-45/mo** |

### Trigger A Production Run

```bash
aws lambda invoke \
  --function-name mosaic-pipeline-runner \
  --payload '{"tickers":["MSFT","NVDA"],"query":"Full risk analysis"}' \
  --region us-east-1 \
  response.json && cat response.json
```

See [`docs/ARCHITECTURE.md`](https://claude.ai/chat/docs/ARCHITECTURE.md) for full deployment documentation.

---

## 🔧 Troubleshooting

### SSL Certificate Error

```bash
unset SSL_CERT_FILE
unset REQUESTS_CA_BUNDLE
```

This happens when a previous Python venv set `SSL_CERT_FILE` globally. Unsetting it fixes tiktoken and httpx.

### `ModuleNotFoundError: No module named 'ingestion'`

Always run from the project root:

```bash
cd mosaic-financial-intelligence
python main.py ...   # ✓ correct
```

Never from inside a subfolder.

### LangSmith 401 / 403 Error

```bash
# regenerate your key at smith.langchain.com
# then update .env
LANGCHAIN_API_KEY=ls__your_new_key
```

### Empty Vector Search Results

The vector store is not built or built with wrong path:

```bash
python main.py process
```

### EDGAR Rate Limiting (429)

The pipeline respects EDGAR's 10 req/s limit at 0.5s delay. If you hit 429:

```bash
# increase delay in .env
SEC_REQUEST_DELAY=1.0
```

### `pydantic ValidationError` on startup

A required field is missing from `.env`. Check:

```bash
python -c "from config.settings import settings; print('OK')"
```

The error message will tell you which field is missing.

---

## 🗺 Roadmap

* [ ] **Earnings call transcript scraping** — add FMP API integration for richer 8-K transcripts
* [ ] **Redis caching layer** — rate limit tracking, embedding cache, HITL distributed locks
* [ ] **Web dashboard** — React frontend for HITL review and signal browsing
* [ ] **Expanded coverage universe** — add META, AMD, TSM, INTC
* [ ] **Parallel agent execution** — run agents concurrently once rate limits allow
* [ ] **Email/Slack alerts** — notify analysts when HITL queue has urgent items
* [ ] **Historical backtesting** — validate signal quality against known market events
* [ ] **Custom coverage universe** — allow users to define their own ticker sets

---

## ❓ FAQ

**Q: Does this require any paid data subscriptions?**
A: No. All data comes from SEC EDGAR (free government API) and yFinance (free). The only costs are OpenAI API calls (~$0.30 one-time for embedding, ~$0.50-1.00 per pipeline run).

**Q: How often should I re-run the ingestion?**
A: Earnings are filed quarterly, but 8-Ks and Form 4s can appear any day. The AWS deployment runs automatically every weekday at 6 AM UTC. Locally, run `python main.py ingest` whenever you want to check for new filings.

**Q: Can I add my own tickers?**
A: Yes. Add the ticker and its SEC CIK number to `ticker_cik_map` in `config/settings.py`. Find CIK numbers at [EDGAR company search](https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany).

**Q: What does HITL mean?**
A: Human-in-the-Loop. High-stakes signals — critical severity, insider transactions, or low-confidence findings — are routed to a human analyst for review before publishing. This is what makes MOSAIC trustworthy enough for actual investment decisions.

**Q: Can I use a different LLM?**
A: The system is built for GPT-4o. Swapping to another model requires updating `settings.py` (openai_model) and potentially tuning the agent system prompts, as some rely on GPT-4o's instruction-following and JSON output quality.

**Q: Is this investment advice?**
A: No. MOSAIC surfaces signals for analyst review. It does not make buy/sell recommendations. All signals require human judgment before acting on them.

---

## 🤝 Contributing

Contributions are welcome. Please open an issue first to discuss what you would like to change.

**How to contribute:**

1. Fork the repository
2. Create your feature branch
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Commit your changes
   ```bash
   git commit -m 'Add: description of your change'
   ```
4. Push to the branch
   ```bash
   git push origin feature/your-feature-name
   ```
5. Open a Pull Request

**Good first issues:**

* Add a new agent (e.g. ESG signal detector)
* Add a new ticker to the coverage universe
* Improve section extraction in `document_parser.py` for edge cases
* Add Redis caching layer

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](https://claude.ai/chat/LICENSE) file for details.

---

## 🙏 Acknowledgements

* **[Krish Naik](https://www.youtube.com/@krishnaik06)** — for the YouTube platform and the opportunity to build this project
* **[LangChain / LangGraph](https://langchain-ai.github.io/langgraph/)** — for the multi-agent orchestration framework
* **[SEC EDGAR](https://www.sec.gov/developer)** — for providing free, open access to all public company filings
* **[OpenAI](https://openai.com/)** — for GPT-4o and the embedding models
* **[Meta FAISS](https://faiss.ai/)** — for the vector similarity search library

---

## 📬 Contact


**Project Link:** [https://github.com/SetuAI/mosaic-financial-intelligence](https://github.com/SetuAI/mosaic-financial-intelligence)

---

<div align="center">
⭐ **Star this repo if MOSAIC helped you understand production multi-agent AI**

Built with ❤️ for the Krish Naik YouTube community
