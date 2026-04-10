# Pharma Pricing Copilot 💊

An AI-powered pharmaceutical pricing analytics platform that combines **ensemble anomaly detection**, **Retrieval-Augmented Generation (RAG)** over regulatory documents, and a **Claude-powered conversational agent** — all wrapped in an interactive Streamlit dashboard.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.36+-red.svg)](https://streamlit.io)
[![Claude](https://img.shields.io/badge/powered%20by-Claude%20Haiku-blueviolet.svg)](https://anthropic.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Features

| Feature | Description |
|---|---|
| **Synthetic Data Simulator** | Generates 15,000 realistic transactions across 50 drugs, 200 pharmacies, 5 drug classes with WAC, ASP, AMP, 340B, GTN, NADAC benchmarks |
| **Ensemble Anomaly Detection** | 6-method ensemble: Z-Score, IQR Fence, Month-over-Month, Isolation Forest, LOF, and Regulatory Rules |
| **Regulatory RAG** | Local vector store (sentence-transformers + numpy) with 60 synthetic pharma pricing knowledge documents |
| **AI Pricing Copilot** | Claude Haiku agent with 6 built-in tools, streaming responses, and multi-turn conversation memory |
| **Streamlit Dashboard** | 4-tab interactive UI with 10+ Plotly charts across Dashboard, Anomaly Explorer, Regulatory Q&A, and AI Copilot |

---

## Live Demo

```bash
streamlit run app/streamlit_app.py
```

Navigate to **http://localhost:8501**, click **Generate & Load Data** in the sidebar, then explore all four tabs.

---

## Project Structure

```
pharma-pricing-copilot/
├── app/
│   └── streamlit_app.py          # 4-tab Streamlit front-end
├── data/
│   ├── raw/                      # Generated pricing CSV (gitignored)
│   ├── processed/                # SQLite DB + anomaly results (gitignored)
│   └── vectorstore/              # Embedding index (gitignored)
├── notebooks/
│   └── eda.ipynb                 # Exploratory Data Analysis
├── src/
│   ├── __init__.py
│   ├── data_simulator.py         # PricingDataSimulator — 15K synthetic rows
│   ├── anomaly_detection.py      # AnomalyDetectionEngine — 6-method ensemble
│   ├── rag_pipeline.py           # TextChunker + VectorStore + RAGPipeline
│   ├── agent.py                  # PharmaPricingAgent — Claude tool-use agent
│   └── utils.py                  # Logging, formatting, I/O helpers
├── tests/
│   └── test_agent.py             # Unit + integration tests
├── .env.example                  # Environment variable template
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/anilmedads-hub/pharma-pricing-copilot.git
cd pharma-pricing-copilot
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set your Anthropic API key

```bash
cp .env.example .env
# Edit .env and paste your ANTHROPIC_API_KEY
```

Or enter it directly in the Streamlit sidebar.

### 3. Launch the app

```bash
streamlit run app/streamlit_app.py
```

Click **Generate & Load Data** in the sidebar — the full pipeline runs automatically (data generation → anomaly detection → dashboard renders).

### 4. Run tests

```bash
pytest tests/ -v
# With coverage:
pytest tests/ --cov=src --cov-report=term-missing
```

---

## Module Guide

### `src/data_simulator.py` — `PricingDataSimulator`

Generates realistic pharmaceutical pricing transactions:

```python
from src.data_simulator import PricingDataSimulator

sim = PricingDataSimulator(seed=42, n_rows=15_000, anomaly_rate=0.07)
df = sim.generate()          # → 15K-row DataFrame + saves CSV
sim.build_sqlite_db()        # → data/processed/pricing.db
```

**Drug catalog**: 50 drugs across 5 classes (Cardiovascular, Respiratory, Oncology, Immunology, Diabetes)
**Pharmacy network**: 200 pharmacies across 6 chains (CVS, Walgreens, Rite Aid, Walmart, Kroger, Independent)
**Price benchmarks**: WAC → ASP → AMP → 340B → GTN → NADAC with realistic correlations
**Anomaly injection**: `price_spike`, `margin_break`, `gtn_deviation`, `volume_outlier`

---

### `src/anomaly_detection.py` — `AnomalyDetectionEngine`

Ensemble of 6 detection methods:

```python
from src.anomaly_detection import AnomalyDetectionEngine

engine = AnomalyDetectionEngine(z_threshold=3.0, iqr_multiplier=2.5)
results = engine.detect_all()   # reads data/raw/drug_pricing.csv
```

| Method | What it catches |
|---|---|
| **Z-Score** | Rolling per-drug deviations (window = 12) |
| **IQR Fence** | Q1 − 2.5·IQR / Q3 + 2.5·IQR outliers |
| **Month-over-Month** | Sudden price spikes or drops > 25% |
| **Isolation Forest** | Multivariate outliers (scikit-learn) |
| **Local Outlier Factor** | Density-based local anomalies (scikit-learn) |
| **Regulatory Rules** | WAC breach, ASP/WAC violation, 340B ceiling breach, GTN floor breach, margin critical |

**Output schema**: `drug_id`, `anomaly_type`, `severity` (high/medium/low), `detection_method`, `anomaly_score`, `description`, `recommended_action`

---

### `src/rag_pipeline.py` — `RAGPipeline`

Local RAG pipeline — no external vector DB required:

```python
from src.rag_pipeline import RAGPipeline

rag = RAGPipeline()
answer = rag.answer("What is the Medicaid Best Price rule?")
print(answer)
```

- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (384-dim, local)
- **Vector store**: numpy dot-product similarity on normalised embeddings
- **Knowledge base**: 60 synthetic documents across 6 topics (WAC Policy, NADAC Benchmarks, 340B Rules, GTN Structures, Anomaly Case Studies, Regulatory Compliance)
- **Generation**: `claude-haiku-4-5-20251001`

---

### `src/agent.py` — `PharmaPricingAgent`

Multi-turn conversational agent with 6 built-in tools:

```python
from src.agent import PharmaPricingAgent

agent = PharmaPricingAgent()

# Blocking
print(agent.chat("What are the top anomalies for DRG025?"))

# Streaming
for chunk in agent.chat_streaming("Explain the 340B breach pattern for Revlimid"):
    print(chunk, end="", flush=True)
```

**Available tools**:

| Tool | Description |
|---|---|
| `get_anomaly_details` | Fetches top anomalies for a given drug |
| `query_pricing_database` | Runs read-only SQL against the pricing SQLite DB |
| `get_causal_summary` | Pre/post price trend analysis with volatility |
| `get_pharmacy_risk_profile` | Risk tier (red/yellow/green) per pharmacy |
| `search_regulatory_knowledge` | Searches the RAG vector store |
| `suggest_pricing_action` | Combines anomaly data + regulatory context into action plan |

---

## Dashboard Tabs

### 📊 Dashboard
- 6 KPI metrics (transactions, drugs, avg WAC, margin %, anomaly rate, high-severity count)
- WAC trend lines for top 8 drugs by volume
- GTN discount % heatmap by drug class × quarter
- Price benchmark comparison (WAC / ASP / AMP / 340B) by drug class
- Anomaly severity donut chart + top 10 most-flagged drugs
- Margin % box plots by drug class

### 🔍 Anomaly Explorer
- Filterable table by severity, anomaly type, and drug
- Anomaly type breakdown bar chart + detection method coverage chart
- WAC vs Actual Price scatter with diagonal reference line
- Anomaly volume timeline by severity

### 📚 Regulatory Q&A
- Conversational RAG interface with history
- Grounded in 60 pharma pricing knowledge documents
- Example questions for quick exploration

### 🤖 AI Copilot
- Streaming responses with real-time token delivery
- 6 data tools invoked automatically based on your question
- Conversation reset and summary controls

---

## Key Pricing Concepts

| Term | Definition |
|---|---|
| **WAC** | Wholesale Acquisition Cost — manufacturer list price to wholesalers |
| **ASP** | Average Sales Price — quarterly CMS-reported for Medicare Part B |
| **AMP** | Average Manufacturer Price — basis for Medicaid rebate calculations |
| **Best Price** | Lowest price available to any commercial customer; must be ≤ AMP |
| **340B** | Federal programme requiring ceiling prices for covered entities (≈ AMP × 0.769) |
| **GTN** | Gross-to-Net — gap between WAC list price and actual net revenue after rebates |
| **NADAC** | National Average Drug Acquisition Cost — retail pharmacy survey benchmark |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Streamlit App (app/)                           │
│  Dashboard │ Anomaly Explorer │ Regulatory Q&A │ AI Copilot      │
└──────┬────────────┬────────────────────┬────────────────────────┘
       │            │                    │
       ▼            ▼                    ▼
┌──────────────┐ ┌──────────────────┐  ┌─────────────────────────┐
│ Pricing      │ │ Anomaly          │  │   PharmaPricingAgent     │
│ DataSimulator│ │ DetectionEngine  │  │  (Claude Haiku + tools)  │
└──────────────┘ └──────────────────┘  └────────────┬────────────┘
                                                     │ tool calls
                                      ┌──────────────▼────────────┐
                                      │       Tool Executor        │
                                      │  get_anomaly_details       │
                                      │  query_pricing_database    │
                                      │  get_causal_summary        │
                                      │  get_pharmacy_risk_profile │
                                      │  search_regulatory_knowledge ──► RAGPipeline
                                      │  suggest_pricing_action    │
                                      └───────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Anthropic Claude Haiku (`claude-haiku-4-5-20251001`) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local, 384-dim) |
| Vector Store | numpy + cosine similarity (no external DB needed) |
| Anomaly Detection | scikit-learn (Isolation Forest, LOF) + scipy (Z-score, IQR) |
| Data | pandas, SQLite |
| Dashboard | Streamlit + Plotly |
| Testing | pytest + unittest.mock |

---

## Development

```bash
# Lint
ruff check src/ && black --check src/

# Type check
mypy src/

# Format
black src/
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Author

Built by [@anilmedads-hub](https://github.com/anilmedads-hub)
