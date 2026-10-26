# 🚀 Hybrid AI Crypto Predictor

A high-performance, autonomous crypto prediction and risk-governance engine built for the **Nemotron hackathon category**, powered by **NVIDIA Nemotron models on Nebius AI Studio**.

Combines **technical indicators** (RSI, EMA distance, ATR), **multi-agent LLM sentiment** (Macro, Retail Hype, Whale), and an **XGBoost** classifier—wrapped in a **Streamlit** dashboard with a simulated **Risk Governor** and circuit breaker.

---

## Tech stack

| Layer | Technologies |
|--------|----------------|
| Language | Python 3.15+ |
| Market data & features | pandas, pandas-ta, NumPy, Binance public API (fallback synthetic OHLCV) |
| Multi-agent AI | OpenAI SDK (`AsyncOpenAI`), Nebius AI Studio, NVIDIA Nemotron (Ultra + Nano) |
| ML | XGBoost (`XGBClassifier`), scikit-learn, joblib |
| UI | Streamlit |
| Risk | Background thread + drawdown monitor (-2.5% circuit breaker) |

### Nemotron routing (Nebius)

| Agent | Model |
|--------|--------|
| Macro Analyst | `nvidia/Nemotron-3-Ultra-550b-a55b` |
| Whale Tracker | `nvidia/Nemotron-3-Ultra-550b-a55b` |
| Retail Hype Tracker | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B` |

Base URL: `https://api.studio.nebius.ai/v1/`

---

## Project layout

| File | Role |
|------|------|
| `quick_features.py` | Fetch/generate BTC 1h OHLCV, engineer indicators, build target → `processed_btc.csv` |
| `multi_agent.py` | Async three-agent sentiment pipeline → composite score |
| `train_model.py` | Train XGB on technicals + sentiment → `xgb_crypto.pkl` |
| `app.py` | Streamlit dashboard: news analysis, trade signal, Risk Governor |

---

## Setup

### 1. Clone and virtual environment

```bash
cd "Hybrid AI Crypto Prediction System"
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install pandas pandas-ta numpy requests openai xgboost joblib scikit-learn streamlit tqdm
```

**macOS (XGBoost):** install OpenMP if import fails:

```bash
brew install libomp
```

**Python 3.15 + pandas-ta:** you may need a compatible `numba` build (e.g. `pip install numba>=0.68`).

### 3. Nebius API key

Create a .env file in the project root:

Paste your Nebius Studio API key pair value

### 4. Build data and model

```bash
python quick_features.py    # → processed_btc.csv
python train_model.py       # → xgb_crypto.pkl
```

Optional smoke test for agents:

```bash
python multi_agent.py
```

### 5. Run the dashboard

```bash
streamlit run app.py
```

Use the sidebar to paste a **live news headline** and run **multi-agent analysis**. The main view shows agent scores, composite sentiment, XGB **next-hour signal**, and Risk Governor status.

---

## Pipeline flow

```text
Binance 1h OHLCV → quick_features.py → processed_btc.csv
News headline    → multi_agent.py (Nemotron) → sentiment scores
Technicals + sentiment → train_model.py → xgb_crypto.pkl
All of the above → app.py (Streamlit + Risk Governor)
```

---

## Disclaimer

This project is for **research and hackathon demonstration** only. It is not financial advice. Simulated portfolio and circuit-breaker logic are not production trading systems.
