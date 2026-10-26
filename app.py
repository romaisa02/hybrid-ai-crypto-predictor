"""
Streamlit dashboard: multi-agent sentiment, XGB trade signal, and Risk Governor.
"""

from __future__ import annotations

import asyncio
import threading
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, Final

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from multi_agent import run_multi_agent_pipeline
from train_model import FEATURE_COLUMNS, OUTPUT_MODEL, SOURCE_COLUMN_MAP

PROCESSED_CSV: Final[str] = "processed_btc.csv"
DRAWDOWN_LIMIT: Final[float] = -0.025
RISK_TICK_SEC: Final[float] = 5.0
_SCRIPT_DIR = Path(__file__).resolve().parent


def _init_session_state() -> None:
    defaults: dict[str, Any] = {
        "agent_scores": None,
        "risk_equity": 1.0,
        "risk_peak_equity": 1.0,
        "risk_drawdown_pct": 0.0,
        "risk_circuit_breaker": False,
        "risk_thread_started": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _risk_governor_loop() -> None:
    """Background simulation: update drawdown every 5 seconds."""
    rng = np.random.default_rng()
    while True:
        time.sleep(RISK_TICK_SEC)
        equity = st.session_state.get("risk_equity", 1.0)
        peak = st.session_state.get("risk_peak_equity", 1.0)

        equity *= 1.0 + float(rng.normal(0.0, 0.003))
        peak = max(peak, equity)
        drawdown = (equity - peak) / peak if peak > 0 else 0.0

        st.session_state["risk_equity"] = equity
        st.session_state["risk_peak_equity"] = peak
        st.session_state["risk_drawdown_pct"] = drawdown
        st.session_state["risk_circuit_breaker"] = drawdown <= DRAWDOWN_LIMIT


def start_risk_governor_thread() -> None:
    if st.session_state.get("risk_thread_started"):
        return
    thread = threading.Thread(target=_risk_governor_loop, daemon=True)
    thread.start()
    st.session_state["risk_thread_started"] = True


@st.cache_resource
def load_model():
    model_path = _SCRIPT_DIR / OUTPUT_MODEL
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Missing {model_path.name}. Run train_model.py first."
        )
    return joblib.load(model_path)


def latest_feature_row(sentiment_score: float) -> tuple[pd.DataFrame, str | None]:
    csv_path = _SCRIPT_DIR / PROCESSED_CSV
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Missing {PROCESSED_CSV}. Run quick_features.py first."
        )
    df = pd.read_csv(csv_path)
    row = df.iloc[[-1]].rename(columns=SOURCE_COLUMN_MAP)
    row["sentiment_score"] = sentiment_score
    timestamp = row["timestamp"].iloc[0] if "timestamp" in row.columns else None
    return row[FEATURE_COLUMNS], timestamp


def format_score(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:+.2f}"


def render_sidebar() -> None:
    st.sidebar.header("Live news")
    st.sidebar.text_area(
        "Live news headline",
        key="news_headline",
        height=120,
        placeholder="Paste a BTC headline or news snippet…",
    )
    if st.sidebar.button("Run multi-agent analysis", type="primary"):
        headline = (st.session_state.get("news_headline") or "").strip()
        if not headline:
            st.sidebar.warning("Enter a headline before running analysis.")
            return
        with st.spinner("Running Macro, Hype, and Whale agents…"):
            try:
                st.session_state["agent_scores"] = asyncio.run(
                    run_multi_agent_pipeline(headline)
                )
            except Exception as exc:
                st.sidebar.error(f"Pipeline failed: {exc}")


def render_agent_metrics() -> None:
    st.subheader("Agent sentiment scores")
    scores = st.session_state.get("agent_scores")

    macro = scores.get("macro_analyst") if scores else None
    retail = scores.get("retail_hype") if scores else None
    whale = scores.get("whale_tracker") if scores else None
    composite = scores.get("composite") if scores else None

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Macro Analyst", format_score(macro if isinstance(macro, float) else None))
    c2.metric("Retail Hype", format_score(retail if isinstance(retail, float) else None))
    c3.metric("Whale Tracker", format_score(whale if isinstance(whale, float) else None))
    c4.metric(
        "Composite Sentiment",
        format_score(composite if isinstance(composite, float) else None),
    )

    if not scores:
        st.caption("Run analysis from the sidebar to populate agent scores.")
    elif isinstance(scores, dict):
        with st.expander("Pipeline metadata"):
            st.json({"weights": scores.get("weights"), "models": scores.get("models")})


def render_trade_signal() -> None:
    st.subheader("Next-hour trade signal (XGBoost)")

    agent_scores = st.session_state.get("agent_scores")
    sentiment = 0.0
    if isinstance(agent_scores, dict) and isinstance(
        agent_scores.get("composite"), (int, float)
    ):
        sentiment = float(agent_scores["composite"])

    try:
        model = load_model()
        features, bar_time = latest_feature_row(sentiment)
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    proba = model.predict_proba(features)[0]
    pred = int(model.predict(features)[0])
    p_up = float(proba[1]) if len(proba) > 1 else float(proba[0])

    if st.session_state.get("risk_circuit_breaker"):
        signal_label = "HALTED (Circuit Breaker)"
    elif pred == 1 or p_up >= 0.5:
        signal_label = "BUY (bullish)"
    else:
        signal_label = "NO TRADE (bearish)"

    m1, m2, m3 = st.columns(3)
    m1.metric("Signal", signal_label)
    m2.metric("P(target=1)", f"{p_up:.1%}")
    m3.metric("Sentiment used", f"{sentiment:+.2f}")

    if bar_time is not None:
        st.caption(f"Technicals from latest bar: {bar_time}")

    st.markdown("**Features used for inference**")
    st.dataframe(features, use_container_width=True, hide_index=True)


@st.fragment(run_every=timedelta(seconds=5))
def render_risk_governor() -> None:
    st.subheader("Risk Governor")
    drawdown = float(st.session_state.get("risk_drawdown_pct", 0.0))
    equity = float(st.session_state.get("risk_equity", 1.0))
    breaker = bool(st.session_state.get("risk_circuit_breaker", False))

    if breaker:
        st.markdown(
            """
            <div style="padding: 1rem; border-radius: 8px; background-color: #5c1010;
            border: 2px solid #ff4b4b; color: #ffffff; font-weight: 700;">
            CIRCUIT BREAKER — Simulated portfolio drawdown exceeded -2.5%. Trading halted.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.error(
            "CIRCUIT BREAKER: Portfolio drawdown exceeded -2.5%. Trading halted."
        )

    r1, r2 = st.columns(2)
    r1.metric("Simulated equity (normalized)", f"{equity:.4f}")
    r2.metric("Drawdown", f"{drawdown * 100:.2f}%")
    st.caption(
        f"Background Risk Governor updates every {int(RISK_TICK_SEC)} seconds "
        "(simulated returns)."
    )


def main() -> None:
    st.set_page_config(page_title="Hybrid AI Crypto", layout="wide")
    _init_session_state()
    start_risk_governor_thread()

    st.title("Hybrid AI Crypto Prediction System")
    st.caption(
        "Nemotron multi-agent sentiment + XGBoost technicals for BTC next-hour signals."
    )

    render_sidebar()

    breaker_slot = st.container()
    with breaker_slot:
        if st.session_state.get("risk_circuit_breaker"):
            st.markdown(
                """
                <div style="padding: 0.75rem; margin-bottom: 1rem; border-radius: 8px;
                background-color: #5c1010; border: 2px solid #ff4b4b; color: #fff;
                font-weight: 700; text-align: center;">
                CIRCUIT BREAKER ACTIVE — Drawdown beyond -2.5%
                </div>
                """,
                unsafe_allow_html=True,
            )

    render_risk_governor()
    render_agent_metrics()
    render_trade_signal()


main()
