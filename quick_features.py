"""
Feature engineering for BTC/USDT 1h candles: technical indicators and forward ATR target.
"""

from __future__ import annotations

import logging
from typing import Final

import numpy as np
import pandas as pd
import pandas_ta as ta
import requests

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SYMBOL: Final[str] = "BTCUSDT"
INTERVAL: Final[str] = "1h"
OUTPUT_CSV: Final[str] = "processed_btc.csv"
BINANCE_KLINES_URL: Final[str] = "https://api.binance.com/api/v3/klines"
FORWARD_BARS: Final[int] = 4
RSI_LENGTH: Final[int] = 14
EMA_LENGTH: Final[int] = 20
ATR_LENGTH: Final[int] = 14
DEFAULT_BARS: Final[int] = 1000
REQUEST_TIMEOUT_SEC: Final[int] = 30


def fetch_btc_usdt_1h(limit: int = DEFAULT_BARS) -> pd.DataFrame:
    """
    Load BTC/USDT 1-hour OHLCV from Binance public klines API.

    Raises on failure so callers can fall back to synthetic data.
    """
    params = {"symbol": SYMBOL, "interval": INTERVAL, "limit": limit}
    response = requests.get(
        BINANCE_KLINES_URL, params=params, timeout=REQUEST_TIMEOUT_SEC
    )
    response.raise_for_status()
    klines = response.json()

    if not klines:
        raise ValueError("Binance klines response was empty")

    rows = []
    for k in klines:
        rows.append(
            {
                "timestamp": pd.to_datetime(int(k[0]), unit="ms", utc=True),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
            }
        )

    df = pd.DataFrame(rows)
    logger.info("Loaded %s rows from Binance (%s %s)", len(df), SYMBOL, INTERVAL)
    return df


def generate_synthetic_ohlcv(n_bars: int = DEFAULT_BARS, seed: int = 42) -> pd.DataFrame:
    """Generate plausible 1h OHLCV when live data is unavailable."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2024-01-01", tz="UTC")
    timestamps = pd.date_range(start=start, periods=n_bars, freq="1h")

    # Geometric random walk for close prices
    returns = rng.normal(loc=0.0, scale=0.002, size=n_bars)
    close = 40_000.0 * np.exp(np.cumsum(returns))

    open_ = np.empty(n_bars)
    high = np.empty(n_bars)
    low = np.empty(n_bars)
    open_[0] = close[0]
    for i in range(n_bars):
        if i > 0:
            open_[i] = close[i - 1] * (1.0 + rng.normal(0, 0.0005))
        wiggle = abs(rng.normal(0, 0.0015))
        high[i] = max(open_[i], close[i]) * (1.0 + wiggle)
        low[i] = min(open_[i], close[i]) * (1.0 - wiggle)

    volume = rng.uniform(100.0, 5_000.0, size=n_bars)

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )
    logger.info("Generated %s synthetic OHLCV bars (seed=%s)", n_bars, seed)
    return df


def load_ohlcv() -> pd.DataFrame:
    """Try Binance; on any error, use synthetic OHLCV."""
    try:
        return fetch_btc_usdt_1h()
    except requests.RequestException as exc:
        logger.warning("Binance request failed (%s); using synthetic data.", exc)
    except (ValueError, KeyError, TypeError) as exc:
        logger.warning("Failed to parse Binance data (%s); using synthetic data.", exc)
    return generate_synthetic_ohlcv()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute RSI, EMA, EMA distance, and ATR with pandas-ta."""
    out = df.copy()

    out.ta.rsi(close="close", length=RSI_LENGTH, append=True)
    out.ta.ema(close="close", length=EMA_LENGTH, append=True)
    out.ta.atr(high="high", low="low", close="close", length=ATR_LENGTH, append=True)

    ema_col = f"EMA_{EMA_LENGTH}"
    atr_col = f"ATRr_{ATR_LENGTH}"
    if atr_col not in out.columns:
        # pandas-ta naming fallback
        atr_col = next((c for c in out.columns if c.startswith("ATR")), atr_col)

    # Scale-free distance of close from EMA (positive = price above EMA)
    out["ema_dist_close"] = (out["close"] - out[ema_col]) / out["close"]
    out = out.rename(columns={atr_col: "ATR_14"})

    return out


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Binary target: 1 if close rises over the next FORWARD_BARS by more than 1x ATR(14).
    """
    out = df.copy()
    future_close = out["close"].shift(-FORWARD_BARS)
    forward_move = future_close - out["close"]
    out["target"] = (forward_move > out["ATR_14"]).astype(int)
    return out


def main() -> None:
    df = load_ohlcv()
    df = add_indicators(df)
    df = add_target(df)
    df = df.dropna()

    df.to_csv(OUTPUT_CSV, index=False)

    target_counts = df["target"].value_counts().sort_index()
    logger.info("Saved %s rows to %s", len(df), OUTPUT_CSV)
    logger.info("Target distribution:\n%s", target_counts.to_string())


if __name__ == "__main__":
    main()
