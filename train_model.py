"""
Train XGBoost classifier on technical features plus sentiment for BTC target labels.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

INPUT_CSV: Final[str] = "processed_btc.csv"
OUTPUT_MODEL: Final[str] = "xgb_crypto.pkl"
FEATURE_COLUMNS: Final[list[str]] = ["RSI", "EMA_dist", "ATR", "sentiment_score"]
TARGET_COLUMN: Final[str] = "target"

SOURCE_COLUMN_MAP: Final[dict[str, str]] = {
    "RSI_14": "RSI",
    "ema_dist_close": "EMA_dist",
    "ATR_14": "ATR",
}

SENTIMENT_SEED: Final[int] = 42


def load_processed_data(path: str = INPUT_CSV) -> pd.DataFrame:
    """Load feature CSV produced by quick_features.py."""
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Missing {csv_path}. Run quick_features.py to generate it."
        )
    return pd.read_csv(csv_path)


def inject_sentiment_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add simulated sentiment in [-1, 1].

    Replace with merged composite from run_multi_agent_pipeline() when news
    is joined by timestamp.
    """
    out = df.copy()
    rng = np.random.default_rng(SENTIMENT_SEED)
    out["sentiment_score"] = rng.uniform(-1.0, 1.0, size=len(out))
    return out


def build_feature_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Rename source columns and return X, y for training."""
    out = df.rename(columns=SOURCE_COLUMN_MAP)
    missing = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required columns after rename: {missing}")

    frame = out[FEATURE_COLUMNS + [TARGET_COLUMN]].dropna()
    x = frame[FEATURE_COLUMNS]
    y = frame[TARGET_COLUMN].astype(int)
    return x, y


def train_classifier(x: pd.DataFrame, y: pd.Series) -> XGBClassifier:
    """Fit XGBClassifier with project hyperparameters."""
    model = XGBClassifier(
        max_depth=3,
        learning_rate=0.05,
        random_state=42,
        eval_metric="logloss",
    )
    model.fit(x, y)
    return model


def log_feature_importance(model: XGBClassifier, feature_names: list[str]) -> None:
    """Print feature importances in column order."""
    importances = model.feature_importances_
    logger.info("Feature importance:")
    for name, importance in zip(feature_names, importances, strict=True):
        logger.info("  %s: %.4f", name, importance)

    ranked = sorted(
        zip(feature_names, importances, strict=True),
        key=lambda item: item[1],
        reverse=True,
    )
    logger.info("Ranked by importance:")
    for name, importance in ranked:
        logger.info("  %s: %.4f", name, importance)


def main() -> None:
    raw = load_processed_data()
    with_sentiment = inject_sentiment_score(raw)
    x, y = build_feature_frame(with_sentiment)

    logger.info("Training samples: %s", len(x))
    logger.info("Target distribution:\n%s", y.value_counts().sort_index().to_string())

    model = train_classifier(x, y)
    joblib.dump(model, OUTPUT_MODEL)

    logger.info("Model saved to %s", OUTPUT_MODEL)
    logger.info("Feature columns (inference order): %s", FEATURE_COLUMNS)
    log_feature_importance(model, FEATURE_COLUMNS)


if __name__ == "__main__":
    main()
