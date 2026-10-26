"""
Async multi-agent sentiment pipeline via Nebius (NVIDIA Nemotron) for crypto news text.
"""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from typing import Final

from openai import AsyncOpenAI

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

NEBIUS_BASE_URL: Final[str] = "https://api.studio.nebius.ai/v1/"
MODEL_REASONING: Final[str] = "nvidia/Nemotron-3-Ultra-550b-a55b"
# Alternative for retail: nvidia/nemotron-3-super-120b-a12b
MODEL_RETAIL_FAST: Final[str] = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"
KEY_PLAIN: Final[str] = "nebius_api_key"
KEY_RTF: Final[str] = "nebius_api_key.rtf"

SCORE_MIN: Final[float] = -1.0
SCORE_MAX: Final[float] = 1.0

WEIGHT_MACRO: Final[float] = 0.4
WEIGHT_RETAIL: Final[float] = 0.3
WEIGHT_WHALE: Final[float] = 0.3

MACRO_SYSTEM_PROMPT: Final[str] = (
    "You are a Macro Analyst for Bitcoin. Focus on regulations, central-bank and "
    "macro policy, geopolitics, and institutional rules that affect BTC. "
    "Respond with ONLY one decimal number from -1.0 (very bearish) to 1.0 (very bullish) "
    "for short-term BTC sentiment implied by the news. No other text."
)

RETAIL_SYSTEM_PROMPT: Final[str] = (
    "You are a Retail Hype Tracker for Bitcoin. Focus on social media buzz, memes, "
    "viral narratives, influencer takes, and crowd fear/greed. "
    "Respond with ONLY one decimal number from -1.0 (very bearish) to 1.0 (very bullish) "
    "for short-term BTC sentiment implied by the news. No other text."
)

WHALE_SYSTEM_PROMPT: Final[str] = (
    "You are a Whale Tracker for Bitcoin. Focus on large on-chain flows, exchange "
    "deposits/withdrawals, accumulation vs distribution, and smart-money positioning. "
    "Respond with ONLY one decimal number from -1.0 (very bearish) to 1.0 (very bullish) "
    "for short-term BTC sentiment implied by the news. No other text."
)

_USER_TEMPLATE: Final[str] = (
    "Score the following news for short-term BTC sentiment. Output only the number.\n\n"
    "{news_text}"
)

_SCRIPT_DIR = Path(__file__).resolve().parent
_client: AsyncOpenAI | None = None


def _strip_rtf(raw: str) -> str:
    """Extract plain text token from a minimal RTF key file."""
    if "\\cf0 " in raw:
        token = raw.split("\\cf0 ", 1)[1]
        token = token.rstrip("}")
        return token.strip()
    # Fallback: remove RTF control words and braces
    text = re.sub(r"\\[a-z]+\d*\s?", "", raw)
    text = text.replace("{", "").replace("}", "")
    return text.strip()


def load_nebius_api_key() -> str:
    """Load API key from nebius_api_key or nebius_api_key.rtf in this folder."""
    plain_path = _SCRIPT_DIR / KEY_PLAIN
    rtf_path = _SCRIPT_DIR / KEY_RTF

    if plain_path.is_file():
        key = plain_path.read_text(encoding="utf-8").strip()
        if key:
            return key

    if rtf_path.is_file():
        key = _strip_rtf(rtf_path.read_text(encoding="utf-8"))
        if key:
            return key

    raise FileNotFoundError(
        f"Nebius API key not found. Add {_SCRIPT_DIR / KEY_PLAIN} "
        f"or {_SCRIPT_DIR / KEY_RTF}."
    )


def get_client() -> AsyncOpenAI:
    """Lazy singleton AsyncOpenAI client."""
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            api_key=load_nebius_api_key(),
            base_url=NEBIUS_BASE_URL,
        )
    return _client


def _clamp_score(value: float) -> float:
    return max(SCORE_MIN, min(SCORE_MAX, value))


def _parse_score(content: str) -> float:
    """Parse model output; tolerate a single number embedded in short text."""
    text = content.strip()
    try:
        return _clamp_score(float(text))
    except ValueError:
        match = re.search(r"-?\d+\.?\d*", text)
        if not match:
            raise ValueError(f"No numeric score in response: {content!r}") from None
        return _clamp_score(float(match.group()))


async def async_query_score(
    client: AsyncOpenAI,
    system_prompt: str,
    news_text: str,
    model: str,
) -> float:
    """
    Generic async LLM sentiment query.

    Returns 0.0 on any API, network, or parse failure.
    """
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": _USER_TEMPLATE.format(news_text=news_text)},
            ],
            temperature=0.2,
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Empty model response")
        return _parse_score(content)
    except Exception as exc:
        logger.warning("async_query_score failed (%s); returning 0.0", exc)
        return 0.0


async def macro_analyst_agent(client: AsyncOpenAI, news_text: str) -> float:
    return await async_query_score(
        client, MACRO_SYSTEM_PROMPT, news_text, model=MODEL_REASONING
    )


async def retail_hype_agent(client: AsyncOpenAI, news_text: str) -> float:
    return await async_query_score(
        client, RETAIL_SYSTEM_PROMPT, news_text, model=MODEL_RETAIL_FAST
    )


async def whale_tracker_agent(client: AsyncOpenAI, news_text: str) -> float:
    return await async_query_score(
        client, WHALE_SYSTEM_PROMPT, news_text, model=MODEL_REASONING
    )


async def run_multi_agent_pipeline(news_text: str) -> dict[str, float | dict[str, float]]:
    """
    Run all three agents concurrently and return individual plus composite scores.
    """
    client = get_client()

    macro, retail, whale = await asyncio.gather(
        macro_analyst_agent(client, news_text),
        retail_hype_agent(client, news_text),
        whale_tracker_agent(client, news_text),
    )

    composite = (
        WEIGHT_MACRO * macro + WEIGHT_RETAIL * retail + WEIGHT_WHALE * whale
    )

    return {
        "macro_analyst": macro,
        "retail_hype": retail,
        "whale_tracker": whale,
        "composite": composite,
        "weights": {
            "macro_analyst": WEIGHT_MACRO,
            "retail_hype": WEIGHT_RETAIL,
            "whale_tracker": WEIGHT_WHALE,
        },
        "models": {
            "macro_analyst": MODEL_REASONING,
            "retail_hype": MODEL_RETAIL_FAST,
            "whale_tracker": MODEL_REASONING,
        },
    }


if __name__ == "__main__":
    sample_news = (
        "Bitcoin ETF inflows hit a weekly record while regulators delay a decision "
        "on a new stablecoin bill. Social media trends show rising retail FOMO."
    )
    outcome = asyncio.run(run_multi_agent_pipeline(sample_news))
    logger.info("Pipeline result: %s", outcome)
