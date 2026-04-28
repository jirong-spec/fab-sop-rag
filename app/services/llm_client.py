import logging
import os

from openai import OpenAI, APIError, APITimeoutError, APIConnectionError

from app.config import settings

logger = logging.getLogger(__name__)

# Ensure env var is set before client init (required by some openai versions)
os.environ.setdefault("OPENAI_API_KEY", settings.openai_api_key)

_client = OpenAI(
    api_key=settings.openai_api_key,
    base_url=settings.openai_api_base,
    timeout=settings.llm_timeout,
)


def chat_completion(
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> str:
    """
    Send a single-turn chat prompt to the local LLM and return the text response.
    Raises RuntimeError on unrecoverable API errors (caller decides how to handle).
    """
    try:
        resp = _client.chat.completions.create(
            model=settings.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = resp.choices[0].message.content
        return (content or "").strip()
    except APITimeoutError as e:
        logger.error("LLM API timeout after %.1fs: %s", settings.llm_timeout, e)
        raise RuntimeError(f"LLM timeout: {e}") from e
    except APIConnectionError as e:
        logger.error("LLM API connection error: %s", e)
        raise RuntimeError(f"LLM connection error: {e}") from e
    except APIError as e:
        logger.error("LLM API error (status=%s): %s", getattr(e, "status_code", "?"), e)
        raise RuntimeError(f"LLM API error: {e}") from e
