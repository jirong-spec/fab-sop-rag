import logging
import os
from collections.abc import Iterator

from openai import OpenAI, APIError, APITimeoutError, APIConnectionError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.config import settings

logger = logging.getLogger(__name__)

# Ensure env var is set before client init (required by some openai versions)
os.environ.setdefault("OPENAI_API_KEY", settings.openai_api_key)

_client = OpenAI(
    api_key=settings.openai_api_key,
    base_url=settings.openai_api_base,
    timeout=settings.llm_timeout,
)

# Retry transient connection/timeout errors up to 3 times with exponential backoff.
# APIError (e.g. 400 bad request, 404 model not found) is NOT retried — those are
# permanent failures that retrying would not fix.
_llm_retry = retry(
    retry=retry_if_exception_type((APIConnectionError, APITimeoutError)),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
    stop=stop_after_attempt(3),
    reraise=True,
)


@_llm_retry
def _call_llm(prompt: str, temperature: float, max_tokens: int) -> str:
    """Raw LLM call — raises original APIError subclasses so tenacity can retry."""
    resp = _client.chat.completions.create(
        model=settings.llm_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()


def chat_completion(
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> str:
    """
    Send a single-turn chat prompt to the local LLM and return the text response.
    Retries up to 3× on transient connection/timeout errors.
    Raises RuntimeError on unrecoverable API errors (caller decides how to handle).
    """
    try:
        return _call_llm(prompt, temperature, max_tokens)
    except APITimeoutError as e:
        logger.error("LLM API timeout after %.1fs: %s", settings.llm_timeout, e)
        raise RuntimeError(f"LLM timeout: {e}") from e
    except APIConnectionError as e:
        logger.error("LLM API connection error: %s", e)
        raise RuntimeError(f"LLM connection error: {e}") from e
    except APIError as e:
        logger.error("LLM API error (status=%s): %s", getattr(e, "status_code", "?"), e)
        raise RuntimeError(f"LLM API error: {e}") from e


@_llm_retry
def _create_stream(prompt: str, temperature: float, max_tokens: int):
    """Create the vLLM streaming context manager — retried on transient failures."""
    return _client.chat.completions.create(
        model=settings.llm_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )


def chat_completion_stream(
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> Iterator[str]:
    """
    Stream token strings from the LLM as they arrive.
    Yields each non-empty delta content string.
    Raises RuntimeError on connection / API errors.
    """
    try:
        with _create_stream(prompt, temperature, max_tokens) as stream:
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
    except APITimeoutError as e:
        logger.error("LLM stream timeout: %s", e)
        raise RuntimeError(f"LLM timeout: {e}") from e
    except APIConnectionError as e:
        logger.error("LLM stream connection error: %s", e)
        raise RuntimeError(f"LLM connection error: {e}") from e
    except APIError as e:
        logger.error("LLM stream API error (status=%s): %s", getattr(e, "status_code", "?"), e)
        raise RuntimeError(f"LLM API error: {e}") from e
