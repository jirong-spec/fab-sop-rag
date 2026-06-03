"""Unit tests for the pure helpers in app.services.answer_service.

_estimate_tokens / _cosine / _fit_context_to_budget do not touch the embedding
model or the LLM, so they run without torch / network.
"""

import pytest

from app.services.answer_service import _cosine, _estimate_tokens, _fit_context_to_budget


def test_estimate_tokens_empty():
    assert _estimate_tokens("") == 0


def test_estimate_tokens_ascii_roughly_third():
    # non-CJK ≈ 3 chars / token (ceil-ish): "abc" -> (3+2)//3 == 1
    assert _estimate_tokens("abc") == 1
    assert _estimate_tokens("abcdef") == 2


def test_estimate_tokens_cjk_one_each():
    assert _estimate_tokens("中文字") == 3


def test_estimate_tokens_mixed():
    assert _estimate_tokens("中文abc") == 3  # 2 CJK + (3+2)//3=1


def test_cosine_identical_is_one():
    assert _cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert _cosine([1.0, 1.0], [1.0, 1.0]) == pytest.approx(1.0)


def test_cosine_orthogonal_is_zero():
    assert _cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_zero_vector_is_zero():
    assert _cosine([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_fit_budget_keeps_small_list_whole():
    scored = [(90, "短triple"), (80, "另一條")]
    assert _fit_context_to_budget("問題？", scored) == scored


def test_fit_budget_trims_oversized_list_but_keeps_at_least_one():
    huge = [(90, "x" * 600) for _ in range(200)]  # far exceeds the context budget
    kept = _fit_context_to_budget("問題？", huge)
    assert 1 <= len(kept) < len(huge)
    # ranking order is preserved (kept is a prefix of the input)
    assert kept == huge[: len(kept)]
