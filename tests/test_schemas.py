"""Unit tests for app.schemas.AskRequest validation / defaults."""

import pytest
from pydantic import ValidationError

from app.schemas import AskRequest


def test_defaults():
    req = AskRequest(question="SOP_Etch_001 步驟？")
    assert req.enable_guards is True
    assert req.debug is False
    assert req.max_hop == 2
    assert req.top_k == 4


@pytest.mark.parametrize("hop", [0, 5, -1])
def test_max_hop_out_of_range(hop):
    with pytest.raises(ValidationError):
        AskRequest(question="x", max_hop=hop)


@pytest.mark.parametrize("k", [0, 21])
def test_top_k_out_of_range(k):
    with pytest.raises(ValidationError):
        AskRequest(question="x", top_k=k)


def test_empty_question_rejected():
    with pytest.raises(ValidationError):
        AskRequest(question="")


def test_overlong_question_rejected():
    with pytest.raises(ValidationError):
        AskRequest(question="x" * 1001)


def test_boundary_values_accepted():
    AskRequest(question="x" * 1000, max_hop=1, top_k=1)
    AskRequest(question="x", max_hop=4, top_k=20)
