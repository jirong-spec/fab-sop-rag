"""Unit tests for the pure metric functions in scripts/eval_rigorous.py."""

from scripts.eval_rigorous import keyword_rate, recall, triple_present


def test_keyword_rate_all_present():
    assert keyword_rate("CheckVacuumPump then VerifyGasFlow", ["CheckVacuumPump", "VerifyGasFlow"]) == 1.0


def test_keyword_rate_partial():
    assert keyword_rate("only CheckVacuumPump", ["CheckVacuumPump", "VerifyGasFlow"]) == 0.5


def test_keyword_rate_none():
    assert keyword_rate("nothing relevant", ["CheckVacuumPump"]) == 0.0


def test_keyword_rate_case_insensitive():
    assert keyword_rate("checkvacuumpump", ["CheckVacuumPump"]) == 1.0


def test_keyword_rate_empty_keywords_is_one():
    assert keyword_rate("anything", []) == 1.0


def test_triple_present_matches_all_three_parts():
    assert triple_present(["A", "NEXT_STEP", "B"], ["(A)-[:NEXT_STEP]->(B)"]) is True


def test_triple_present_wrong_rel_type():
    assert triple_present(["A", "NEXT_STEP", "B"], ["(A)-[:DEPENDS_ON]->(B)"]) is False


def test_triple_present_wrong_nodes():
    assert triple_present(["A", "NEXT_STEP", "B"], ["(X)-[:NEXT_STEP]->(Y)"]) is False


def test_recall_none_when_no_gold():
    assert recall([], ["(A)-[:R]->(B)"]) is None


def test_recall_full_and_partial():
    triples = ["(A)-[:R]->(B)"]
    assert recall([["A", "R", "B"]], triples) == 1.0
    assert recall([["A", "R", "B"], ["C", "R", "D"]], triples) == 0.5
