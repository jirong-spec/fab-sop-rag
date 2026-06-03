"""Unit tests for app.utils.text_utils entity / source-doc extraction."""

from app.utils.text_utils import extract_candidate_tokens, extract_source_docs


def test_entity_code_titlecase_acronym_all_captured():
    toks = extract_candidate_tokens("處理 SOP_Etch_001 時確認 TurboVacuumPump，製程為 CVD")
    assert "SOP_Etch_001" in toks  # entity code (underscore id)
    assert "TurboVacuumPump" in toks  # multi-word TitleCase
    assert "CVD" in toks  # ALL-CAPS acronym


def test_entity_code_ordered_before_acronym():
    toks = extract_candidate_tokens("CVD 與 SOP_Etch_001")
    assert toks.index("SOP_Etch_001") < toks.index("CVD")


def test_dedup_keeps_first_occurrence_only():
    toks = extract_candidate_tokens("EtchStation EtchStation EtchStation")
    assert toks.count("EtchStation") == 1


def test_single_capitalised_word_not_titlecase():
    # _TITLE_CASE requires >= 2 capitalised components, so "What"/"Please" are skipped.
    assert "Please" not in extract_candidate_tokens("Please check VacuumPump")


def test_extract_source_docs_unique_and_ordered():
    triples = [
        "(SOP_Etch_001)-[:CROSS_DOC_DEPENDENCY]->(SOP_Pump_002)",
        "(SOP_Etch_001)-[:FIRST_STEP]->(CheckVacuumPump)",
    ]
    assert extract_source_docs(triples) == ["SOP_Etch_001", "SOP_Pump_002"]


def test_extract_source_docs_empty():
    assert extract_source_docs(["(CheckVacuumPump)-[:NEXT_STEP]->(VerifyGasFlow)"]) == []
