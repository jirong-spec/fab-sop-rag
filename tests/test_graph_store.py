"""Unit tests for the pure serialization helpers in app.services.graph_store.

_node_label / _edge_gloss / _rel_to_triple are string logic. Neo4j node/relationship
objects are duck-typed here: a node is anything with .get(); a relationship is a dict
subclass (so dict(rel) yields its properties) plus a .type attribute.
"""

from app.services.graph_store import _edge_gloss, _node_label, _rel_to_triple


class FakeRel(dict):
    """Mimics a neo4j Relationship: dict(rel) -> properties, rel.type -> str."""

    def __init__(self, rel_type, props=None):
        super().__init__(props or {})
        self.type = rel_type


def test_node_label_with_title():
    assert _node_label({"id": "SOP_Etch_001", "title": "蝕刻"}) == "SOP_Etch_001[蝕刻]"


def test_node_label_without_title():
    assert _node_label({"id": "CheckVacuumPump"}) == "CheckVacuumPump"


def test_edge_gloss_requires_status_includes_value():
    g = _edge_gloss("REQUIRES_STATUS", "CheckVacuumPump", "TurboVacuumPump", {"required_status": "RUNNING"})
    assert g is not None and "RUNNING" in g and "CheckVacuumPump" in g and "TurboVacuumPump" in g


def test_edge_gloss_interlock_includes_trigger_action():
    g = _edge_gloss(
        "INTERLOCK_WITH",
        "EtchStation",
        "PressureInterlock",
        {"trigger": "pressure > 10 mTorr", "action": "disable RF power"},
    )
    assert "pressure > 10 mTorr" in g and "disable RF power" in g


def test_edge_gloss_unknown_type_is_none():
    # NEXT_STEP / DEPENDS_ON already ship a description in the seed, so no gloss is synthesised.
    assert _edge_gloss("NEXT_STEP", "A", "B", {}) is None


def test_rel_to_triple_adds_gloss_when_no_description():
    rel = FakeRel("REQUIRES_STATUS", {"required_status": "RUNNING"})
    triple = _rel_to_triple(rel, {"id": "CheckVacuumPump"}, {"id": "TurboVacuumPump"})
    assert triple.startswith("(CheckVacuumPump)-[:REQUIRES_STATUS")
    assert "(TurboVacuumPump)" in triple
    assert "description" in triple  # synthesised gloss was injected


def test_rel_to_triple_preserves_existing_description():
    rel = FakeRel("NEXT_STEP", {"description": "原本的說明"})
    triple = _rel_to_triple(rel, {"id": "A"}, {"id": "B"})
    assert "原本的說明" in triple


def test_rel_to_triple_none_when_endpoint_missing_id():
    rel = FakeRel("NEXT_STEP", {})
    assert _rel_to_triple(rel, {"id": ""}, {"id": "B"}) is None
