import logging
import threading

from neo4j import Driver, GraphDatabase
from neo4j.exceptions import ServiceUnavailable, SessionExpired
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.config import settings

logger = logging.getLogger(__name__)

_neo4j_retry = retry(
    retry=retry_if_exception_type((ServiceUnavailable, SessionExpired)),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=8),
    stop=stop_after_attempt(3),
    reraise=True,
)

_driver: Driver | None = None
_driver_lock = threading.Lock()


def _get_driver() -> Driver:
    """Lazy-init the Neo4j driver; singleton with double-checked locking."""
    global _driver
    if _driver is None:
        with _driver_lock:
            if _driver is None:
                logger.info("Connecting to Neo4j at %s", settings.neo4j_uri)
                _driver = GraphDatabase.driver(
                    settings.neo4j_uri,
                    auth=(settings.neo4j_username, settings.neo4j_password),
                )
    return _driver


def _node_label(node) -> str:
    """Return 'id[title]' when a title property exists, otherwise just 'id'."""
    node_id = node.get("id", "")
    title = node.get("title", "")
    return f"{node_id}[{title}]" if title else node_id


def _edge_gloss(rel_type: str, s: str, e: str, props: dict) -> str | None:
    """Synthesize a Traditional-Chinese description for an edge from its type +
    node IDs + props. All nine schema edge types are templated here, and
    _rel_to_triple always applies the gloss — so this function is the single
    source of truth for every edge's `description`. The gloss matters because
    CamelCase-only edges (REQUIRES_STATUS, NEXT_STEP, …) embed poorly against
    Chinese questions — without it the bi-encoder rerank pushes them down and the
    dynamic cap drops them."""
    st = props.get("required_status", "")
    if rel_type == "REQUIRES_STATUS":
        return f"{s} 步驟執行時要求設備 {e} 的狀態為 {st}"
    if rel_type == "PRECONDITION":
        return f"執行 {s} 前，設備 {e} 的前置狀態必須為 {st}"
    if rel_type == "INTERLOCK_WITH":
        return f"設備 {s} 與 {e} 聯鎖：當 {props.get('trigger', '')} 時，動作為 {props.get('action', '')}"
    if rel_type == "TRIGGERS_SOP":
        return f"異常 {s} 觸發應執行的 SOP 文件 {e}"
    if rel_type == "FIRST_STEP":
        return f"SOP 文件 {s} 的第一個步驟是 {e}"
    if rel_type == "DEFINED_IN":
        return f"步驟 {s} 定義於 SOP 文件 {e}"
    if rel_type == "CROSS_DOC_DEPENDENCY":
        return f"SOP 文件 {s} 跨文件依賴 {e}：{props.get('reason', '')}"
    if rel_type == "NEXT_STEP":
        return f"{s} 完成後，下一步執行 {e}"
    if rel_type == "DEPENDS_ON":
        return f"{s} 執行前必須先完成前置依賴步驟 {e}"
    return None


def _rel_to_triple(rel, start_node, end_node) -> str | None:
    """Serialize a relationship to a triple string, always reflecting the
    stored edge direction (start_node → end_node) regardless of how it was
    matched. Returns None if either endpoint has no id."""
    start_name = _node_label(start_node)
    end_name = _node_label(end_node)
    if not start_name or not end_name:
        return None
    rel_props = dict(rel)
    # _edge_gloss is the single source of truth for `description`: always (re)generate
    # it from the edge type, overwriting any stored value, so the wording lives in one place.
    gloss = _edge_gloss(rel.type, start_node.get("id", ""), end_node.get("id", ""), rel_props)
    if gloss:
        rel_props["description"] = gloss
    if rel_props:
        prop_str = ", ".join(f"{k}: {v!r}" for k, v in rel_props.items())
        return f"({start_name})-[:{rel.type} {{{prop_str}}}]->({end_name})"
    return f"({start_name})-[:{rel.type}]->({end_name})"


def graph_expand(entities: list[str], hop: int = 2) -> list[str]:
    """
    Expand from seed entities via graph traversal up to `hop` hops.

    Returns a deduplicated, insertion-ordered list of triples:
        (StartNode)-[:REL_TYPE]->(EndNode)

    Collects the DISTINCT relationships reachable within `hop` hops of any seed,
    rather than enumerating variable-length paths (which explodes and gets
    truncated on a connected graph). Each triple string reflects the stored edge
    direction regardless of how it was matched (see _rel_to_triple); duplicates
    are removed by the `seen` set below.
    """
    if not entities:
        return []

    driver = _get_driver()
    # hop is validated as int (1-4) by AskRequest; explicit cast prevents any
    # future code path from accidentally passing a string here.
    hop = int(hop)
    query = f"""
    MATCH (n)-[r*1..{hop}]-(m)
    WHERE n.id IN $ents
    UNWIND r AS rel
    WITH DISTINCT rel
    RETURN startNode(rel) AS s, rel AS r, endNode(rel) AS e
    LIMIT 500
    """

    @_neo4j_retry
    def _query():
        with driver.session() as session:
            return list(session.run(query, ents=entities))

    records = _query()
    if len(records) == 500:
        logger.warning(
            "graph_expand hit LIMIT 500 for entities=%s hop=%d — results may be truncated",
            entities,
            hop,
        )

    seen: set[str] = set()
    result: list[str] = []
    for record in records:
        triple = _rel_to_triple(record["r"], record["s"], record["e"])
        if triple and triple not in seen:
            seen.add(triple)
            result.append(triple)

    return result
