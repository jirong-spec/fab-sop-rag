import logging
from functools import lru_cache

from neo4j import GraphDatabase, Driver

from app.config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_driver() -> Driver:
    """Lazy-init the Neo4j driver; cached after first call."""
    logger.info("Connecting to Neo4j at %s", settings.neo4j_uri)
    return GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_username, settings.neo4j_password),
    )


def _node_label(node) -> str:
    """Return 'id[title]' when a title property exists, otherwise just 'id'."""
    node_id = node.get("id", "")
    title = node.get("title", "")
    return f"{node_id}[{title}]" if title else node_id


def graph_expand(entities: list[str], hop: int = 2) -> list[str]:
    """
    Expand from seed entities via graph traversal up to `hop` hops.

    Returns a deduplicated, insertion-ordered list of triples:
        (StartNode)-[:REL_TYPE]->(EndNode)

    The undirected match lets us discover edges in both directions;
    the triple string always reflects the stored edge direction.

    MVP note: cycle avoidance is handled at the triple-dedup level.
    A future improvement could add `WHERE ALL(n IN nodes(p) WHERE
    single(x IN nodes(p) WHERE x = n))` for strict simple-path filtering.
    """
    if not entities:
        return []

    driver = _get_driver()
    query = f"""
    MATCH p=(n)-[*1..{hop}]-(m)
    WHERE n.id IN $ents
    RETURN p LIMIT 200
    """

    seen: set[str] = set()
    result: list[str] = []

    with driver.session() as session:
        records = session.run(query, ents=entities)
        for record in records:
            path = record["p"]
            for rel in path.relationships:
                start_name = _node_label(rel.start_node)
                end_name = _node_label(rel.end_node)
                if not start_name or not end_name:
                    continue
                # Include edge properties (e.g. required_status, reason) so the
                # LLM knows not just *what* is required but *which value* is needed.
                rel_props = {k: v for k, v in dict(rel).items()}
                if rel_props:
                    prop_str = ", ".join(f"{k}: {v!r}" for k, v in rel_props.items())
                    triple = f"({start_name})-[:{rel.type} {{{prop_str}}}]->({end_name})"
                else:
                    triple = f"({start_name})-[:{rel.type}]->({end_name})"
                if triple not in seen:
                    seen.add(triple)
                    result.append(triple)

    return result
