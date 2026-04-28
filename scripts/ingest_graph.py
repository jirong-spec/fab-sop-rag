"""
Seed Neo4j with the SOP graph from data/graph_seed/nodes.json and edges.json.

Usage (inside Docker):
    docker compose run --rm api python scripts/ingest_graph.py

Usage (local, from fab-sop-rag/):
    NEO4J_URI=bolt://localhost:7687 python scripts/ingest_graph.py

The script uses MERGE so it is safe to re-run: existing nodes/edges are
updated in place rather than duplicated.
"""

import json
import logging
import sys
from pathlib import Path

# Allow `from app.config import settings` when running from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

_GRAPH_SEED_DIR = Path(__file__).resolve().parent.parent / "data" / "graph_seed"


def _load_json(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _merge_nodes(session, nodes: list[dict]) -> None:
    for node in nodes:
        label = node["label"]
        props = node["properties"]
        node_id = props["id"]

        # MERGE on the id property, then SET all remaining properties
        cypher = (
            f"MERGE (n:{label} {{id: $id}}) "
            "SET n += $props "
            "RETURN n.id AS id"
        )
        result = session.run(cypher, id=node_id, props=props)
        record = result.single()
        logger.info("MERGE node  (%s {id: %r})", label, record["id"] if record else node_id)


def _merge_edges(session, edges: list[dict]) -> None:
    for edge in edges:
        rel_type = edge["type"]
        from_label = edge["from_label"]
        from_id = edge["from_id"]
        to_label = edge["to_label"]
        to_id = edge["to_id"]
        props = edge.get("properties", {})

        cypher = (
            f"MATCH (a:{from_label} {{id: $from_id}}) "
            f"MATCH (b:{to_label} {{id: $to_id}}) "
            f"MERGE (a)-[r:{rel_type}]->(b) "
            "SET r += $props "
            "RETURN type(r) AS rel, a.id AS from_id, b.id AS to_id"
        )
        result = session.run(cypher, from_id=from_id, to_id=to_id, props=props)
        record = result.single()
        if record:
            logger.info(
                "MERGE edge  (%s)-[%s]->(%s)",
                record["from_id"],
                record["rel"],
                record["to_id"],
            )
        else:
            logger.warning(
                "Edge skipped — node not found: (%s {id:%r})-[%s]->(%s {id:%r})",
                from_label, from_id, rel_type, to_label, to_id,
            )


def main() -> None:
    from neo4j import GraphDatabase  # imported here to fail fast with a clear message

    nodes_path = _GRAPH_SEED_DIR / "nodes.json"
    edges_path = _GRAPH_SEED_DIR / "edges.json"

    logger.info("Loading nodes from %s", nodes_path)
    nodes = _load_json(nodes_path)
    logger.info("Loading edges from %s", edges_path)
    edges = _load_json(edges_path)

    logger.info(
        "Connecting to Neo4j at %s (user=%s)", settings.neo4j_uri, settings.neo4j_username
    )
    driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_username, settings.neo4j_password),
    )

    try:
        with driver.session() as session:
            logger.info("--- Merging %d nodes ---", len(nodes))
            _merge_nodes(session, nodes)

            logger.info("--- Merging %d edges ---", len(edges))
            _merge_edges(session, edges)

        logger.info("Graph seed complete.")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
