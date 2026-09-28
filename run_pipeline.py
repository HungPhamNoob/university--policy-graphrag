"""
run_pipeline.py — End-to-end build orchestrator for the UET GraphRAG system.

Steps
-----
1. Connect to Neo4j and clear any previous graph (idempotent rebuilds).
2. Load + clean the policy pages of ``data/UET_HR.pdf`` (Sections 1–29, Annex A/B).
3. Extract structured entities/relationships with the LLM (``task2``).
4. Build the Academic Community Policy sub-graph (``task3.build_policy_graph``).
5. Seed the Teaching-Space & Scheduling sub-graph (``task3.build_room_graph``).
6. Print a build report (node/relationship counts).

Run with::

    uv run python run_pipeline.py

Then ask questions over the graph with::

    uv run python task4_query.py
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv

from task2_extraction import build_extractors, extract_sections_and_log, load_policy_sections
from task3_graph import (
    build_policy_graph,
    build_room_graph,
    clear_graph,
    count_entities_by_type,
    get_neo4j_graph,
)

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("graphrag.pipeline")


def main() -> None:
    file_path = os.environ.get("FILE_PATH", "data/UET_HR.pdf")
    model_name = os.environ.get("LLM_MODEL", "nexai")
    reset = os.environ.get("RESET_GRAPH", "1") not in ("0", "false", "False", "")

    log.info("=== University Policy GraphRAG build ===")
    log.info("Source document : %s", file_path)
    log.info("LLM model       : %s", model_name)
    log.info("Reset graph     : %s", reset)

    graph = get_neo4j_graph()
    if reset:
        clear_graph(graph)

    # --- Domain A: Academic Community Policy (LLM extraction) ---------------
    sections = load_policy_sections(file_path)
    structured_chain, plain_llm = build_extractors(model_name=model_name)
    extractions = extract_sections_and_log(sections, structured_chain, plain_llm)
    policy_counts = build_policy_graph(extractions, graph)

    # --- Domain B: Teaching-Space & Scheduling (deterministic seed) --------
    room_counts = build_room_graph(graph)

    # --- Build report -------------------------------------------------------
    log.info("=== BUILD REPORT ===")
    log.info("Policy extraction : %s", policy_counts)
    log.info("Room/scheduling   : %s", room_counts)
    log.info("Neo4j node counts :")
    for row in count_entities_by_type(graph):
        log.info("   %-20s %d", row["EntityType"], row["TotalCount"])
    rel = graph.query("MATCH ()-[r]->() RETURN type(r) AS RelType, count(r) AS n ORDER BY n DESC")
    log.info("Neo4j relationships:")
    for row in rel:
        log.info("   %-20s %d", row["RelType"], row["n"])
    log.info("=== Build complete. Run `uv run python task4_query.py` to query the graph. ===")


if __name__ == "__main__":
    main()
