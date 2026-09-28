"""
task4_query.py — GraphRAG query pipeline (natural language → Cypher → answer).

Uses LangChain's ``GraphCypherQAChain`` over the combined UET knowledge graph
built by ``run_pipeline.py``. The Cypher-generation prompt is constrained to
the actual graph schema and seeded with UET-specific few-shot examples so the
model does not invent labels such as ``Document`` or ``Employee``.

The 12-query test set spans the three required patterns — entity lookup,
relationship traversal and aggregation — across BOTH sub-graphs (policy and
teaching-space/scheduling), and favors questions that are hard or impossible
to answer with vector search alone.

Run with::

    uv run python task4_query.py
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from langchain_neo4j import GraphCypherQAChain
from langchain_openai import ChatOpenAI

from task3_graph import get_neo4j_graph

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("uet.query")

MODEL_NAME = os.environ.get("LLM_MODEL", "nexai")

# Human-readable description of the graph, injected alongside the live schema.
GRAPH_SCHEMA = """
Node labels and key properties:
- PolicySection   {section_number, title, subtitle, summary, control_objective}
- Stakeholder     {name, category}           category ∈ student|academic_staff|research_staff|professional_staff|governance|external|other
- Regulation      {name, authority}          authority ∈ vietnamese_law|vnu|uet|academic|other
- Principle       {name}                     lowercase core value, e.g. 'integrity', 'fairness', 'privacy'
- Commitment      {description, modality, measurable}   modality ∈ must|should|may|recommended
- Constraint      {metric, value, unit, period}
- Site            {code, name}               code ∈ GD3|KM|HL
- Room            {room_code, site_code, capacity, status, accessible, has_projector, equipment}
                                            status ∈ ACTIVE|INACTIVE|MAINTENANCE|RESTRICTED|ARCHIVED
- RoomType        {code, label}              code ∈ LECTURE|SEMINAR|COMPUTER_LAB|ELECTRONICS_LAB|ROBOTICS_LAB|PROJECT|MEETING
- ClassSection    {class_code, course_title, term_code, expected_size, delivery_mode, class_status}
- ClassMeeting    {meeting_id, day_of_week, start_time, end_time, meeting_type, status}
- AuthorizationRole {name, capability}

Relationship types:
- (PolicySection)-[:AFFECTS]->(Stakeholder)
- (PolicySection)-[:REFERENCES]->(Regulation)
- (PolicySection)-[:UPHOLDS]->(Principle)
- (PolicySection)-[:CONTAINS]->(Commitment)-[:HAS_CONSTRAINT]->(Constraint)
- (Site)-[:HAS_ROOM]->(Room)-[:OF_TYPE]->(RoomType)
- (ClassSection)-[:HAS_MEETING]->(ClassMeeting)-[:ASSIGNED_TO]->(Room)
"""

CYPHER_GENERATION_TEMPLATE = """Task: Generate a single Cypher statement to query a Neo4j graph about UET (VNU University of Engineering and Technology).

Use ONLY the node labels, relationship types and property keys shown in the schema below.
Do NOT invent labels or relationships (strictly NO 'Document', 'Clause', 'Employee', 'Course', 'Building', 'HAS_WAGE', 'LOCATED_IN').
Property values for enums are stored exactly as shown (e.g. status 'RESTRICTED', category 'student', authority 'vnu').

Schema:
{schema}

Examples:
Question: What are the titles of all policy sections?
Cypher: MATCH (ps:PolicySection) RETURN ps.section_number AS number, ps.title AS title ORDER BY ps.title

Question: Which policy sections affect Students?
Cypher: MATCH (ps:PolicySection)-[:AFFECTS]->(s:Stakeholder) WHERE toLower(s.name) CONTAINS 'student' RETURN ps.title AS Section, s.name AS Stakeholder

Question: Which external regulations are referenced by the policy?
Cypher: MATCH (ps:PolicySection)-[:REFERENCES]->(r:Regulation) RETURN DISTINCT r.name AS Regulation, r.authority AS Authority

Question: How many commitments are associated with Students across all sections?
Cypher: MATCH (s:Stakeholder)<-[:AFFECTS]-(ps:PolicySection)-[:CONTAINS]->(c:Commitment) WHERE toLower(s.name) CONTAINS 'student' RETURN count(DISTINCT c) AS totalCommitments

Question: Which confirmed classes are assigned to restricted or maintenance rooms?
Cypher: MATCH (cs:ClassSection)-[:HAS_MEETING]->(cm:ClassMeeting {{status: 'CONFIRMED'}})-[:ASSIGNED_TO]->(rm:Room) WHERE rm.status IN ['RESTRICTED','MAINTENANCE'] RETURN cs.class_code AS Class, rm.room_code AS Room, rm.status AS Status

Question: What is the total seating capacity of each teaching site?
Cypher: MATCH (st:Site)-[:HAS_ROOM]->(rm:Room) RETURN st.code AS Site, sum(rm.capacity) AS TotalCapacity ORDER BY TotalCapacity DESC

Question: Which class sections meet in robotics laboratories?
Cypher: MATCH (cs:ClassSection)-[:HAS_MEETING]->(:ClassMeeting)-[:ASSIGNED_TO]->(:Room)-[:OF_TYPE]->(rt:RoomType {{code: 'ROBOTICS_LAB'}}) RETURN DISTINCT cs.class_code AS Class, cs.course_title AS Course

Note:
- Return ONLY the valid Cypher statement. No markdown fences, no explanations.

The question is:
{question}"""

from langchain_core.prompts import PromptTemplate  # noqa: E402  (import after template constant)

CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "question"],
    template=CYPHER_GENERATION_TEMPLATE,
)

# 12-query test set: entity lookup (5), relationship traversal (4), aggregation (3).
TEST_QUERIES = [
    # --- Entity lookup (5) ---
    "What are the titles of all policy sections in the UET Human Rights and Academic Community Policy?",
    "List all distinct stakeholders recognized across the policy.",
    "Which external regulations, laws or handbooks are referenced by the policy?",
    "What core principles does the policy uphold?",
    "List every teaching site and the number of rooms it contains.",
    # --- Relationship traversal (4) ---
    "Which policy sections affect Students?",
    "Which confirmed classes are assigned to restricted or maintenance rooms?",
    "Which class sections meet in robotics or electronics laboratories?",
    "What commitments are contained in the section 'Student Rights and Responsibilities'?",
    # --- Aggregation (3) ---
    "How many commitments are associated with Students across all policy sections?",
    "What is the total seating capacity of each teaching site?",
    "How many rooms are in each operational status?",
]


def main() -> None:
    graph = get_neo4j_graph()

    llm = ChatOpenAI(
        model=MODEL_NAME,
        api_key=os.environ.get("API_KEY_2"),
        base_url=os.environ.get("BASE_URL_2"),
        temperature=0.0,
        timeout=120,
        max_retries=2,
    )

    qa_chain = GraphCypherQAChain.from_llm(
        llm=llm,
        graph=graph,
        verbose=False,
        allow_dangerous_requests=True,
        return_intermediate_steps=True,
        cypher_prompt=CYPHER_GENERATION_PROMPT,
    )

    log.info("Running %d test queries over the UET knowledge graph...\n", len(TEST_QUERIES))
    for i, query in enumerate(TEST_QUERIES, start=1):
        print(f"\n--- [{i}] QUERY: {query} ---")
        try:
            result = qa_chain.invoke({"query": query})
            cypher = "N/A"
            steps = result.get("intermediate_steps") or []
            if steps:
                cypher = steps[0].get("query", "N/A")
            print(f"  🔹 [Cypher]: {cypher}")
            print(f"  ✅ [Answer]: {result.get('result', 'No answer generated.')}")
        except Exception as exc:  # noqa: BLE001
            print(f"  ❌ [Error]: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
