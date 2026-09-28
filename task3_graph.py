"""
task3_graph.py — Knowledge-graph construction and Cypher helpers.

Two sub-graphs are written into the same Neo4j database:

  A. Academic Community Policy  — built from LLM extractions (``task2``)
     using ``MERGE`` so repeated runs stay idempotent.

  B. Teaching-Space & Scheduling — seeded deterministically from the
     illustrative registry in Section C of ``data/UET_HR.pdf`` (rooms,
     class sections, meetings, sites, room types, authorization roles).
     Section C is structured tabular data, so seeding it directly is more
     reliable than LLM extraction and still exercises every graph pattern
     (multi-hop traversal, capacity constraints, status filtering).

Graph schema
------------
    (PolicySection)-[:AFFECTS]->(Stakeholder)
    (PolicySection)-[:REFERENCES]->(Regulation)
    (PolicySection)-[:UPHOLDS]->(Principle)
    (PolicySection)-[:CONTAINS]->(Commitment)-[:HAS_CONSTRAINT]->(Constraint)

    (Site)-[:HAS_ROOM]->(Room)-[:OF_TYPE]->(RoomType)
    (ClassSection)-[:HAS_MEETING]->(ClassMeeting)-[:ASSIGNED_TO]->(Room)
    (AuthorizationRole)   -- capability catalog over the room/scheduling data

Run with::

    uv run python task3_graph.py     # seeds the room/scheduling sub-graph
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph

from task1_schema import PolicySectionExtraction

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("uet.graph")


# -----------------------------------------------------------------------------
# Connection helpers
# -----------------------------------------------------------------------------
def get_neo4j_graph(refresh_schema: bool = True) -> Neo4jGraph:
    """Open a Neo4j connection from environment variables (with sane defaults)."""
    return Neo4jGraph(
        url=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        username=os.environ.get("NEO4J_USERNAME", "neo4j"),
        password=os.environ.get("NEO4J_PASSWORD", "password"),
        refresh_schema=refresh_schema,
    )


def clear_graph(graph: Neo4jGraph) -> None:
    """Detach-delete every node so a rebuild starts from a clean slate."""
    graph.query("MATCH (n) DETACH DELETE n")
    log.info("Cleared existing graph.")


# -----------------------------------------------------------------------------
# DOMAIN B seed data — transcribed from Section C of data/UET_HR.pdf
# (illustrative reference data, not official UET inventory)
# -----------------------------------------------------------------------------
SITES: List[Dict[str, str]] = [
    {"code": "GD3", "name": "Giảng đường 3"},
    {"code": "KM", "name": "Kiều Mai teaching area"},
    {"code": "HL", "name": "Hòa Lạc campus"},
]

ROOM_TYPES: List[Dict[str, str]] = [
    {"code": "LECTURE", "label": "Lecture room"},
    {"code": "SEMINAR", "label": "Seminar room"},
    {"code": "COMPUTER_LAB", "label": "Computer laboratory"},
    {"code": "ELECTRONICS_LAB", "label": "Electronics laboratory"},
    {"code": "ROBOTICS_LAB", "label": "Robotics / automation laboratory"},
    {"code": "PROJECT", "label": "Project room"},
    {"code": "MEETING", "label": "Meeting room"},
]

# room_code, site, type, capacity, status, equipment
_ROOM_ROWS = [
    ("GD3-101", "GD3", "LECTURE", 120, "ACTIVE", ["projector", "microphone", "capture"]),
    ("GD3-102", "GD3", "LECTURE", 90, "ACTIVE", ["projector", "speakers"]),
    ("GD3-201", "GD3", "LECTURE", 80, "ACTIVE", ["projector", "display"]),
    ("GD3-202", "GD3", "SEMINAR", 48, "ACTIVE", ["display", "flexible seating"]),
    ("GD3-301", "GD3", "COMPUTER_LAB", 42, "ACTIVE", ["42 workstations", "wired network"]),
    ("GD3-302", "GD3", "COMPUTER_LAB", 36, "MAINTENANCE", ["36 workstations", "lab VLAN"]),
    ("GD3-401", "GD3", "PROJECT", 28, "ACTIVE", ["display", "whiteboard"]),
    ("GD3-402", "GD3", "MEETING", 16, "ACTIVE", ["video conference"]),
    ("KM-101", "KM", "LECTURE", 100, "ACTIVE", ["projector", "microphone"]),
    ("KM-102", "KM", "LECTURE", 70, "ACTIVE", ["projector", "speakers"]),
    ("KM-201", "KM", "SEMINAR", 40, "ACTIVE", ["display", "movable desks"]),
    ("KM-202", "KM", "COMPUTER_LAB", 40, "ACTIVE", ["40 workstations", "wired network"]),
    ("KM-301", "KM", "ELECTRONICS_LAB", 32, "RESTRICTED", ["oscilloscopes", "bench supplies"]),
    ("KM-302", "KM", "PROJECT", 24, "ACTIVE", ["whiteboards", "team tables"]),
    ("HL-A101", "HL", "LECTURE", 150, "ACTIVE", ["projector", "microphone", "capture"]),
    ("HL-A102", "HL", "LECTURE", 120, "ACTIVE", ["dual display", "microphone"]),
    ("HL-A201", "HL", "SEMINAR", 54, "ACTIVE", ["display", "flexible seating"]),
    ("HL-A301", "HL", "COMPUTER_LAB", 48, "ACTIVE", ["48 workstations", "lab network"]),
    ("HL-B101", "HL", "ELECTRONICS_LAB", 36, "ACTIVE", ["bench supplies", "instruments"]),
    ("HL-B201", "HL", "ROBOTICS_LAB", 30, "RESTRICTED", ["robot platforms", "safety zone"]),
    ("HL-B202", "HL", "ROBOTICS_LAB", 24, "MAINTENANCE", ["automation cells"]),
    ("HL-C101", "HL", "PROJECT", 36, "ACTIVE", ["team tables", "display"]),
    ("HL-C102", "HL", "PROJECT", 24, "ACTIVE", ["whiteboards", "storage"]),
    ("HL-C201", "HL", "MEETING", 18, "ACTIVE", ["video conference"]),
]

ROOMS: List[Dict[str, Any]] = [
    {
        "room_code": code,
        "site_code": site,
        "room_type": rtype,
        "capacity": cap,
        "status": status,
        "accessible": True,
        "has_projector": any("projector" in e.lower() for e in equip),
        "equipment": equip,
    }
    for (code, site, rtype, cap, status, equip) in _ROOM_ROWS
]

# class_code, course_title, expected_size, day, start, end, room_code, state
_CLASS_ROWS = [
    ("AI-01", "Introduction to Artificial Intelligence", 78, "MON", "09:00", "10:50", "GD3-201", "CONFIRMED"),
    ("AI-02", "Introduction to Artificial Intelligence", 44, "TUE", "13:00", "14:50", "KM-201", "DRAFT"),
    ("SE-01", "Software Engineering", 88, "WED", "09:00", "10:50", "GD3-102", "CONFIRMED"),
    ("DSA-01", "Data Structures and Algorithms", 118, "THU", "07:00", "08:50", "GD3-101", "CONFIRMED"),
    ("DB-01", "Database Systems", 38, "FRI", "13:00", "15:50", "KM-202", "CONFIRMED"),
    ("NET-01", "Computer Networks Lab", 34, "TUE", "09:00", "11:50", "GD3-301", "CONFIRMED"),
    ("EMB-01", "Embedded Systems Practice", 30, "WED", "13:00", "15:50", "KM-301", "CONFIRMED"),
    ("ROB-01", "Robotics Laboratory", 26, "THU", "13:00", "15:50", "HL-B201", "CONFIRMED"),
    ("ML-01", "Machine Learning", 116, "FRI", "09:00", "10:50", "HL-A102", "CONFIRMED"),
    ("HCI-01", "Human-Computer Interaction", 50, "MON", "13:00", "14:50", "HL-A201", "CONFIRMED"),
    ("CAP-01", "Capstone Project Studio", 32, "SAT", "08:00", "11:50", "HL-C101", "CONFIRMED"),
    ("SEM-01", "Research Seminar", 22, "FRI", "15:00", "16:50", "KM-302", "CONFIRMED"),
    ("ADM-01", "Academic Coordination Meeting", 14, "WED", "16:00", "17:00", "HL-C201", "CONFIRMED"),
    ("AI-01-L", "AI Practical Session", 40, "THU", "09:00", "11:50", "HL-A301", "CONFIRMED"),
    ("ELEC-01", "Digital Electronics Practice", 32, "FRI", "13:00", "15:50", "HL-B101", "CONFIRMED"),
]

_ROOM_TYPE_BY_CODE = {r["room_code"]: r["room_type"] for r in ROOMS}
_MEETING_TYPE_BY_ROOM_TYPE = {
    "LECTURE": "LECTURE",
    "SEMINAR": "TUTORIAL",
    "COMPUTER_LAB": "PRACTICAL",
    "ELECTRONICS_LAB": "PRACTICAL",
    "ROBOTICS_LAB": "PRACTICAL",
    "PROJECT": "PRACTICAL",
    "MEETING": "TUTORIAL",
}


def _meeting_type(room_code: str) -> str:
    return _MEETING_TYPE_BY_ROOM_TYPE.get(_ROOM_TYPE_BY_CODE.get(room_code, "LECTURE"), "LECTURE")


CLASS_SECTIONS: List[Dict[str, Any]] = [
    {
        "class_code": code,
        "course_title": title,
        "term_code": "2026-FALL",
        "expected_size": size,
        "delivery_mode": "IN_PERSON",
        "class_status": "DRAFT" if state == "DRAFT" else "ACTIVE",
    }
    for (code, title, size, _d, _s, _e, _r, state) in _CLASS_ROWS
]

CLASS_MEETINGS: List[Dict[str, Any]] = [
    {
        "meeting_id": f"MTG-{code}",
        "class_code": code,
        "room_code": room,
        "day_of_week": day,
        "start_time": start,
        "end_time": end,
        "meeting_type": _meeting_type(room),
        "status": state,
    }
    for (code, _t, _sz, day, start, end, room, state) in _CLASS_ROWS
]

AUTHORIZATION_ROLES: List[Dict[str, str]] = [
    {"name": "Viewer", "capability": "Read room registry and published schedules"},
    {"name": "Scheduler", "capability": "Create/update class meetings and reservations; cannot change structural room attributes"},
    {"name": "Facility administrator", "capability": "Maintain room capacity, type, equipment and maintenance state"},
    {"name": "System administrator", "capability": "Manage technical configuration and exceptional recovery; no automatic academic scheduling authority"},
]


# -----------------------------------------------------------------------------
# DOMAIN A — policy graph builder
# -----------------------------------------------------------------------------
def build_policy_graph(extractions: List[PolicySectionExtraction], graph: Neo4jGraph) -> Dict[str, int]:
    """Write extracted policy sections into Neo4j with idempotent MERGE upserts."""
    counts = {"sections": 0, "stakeholders": 0, "regulations": 0, "principles": 0, "commitments": 0, "constraints": 0}

    for ex in extractions:
        title = (ex.title or "").strip()
        if not title:
            continue

        graph.query(
            """
            MERGE (ps:PolicySection {title: $title})
            SET ps.section_number = $section_number,
                ps.subtitle       = $subtitle,
                ps.summary        = $summary,
                ps.control_objective = $control_objective
            """,
            {
                "title": title,
                "section_number": ex.section_number,
                "subtitle": ex.subtitle,
                "summary": ex.summary,
                "control_objective": ex.control_objective,
            },
        )
        counts["sections"] += 1

        stakeholders = [{"name": s.name.strip(), "category": s.category.value} for s in ex.stakeholders if s.name and s.name.strip()]
        if stakeholders:
            graph.query(
                """
                MATCH (ps:PolicySection {title: $title})
                UNWIND $rows AS row
                MERGE (s:Stakeholder {name: row.name})
                SET s.category = row.category
                MERGE (ps)-[:AFFECTS]->(s)
                """,
                {"title": title, "rows": stakeholders},
            )
            counts["stakeholders"] += len(stakeholders)

        regulations = [{"name": r.name.strip(), "authority": r.authority.value} for r in ex.regulations if r.name and r.name.strip()]
        if regulations:
            graph.query(
                """
                MATCH (ps:PolicySection {title: $title})
                UNWIND $rows AS row
                MERGE (r:Regulation {name: row.name})
                SET r.authority = row.authority
                MERGE (ps)-[:REFERENCES]->(r)
                """,
                {"title": title, "rows": regulations},
            )
            counts["regulations"] += len(regulations)

        principles = sorted({(p or "").strip().lower() for p in ex.principles if p and p.strip()})
        if principles:
            graph.query(
                """
                MATCH (ps:PolicySection {title: $title})
                UNWIND $rows AS name
                MERGE (p:Principle {name: name})
                MERGE (ps)-[:UPHOLDS]->(p)
                """,
                {"title": title, "rows": principles},
            )
            counts["principles"] += len(principles)

        commitments = []
        for c in ex.commitments:
            desc = (c.description or "").strip()
            if not desc:
                continue
            commitments.append(
                {
                    "description": desc,
                    "modality": c.modality.value,
                    "measurable": bool(c.measurable),
                    "constraints": [
                        {
                            "metric": con.metric.strip(),
                            "value": float(con.value),
                            "unit": con.unit.value,
                            "period": con.period.value,
                        }
                        for con in c.constraints
                        if con.metric and con.metric.strip()
                    ],
                }
            )
        if commitments:
            graph.query(
                """
                MATCH (ps:PolicySection {title: $title})
                UNWIND $rows AS row
                MERGE (com:Commitment {description: row.description})
                SET com.modality = row.modality, com.measurable = row.measurable
                MERGE (ps)-[:CONTAINS]->(com)
                WITH com, row
                UNWIND row.constraints AS con
                MERGE (c:Constraint {metric: con.metric, unit: con.unit, period: con.period})
                SET c.value = con.value
                MERGE (com)-[:HAS_CONSTRAINT]->(c)
                """,
                {"title": title, "rows": commitments},
            )
            counts["commitments"] += len(commitments)
            counts["constraints"] += sum(len(c["constraints"]) for c in commitments)

    log.info("Policy graph built: %s", counts)
    return counts


# -----------------------------------------------------------------------------
# DOMAIN B — room / scheduling graph builder (deterministic seed)
# -----------------------------------------------------------------------------
def build_room_graph(graph: Neo4jGraph) -> Dict[str, int]:
    """Seed the teaching-space & scheduling sub-graph from Section C data."""
    graph.query("UNWIND $rows AS row MERGE (st:Site {code: row.code}) SET st.name = row.name", {"rows": SITES})
    graph.query("UNWIND $rows AS row MERGE (rt:RoomType {code: row.code}) SET rt.label = row.label", {"rows": ROOM_TYPES})

    graph.query(
        """
        UNWIND $rows AS row
        MERGE (rm:Room {room_code: row.room_code})
        SET rm.site_code     = row.site_code,
            rm.capacity      = row.capacity,
            rm.status        = row.status,
            rm.accessible    = row.accessible,
            rm.has_projector = row.has_projector,
            rm.equipment     = row.equipment
        WITH rm, row
        MATCH (st:Site {code: row.site_code})
        MERGE (st)-[:HAS_ROOM]->(rm)
        WITH rm, row
        MATCH (rt:RoomType {code: row.room_type})
        MERGE (rm)-[:OF_TYPE]->(rt)
        """,
        {"rows": ROOMS},
    )

    graph.query(
        """
        UNWIND $rows AS row
        MERGE (cs:ClassSection {class_code: row.class_code})
        SET cs.course_title   = row.course_title,
            cs.term_code      = row.term_code,
            cs.expected_size  = row.expected_size,
            cs.delivery_mode  = row.delivery_mode,
            cs.class_status   = row.class_status
        """,
        {"rows": CLASS_SECTIONS},
    )

    graph.query(
        """
        UNWIND $rows AS row
        MERGE (cm:ClassMeeting {meeting_id: row.meeting_id})
        SET cm.day_of_week = row.day_of_week,
            cm.start_time  = row.start_time,
            cm.end_time    = row.end_time,
            cm.meeting_type = row.meeting_type,
            cm.status      = row.status
        WITH cm, row
        MATCH (cs:ClassSection {class_code: row.class_code})
        MERGE (cs)-[:HAS_MEETING]->(cm)
        WITH cm, row
        MATCH (rm:Room {room_code: row.room_code})
        MERGE (cm)-[:ASSIGNED_TO]->(rm)
        """,
        {"rows": CLASS_MEETINGS},
    )

    graph.query(
        "UNWIND $rows AS row MERGE (ar:AuthorizationRole {name: row.name}) SET ar.capability = row.capability",
        {"rows": AUTHORIZATION_ROLES},
    )

    counts = {
        "sites": len(SITES),
        "room_types": len(ROOM_TYPES),
        "rooms": len(ROOMS),
        "class_sections": len(CLASS_SECTIONS),
        "class_meetings": len(CLASS_MEETINGS),
        "roles": len(AUTHORIZATION_ROLES),
    }
    log.info("Room/scheduling graph seeded: %s", counts)
    return counts


# -----------------------------------------------------------------------------
# Reusable Cypher query helpers (relationship-aware retrieval)
# -----------------------------------------------------------------------------
def count_entities_by_type(graph: Neo4jGraph) -> List[Dict[str, Any]]:
    """Total node count grouped by primary label."""
    return graph.query(
        """
        MATCH (n)
        RETURN labels(n)[0] AS EntityType, count(n) AS TotalCount
        ORDER BY TotalCount DESC
        """
    )


def sections_affecting_stakeholder(graph: Neo4jGraph, stakeholder: str) -> List[Dict[str, Any]]:
    """Policy sections that affect a given stakeholder (case-insensitive)."""
    return graph.query(
        """
        MATCH (ps:PolicySection)-[:AFFECTS]->(s:Stakeholder)
        WHERE toLower(s.name) CONTAINS toLower($name)
        RETURN s.name AS Stakeholder, collect(ps.title) AS Sections
        """,
        {"name": stakeholder},
    )


def section_commitment_constraints(graph: Neo4jGraph, title: str) -> List[Dict[str, Any]]:
    """Multi-hop traversal section → commitment → measurable constraint."""
    return graph.query(
        """
        MATCH (ps:PolicySection {title: $title})-[:CONTAINS]->(com:Commitment)-[:HAS_CONSTRAINT]->(con:Constraint)
        RETURN com.description AS Commitment, con.metric AS Metric,
               con.value AS Value, con.unit AS Unit, con.period AS Period
        """,
        {"title": title},
    )


def commitments_per_stakeholder(graph: Neo4jGraph) -> List[Dict[str, Any]]:
    """Aggregate commitments across all sections touching each stakeholder."""
    return graph.query(
        """
        MATCH (s:Stakeholder)<-[:AFFECTS]-(ps:PolicySection)-[:CONTAINS]->(com:Commitment)
        RETURN s.name AS Stakeholder, count(DISTINCT com) AS TotalCommitments
        ORDER BY TotalCommitments DESC
        """
    )


def sections_upholding_principle(graph: Neo4jGraph, principle: str) -> List[Dict[str, Any]]:
    """Sections that uphold a given core principle/value."""
    return graph.query(
        """
        MATCH (ps:PolicySection)-[:UPHOLDS]->(p:Principle {name: toLower($name)})
        RETURN p.name AS Principle, collect(ps.title) AS Sections
        """,
        {"name": principle},
    )


def rooms_by_site(graph: Neo4jGraph) -> List[Dict[str, Any]]:
    """Room count and total capacity per teaching site."""
    return graph.query(
        """
        MATCH (st:Site)-[:HAS_ROOM]->(rm:Room)
        RETURN st.code AS Site, st.name AS Name,
               count(rm) AS Rooms, sum(rm.capacity) AS TotalCapacity
        ORDER BY TotalCapacity DESC
        """
    )


def capacity_conflicts(graph: Neo4jGraph) -> List[Dict[str, Any]]:
    """Confirmed in-person classes whose expected size exceeds room capacity."""
    return graph.query(
        """
        MATCH (cs:ClassSection)-[:HAS_MEETING]->(cm:ClassMeeting {status: 'CONFIRMED'})-[:ASSIGNED_TO]->(rm:Room)
        WHERE cs.expected_size > rm.capacity
        RETURN cs.class_code AS Class, cs.expected_size AS ExpectedSize,
               rm.room_code AS Room, rm.capacity AS Capacity
        """
    )


def classes_in_unavailable_rooms(graph: Neo4jGraph) -> List[Dict[str, Any]]:
    """Confirmed meetings assigned to RESTRICTED or MAINTENANCE rooms."""
    return graph.query(
        """
        MATCH (cs:ClassSection)-[:HAS_MEETING]->(cm:ClassMeeting {status: 'CONFIRMED'})-[:ASSIGNED_TO]->(rm:Room)
        WHERE rm.status IN ['RESTRICTED', 'MAINTENANCE']
        RETURN cs.class_code AS Class, cs.course_title AS Course,
               rm.room_code AS Room, rm.status AS RoomStatus
        ORDER BY rm.status, cs.class_code
        """
    )


def rooms_by_status(graph: Neo4jGraph) -> List[Dict[str, Any]]:
    """Operational dashboard: rooms grouped by lifecycle status."""
    return graph.query(
        """
        MATCH (rm:Room)
        RETURN rm.status AS Status, count(rm) AS Rooms
        ORDER BY Rooms DESC
        """
    )


def classes_by_room_type(graph: Neo4jGraph) -> List[Dict[str, Any]]:
    """Class sections grouped by the type of room they meet in."""
    return graph.query(
        """
        MATCH (cs:ClassSection)-[:HAS_MEETING]->(:ClassMeeting)-[:ASSIGNED_TO]->(:Room)-[:OF_TYPE]->(rt:RoomType)
        RETURN rt.code AS RoomType, count(DISTINCT cs) AS Classes
        ORDER BY Classes DESC
        """
    )


if __name__ == "__main__":
    g = get_neo4j_graph()
    build_room_graph(g)
    log.info("Entity counts: %s", count_entities_by_type(g))
    log.info("Rooms by site: %s", rooms_by_site(g))
    log.info("Classes in restricted/maintenance rooms: %s", classes_in_unavailable_rooms(g))
