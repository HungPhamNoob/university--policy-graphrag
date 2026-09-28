"""
task1_schema.py — Domain schema for the UET GraphRAG system.

Source document
---------------
``data/UET_HR.pdf`` — "Human Rights & Academic Community Policy" of
UET · VNU University of Engineering and Technology (Vietnam National
University, Hanoi). The document has two distinct parts and this schema
models both:

  A. Academic Community Policy  (Sections 1–29 + Annex A/B)
     -> extracted from natural-language text with an LLM.

  B. Teaching-Space & Scheduling reference model  (Section C)
     -> a structured CRUD data model (sites, rooms, class sections,
        class meetings, authorization roles) seeded deterministically.

Modeling both parts lets the knowledge graph answer relationship-aware
questions that pure vector search cannot, e.g. "which confirmed classes
are assigned to restricted laboratories?" or "which policy sections that
uphold the *integrity* principle affect *students*?".

Every model below is a Pydantic v2 ``BaseModel`` so that LLM output can be
validated against a strict contract before it ever reaches Neo4j.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

# =============================================================================
# DOMAIN A — Academic Community Policy (LLM-extracted)
# =============================================================================


class StakeholderCategory(str, Enum):
    """Broad grouping of a party inside the UET academic community."""

    STUDENT = "student"                       # undergraduate / graduate students
    ACADEMIC_STAFF = "academic_staff"         # lecturers, supervisors, mentors, instructors
    RESEARCH_STAFF = "research_staff"         # researchers, PIs, research assistants
    PROFESSIONAL_STAFF = "professional_staff"  # administrative / professional / technical staff
    GOVERNANCE = "governance"                 # leadership, faculties, institutes, committees, units
    EXTERNAL = "external"                     # partners, contractors, visitors, research participants
    OTHER = "other"


class RegulationAuthority(str, Enum):
    """Issuing authority of a referenced rule, law or standard."""

    VIETNAMESE_LAW = "vietnamese_law"  # applicable Vietnamese law
    VNU = "vnu"                        # Vietnam National University regulations
    UET = "uet"                        # UET rules / Student Handbook / procedures
    ACADEMIC = "academic"              # course, journal, research-protocol or academic regulations
    OTHER = "other"


class Modality(str, Enum):
    """Normative strength of a commitment, mirrored from the document's wording."""

    MUST = "must"               # mandatory obligation
    SHOULD = "should"           # strong expectation (the document's default)
    MAY = "may"                 # permitted option
    RECOMMENDED = "recommended"  # advised good practice


class ConstraintUnit(str, Enum):
    """Canonical units used to normalize measurable policy constraints."""

    HOURS = "hours"
    DAYS = "days"
    WEEKS = "weeks"
    MONTHS = "months"
    YEARS = "years"
    PERSONS = "persons"
    PERCENT = "percent"
    TIMES = "times"
    LEVEL = "level"
    OTHER = "other"


class ConstraintPeriod(str, Enum):
    """Time window in which a constraint is evaluated or enforced."""

    PER_SESSION = "per_session"
    PER_DAY = "per_day"
    PER_WEEK = "per_week"
    PER_MONTH = "per_month"
    PER_TERM = "per_term"
    PER_YEAR = "per_year"
    NONE = "none"


class Constraint(BaseModel):
    """A measurable limit or requirement attached to a commitment.

    The UET draft deliberately avoids inventing numeric deadlines, so most
    commitments are qualitative. Constraints are populated only when the
    text states an explicit, quantifiable bound.
    """

    metric: str = Field(description="What is measured, e.g. 'review turnaround', 'training hours'")
    value: float = Field(description="Numeric value of the constraint")
    unit: ConstraintUnit = Field(description="Unit of measurement")
    period: ConstraintPeriod = Field(description="Time frame the constraint applies to")


class Commitment(BaseModel):
    """A promise, obligation, expectation or rule stated in a policy section.

    Commitments are usually drawn from the section's "Core expectations"
    bullets. A commitment may be qualitative only, or paired with one or
    more measurable :class:`Constraint` objects.
    """

    description: str = Field(description="Clear, concise statement of the commitment")
    modality: Modality = Field(default=Modality.SHOULD, description="Normative strength (must/should/may/recommended)")
    measurable: bool = Field(default=False, description="True when the commitment carries numeric constraints")
    constraints: List[Constraint] = Field(default_factory=list, description="Measurable limits for this commitment")


class Stakeholder(BaseModel):
    """A party affected by, or responsible under, a policy section."""

    name: str = Field(description="Normalized stakeholder name, e.g. 'Students', 'Supervisors'")
    category: StakeholderCategory = Field(default=StakeholderCategory.OTHER, description="Broad community grouping")


class Regulation(BaseModel):
    """An external law, regulation, handbook or standard referenced by a section."""

    name: str = Field(description="Name of the referenced rule, e.g. 'UET Student Handbook'")
    authority: RegulationAuthority = Field(default=RegulationAuthority.OTHER, description="Issuing authority")


class PolicySectionExtraction(BaseModel):
    """Top-level extraction model for one policy section / annex of the document.

    The fields map directly onto the Neo4j graph schema:

        (PolicySection)-[:AFFECTS]->(Stakeholder)
        (PolicySection)-[:REFERENCES]->(Regulation)
        (PolicySection)-[:UPHOLDS]->(Principle)
        (PolicySection)-[:CONTAINS]->(Commitment)-[:HAS_CONSTRAINT]->(Constraint)
    """

    # Be tolerant of providers that emit numbers where strings are expected
    # (e.g. section_number) so a single coercion does not fail validation.
    model_config = ConfigDict(coerce_numbers_to_str=True)

    section_number: Optional[str] = Field(default=None, description="Section number as printed, e.g. '7' or 'Annex A'")
    title: str = Field(description="Section heading, e.g. 'Student Rights and Responsibilities'")
    subtitle: Optional[str] = Field(default=None, description="One-line tagline under the heading")
    summary: str = Field(description="Brief summary of what the section requires")
    control_objective: Optional[str] = Field(default=None, description="The section's stated 'Control objective'")
    stakeholders: List[Stakeholder] = Field(default_factory=list, description="Parties affected or responsible")
    regulations: List[Regulation] = Field(default_factory=list, description="External rules / standards referenced")
    principles: List[str] = Field(
        default_factory=list,
        description="Core values upheld, e.g. 'dignity', 'fairness', 'integrity', 'safety', 'privacy', 'accessibility'",
    )
    commitments: List[Commitment] = Field(default_factory=list, description="Expectations, obligations and rules")


# =============================================================================
# DOMAIN B — Teaching-Space & Scheduling reference model (Section C, seeded)
# =============================================================================


class RoomType(str, Enum):
    """Controlled category used for room-suitability checks (Section C2.1)."""

    LECTURE = "LECTURE"
    SEMINAR = "SEMINAR"
    COMPUTER_LAB = "COMPUTER_LAB"
    ELECTRONICS_LAB = "ELECTRONICS_LAB"
    ROBOTICS_LAB = "ROBOTICS_LAB"
    PROJECT = "PROJECT"
    MEETING = "MEETING"


class RoomStatus(str, Enum):
    """Operational lifecycle state of a room (Section C1.3)."""

    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"
    MAINTENANCE = "MAINTENANCE"
    RESTRICTED = "RESTRICTED"
    ARCHIVED = "ARCHIVED"


class DeliveryMode(str, Enum):
    """How a class section is delivered (Section C4.1)."""

    IN_PERSON = "IN_PERSON"
    HYBRID = "HYBRID"
    ONLINE = "ONLINE"


class ClassStatus(str, Enum):
    """Lifecycle state of a class section (Section C4.1)."""

    DRAFT = "DRAFT"
    OPEN = "OPEN"
    ACTIVE = "ACTIVE"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"
    ARCHIVED = "ARCHIVED"


class MeetingType(str, Enum):
    """Kind of a scheduled class meeting (Section C4.2)."""

    LECTURE = "LECTURE"
    PRACTICAL = "PRACTICAL"
    TUTORIAL = "TUTORIAL"
    EXAM = "EXAM"
    MAKEUP = "MAKEUP"


class MeetingStatus(str, Enum):
    """Lifecycle state of a class meeting (Section C4.2)."""

    DRAFT = "DRAFT"
    CONFIRMED = "CONFIRMED"
    CANCELLED = "CANCELLED"
    COMPLETED = "COMPLETED"


class Site(BaseModel):
    """A physical UET teaching location / building group (Section C2)."""

    code: str = Field(description="Short site code, e.g. 'GD3', 'KM', 'HL'")
    name: str = Field(description="Human-readable site name")


class Room(BaseModel):
    """Canonical teaching-space record (Section C1.1)."""

    room_code: str = Field(description="Unique display code, e.g. 'GD3-201'")
    site_code: str = Field(description="Owning site code, e.g. 'GD3'")
    room_type: RoomType = Field(description="Controlled room category")
    capacity: int = Field(description="Maximum normal occupancy used by scheduling validation")
    status: RoomStatus = Field(default=RoomStatus.ACTIVE, description="Operational lifecycle state")
    accessible: bool = Field(default=True, description="Baseline physical accessibility support")
    has_projector: bool = Field(default=False, description="Convenience flag for frequently queried equipment")
    equipment: List[str] = Field(default_factory=list, description="Selected equipment profile")


class ClassSection(BaseModel):
    """One teaching instance of a subject in a term (Section C4.1)."""

    class_code: str = Field(description="Human-readable section code, unique within a term, e.g. 'AI-01'")
    course_title: str = Field(description="Displayed academic subject title")
    term_code: str = Field(default="2026-FALL", description="Academic term reference")
    expected_size: int = Field(description="Expected enrollment used by room-capacity checks")
    delivery_mode: DeliveryMode = Field(default=DeliveryMode.IN_PERSON, description="In-person / hybrid / online")
    class_status: ClassStatus = Field(default=ClassStatus.ACTIVE, description="Lifecycle state of the section")


class ClassMeeting(BaseModel):
    """A date/time pattern that assigns a class section to a room (Section C4.2)."""

    meeting_id: str = Field(description="Immutable meeting identifier, e.g. 'MTG-AI-01'")
    class_code: str = Field(description="Parent class section code")
    room_code: str = Field(description="Assigned room code")
    day_of_week: str = Field(description="Controlled weekday, e.g. 'MON'")
    start_time: str = Field(description="HH:MM start, must precede end")
    end_time: str = Field(description="HH:MM end")
    meeting_type: MeetingType = Field(default=MeetingType.LECTURE, description="Lecture / practical / tutorial / exam / makeup")
    status: MeetingStatus = Field(default=MeetingStatus.CONFIRMED, description="Lifecycle state of the meeting")


class AuthorizationRole(BaseModel):
    """Capability-based authorization role over room/scheduling data (Section C7.5)."""

    name: str = Field(description="Role name, e.g. 'Scheduler'")
    capability: str = Field(description="What the role is allowed to do")


# =============================================================================
# Reference example used for prompt design / schema documentation.
# Mirrors Section 7 "Student Rights and Responsibilities" of the UET draft.
# =============================================================================
REFERENCE_EXAMPLE = """
{
  "section_number": "7",
  "title": "Student Rights and Responsibilities",
  "subtitle": "Clear expectations for learning, participation and community conduct",
  "summary": "Students receive timely information and fair treatment, and in turn meet academic requirements, act with integrity and respect others.",
  "control_objective": "Students also have a responsibility to protect the learning environment.",
  "stakeholders": [
    {"name": "Students", "category": "student"},
    {"name": "Teaching staff", "category": "academic_staff"},
    {"name": "Support staff", "category": "professional_staff"}
  ],
  "regulations": [
    {"name": "UET Student Handbook", "authority": "uet"},
    {"name": "Academic regulations", "authority": "academic"}
  ],
  "principles": ["fairness", "integrity", "respect"],
  "commitments": [
    {
      "description": "Students should attend required learning activities and meet deadlines unless an approved exception applies",
      "modality": "should",
      "measurable": false,
      "constraints": []
    },
    {
      "description": "Students must carry out individual and group work honestly and disclose assistance, tools and sources when required",
      "modality": "must",
      "measurable": false,
      "constraints": []
    }
  ]
}
"""
