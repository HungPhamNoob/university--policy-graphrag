"""
task2_extraction.py — Entity & relationship extraction for the University Policy GraphRAG system.

Pipeline
--------
1. Load ``data/UET_HR.pdf`` and keep only the *policy* pages (Sections 1–29
   plus Annex A/B). Section C (room/scheduling tables) is seeded
   deterministically in ``task3_graph.py`` instead of being LLM-extracted.
2. Clean repeated page furniture (running header/footer and the boilerplate
   "MINIMUM STANDARD" / "USE OF ANNEX" blocks) so each chunk is one section.
3. Extract a :class:`PolicySectionExtraction` per section with an LLM. Two
   strategies are tried, most-reliable first:
       a. ``with_structured_output`` (provider-enforced schema), then
       b. a plain JSON completion repaired with ``json_repair`` and validated
          by Pydantic — a fallback for OpenAI-compatible gateways whose
          tool-calling route is flaky.
4. Log extraction statistics; isolate per-section errors and retry transient
   provider failures (HTTP 502 / timeouts) so one bad response never kills the run.

Run with::

    uv run python task2_extraction.py
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, List, Optional, Tuple

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from task1_schema import REFERENCE_EXAMPLE, PolicySectionExtraction

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("graphrag.extraction")

# The draft repeats these blocks verbatim at the foot of every policy page.
_BOILERPLATE_PATTERNS = [
    r"MINIMUM STANDARD\s+The responsible unit.*?not sufficient\.",
    r"USE OF ANNEX\s+These examples and checklists.*?control document\.",
]
_PAGE_FURNITURE = [
    "UET VNU UNIVERSITY OF ENGINEERING",
    "AND TECHNOLOGY",
    "Draft reference document",
    "Prepared 26-Sep-2026",
    "University of Engineering and Technology - Vietnam National University, Hanoi",
    "uet.vnu.edu.vn",
]

SYSTEM_PROMPT = """You are an expert higher-education policy analyst and knowledge-graph engineer.
Extract structured information from one section of a university "Human Rights & Academic Community Policy".

Return a single JSON object with exactly these keys:
- section_number: string|null  (printed number "1".."29", or "Annex A"/"Annex B")
- title: string                (section heading)
- subtitle: string|null        (one-line tagline under the heading)
- summary: string              (1–2 sentences: what the section requires)
- control_objective: string|null (the text under "Control objective")
- stakeholders: array of {name, category}
    category ∈ student | academic_staff | research_staff | professional_staff | governance | external | other
    (Students→student; Lecturers/Supervisors/Mentors→academic_staff; Researchers/PIs→research_staff;
     Administrative/technical staff→professional_staff; Faculties/Committees/University leadership→governance;
     Partners/Contractors/Visiting scholars/Research participants→external)
- regulations: array of {name, authority}
    authority ∈ vietnamese_law | vnu | uet | academic | other
- principles: array of lowercase value words (dignity, fairness, integrity, safety, privacy,
    accessibility, accountability, freedom, inclusion, transparency, respect, responsibility, ...)
- commitments: array of {description, modality, measurable, constraints}
    modality ∈ must | should | may | recommended   (taken from the section's wording)
    measurable: bool; constraints: array of {metric, value:number, unit, period}
    unit ∈ hours|days|weeks|months|years|persons|percent|times|level|other
    period ∈ per_session|per_day|per_week|per_month|per_term|per_year|none

RULES:
- DO NOT invent facts, offices, numbers, deadlines or penalties absent from the text.
- The draft intentionally avoids numeric limits, so most commitments are qualitative — that is expected
  (measurable=false, constraints=[]).
- Use null for missing scalars and [] for missing lists. Merge duplicate stakeholders/regulations.
- Return ONLY the JSON object. No markdown fences, no commentary."""


def _clean_page_text(text: str) -> str:
    """Strip running header/footer lines, page numbers and repeated boilerplate."""
    for pattern in _BOILERPLATE_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.DOTALL)

    kept: List[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if re.fullmatch(r"\d+\s*/\s*\d+", stripped):  # e.g. "7/43"
            continue
        if any(f in stripped for f in _PAGE_FURNITURE):
            continue
        kept.append(line)
    return re.sub(r"[ \t]+", " ", "\n".join(kept)).strip()


def load_policy_sections(file_path: str, first_page: int = 4, last_page: int = 34) -> List[str]:
    """Return one cleaned text block per policy section/annex page.

    ``first_page`` / ``last_page`` are 1-indexed and inclusive. The default
    range covers Sections 1–29 (pages 4–32) and Annex A/B (pages 33–34),
    excluding the cover/contents pages and the Section C room tables.
    """
    from pypdf import PdfReader

    reader = PdfReader(file_path)
    total = len(reader.pages)
    lo, hi = max(1, first_page), min(total, last_page)

    sections: List[str] = []
    for page_no in range(lo, hi + 1):
        cleaned = _clean_page_text(reader.pages[page_no - 1].extract_text() or "")
        if cleaned:
            sections.append(cleaned)
    log.info("Loaded %d policy section chunks from pages %d–%d of '%s'", len(sections), lo, hi, file_path)
    return sections


def build_extractors(
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.0,
    timeout: int = 240,
) -> Tuple[Any, ChatOpenAI]:
    """Return ``(structured_chain, plain_llm)`` for the hybrid extraction strategy."""
    model_name = model_name or os.environ.get("LLM_MODEL", "nexai")
    api_key = api_key or os.environ.get("API_KEY_2")
    base_url = base_url or os.environ.get("BASE_URL_2")

    common = dict(model=model_name, api_key=api_key, base_url=base_url,
                  temperature=temperature, timeout=timeout, max_retries=1)

    structured_chain = (
        ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                ("human", "Example output for one section:\n{example}\n\nNow extract from this section text:\n{section_text}"),
            ]
        ).partial(example=REFERENCE_EXAMPLE.strip())
        | ChatOpenAI(**common).with_structured_output(PolicySectionExtraction)
    )
    plain_llm = ChatOpenAI(**common)
    return structured_chain, plain_llm


def _coerce_json(raw: str) -> PolicySectionExtraction:
    """Repair + parse a plain-text JSON completion into the Pydantic model."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        text = text[start : end + 1]
    try:
        from json_repair import repair_json  # type: ignore

        data = json.loads(repair_json(text))
    except Exception:  # noqa: BLE001 — fall back to strict parsing
        data = json.loads(text)
    return PolicySectionExtraction.model_validate(data)


def _extract_one(section_text: str, structured_chain: Any, plain_llm: ChatOpenAI) -> PolicySectionExtraction:
    """Try schema-enforced structured output, then a repaired plain-JSON fallback."""
    try:
        return structured_chain.invoke({"section_text": section_text})
    except ValidationError:
        raise
    except Exception as structured_err:  # noqa: BLE001 — provider/tool route failed
        log.debug("Structured output failed (%s); trying plain-JSON fallback", type(structured_err).__name__)
        raw = plain_llm.invoke([("system", SYSTEM_PROMPT), ("human", section_text)]).content
        if isinstance(raw, list):  # some providers return content blocks
            raw = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in raw)
        return _coerce_json(str(raw))


def _extract_with_retry(
    section_text: str,
    structured_chain: Any,
    plain_llm: ChatOpenAI,
    attempts: int = 6,
    backoff: float = 6.0,
    max_backoff: float = 30.0,
) -> PolicySectionExtraction:
    """Retry transient provider errors (502/timeout) with capped exponential backoff."""
    last_err: Optional[Exception] = None
    for attempt in range(1, attempts + 1):
        try:
            return _extract_one(section_text, structured_chain, plain_llm)
        except ValidationError:
            raise  # schema failures are not transient
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if attempt < attempts:
                wait = min(backoff * attempt, max_backoff)
                log.warning("Attempt %d/%d failed (%s); retrying in %.0fs", attempt, attempts, type(exc).__name__, wait)
                time.sleep(wait)
    assert last_err is not None
    raise last_err


def extract_sections_and_log(
    sections: List[str], structured_chain: Any, plain_llm: ChatOpenAI
) -> List[PolicySectionExtraction]:
    """Extract every section, validating against the schema and logging statistics."""
    extractions: List[PolicySectionExtraction] = []
    stats = dict(total=len(sections), ok=0, bad_schema=0, bad_runtime=0,
                 stakeholders=0, regulations=0, principles=0, commitments=0, constraints=0)

    for i, section_text in enumerate(sections, start=1):
        try:
            result = _extract_with_retry(section_text, structured_chain, plain_llm)
        except ValidationError as exc:
            log.error("Section %d failed schema validation: %s", i, exc)
            stats["bad_schema"] += 1
            continue
        except Exception as exc:  # noqa: BLE001
            log.error("Section %d failed after retries: %s", i, exc)
            stats["bad_runtime"] += 1
            continue

        extractions.append(result)
        stats["ok"] += 1
        stats["stakeholders"] += len(result.stakeholders)
        stats["regulations"] += len(result.regulations)
        stats["principles"] += len(result.principles)
        stats["commitments"] += len(result.commitments)
        stats["constraints"] += sum(len(c.constraints) for c in result.commitments)
        log.info("[%2d/%2d] %-52s → %d sh, %d rg, %d com",
                 i, len(sections), (result.title or "")[:52],
                 len(result.stakeholders), len(result.regulations), len(result.commitments))

    log.info("--- EXTRACTION STATISTICS ---")
    log.info("Sections : %d total | %d ok | %d schema-failed | %d runtime-failed",
             stats["total"], stats["ok"], stats["bad_schema"], stats["bad_runtime"])
    log.info("Entities : %d stakeholders, %d regulations, %d principles",
             stats["stakeholders"], stats["regulations"], stats["principles"])
    log.info("Relations: %d commitments, %d constraints", stats["commitments"], stats["constraints"])
    log.info("-----------------------------")
    return extractions


if __name__ == "__main__":
    file_path = os.environ.get("FILE_PATH", "data/UET_HR.pdf")
    policy_sections = load_policy_sections(file_path)
    structured, plain = build_extractors(model_name=os.environ.get("LLM_MODEL", "nexai"))
    extracted = extract_sections_and_log(policy_sections, structured, plain)
    log.info("Extraction complete: %d validated PolicySectionExtraction objects.", len(extracted))
