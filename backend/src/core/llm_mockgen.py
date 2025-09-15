# backend/src/core/openai_qg.py
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError, RootModel, field_validator

OPENAI_DEFAULT_MODEL = "gpt-4o-mini"


# ============================================================
# OpenAI configuration
# ============================================================
def configure_openai(api_key: Optional[str] = None) -> OpenAI:
    """
    Configure OpenAI client with a required API key.
    """
    key = (api_key or os.environ.get("OPENAI_API_KEY") or "").strip()
    if not key:
        raise RuntimeError(
            "OpenAI API key not found. Set OPENAI_API_KEY or pass api_key=..."
        )
    return OpenAI(api_key=key)


# ============================================================
# Structured output schema (local, lightweight)
# ============================================================
class Asset(BaseModel):
    """Non-text assets the renderer can fetch/generate later (e.g., images)."""
    question_id: str = Field(..., description="ID of the question this asset belongs to")
    kind: str = Field("image", description="Currently only 'image' is supported")
    prompt: str = Field(..., description="Short, precise prompt for the asset generator")


class Question(BaseModel):
    """
    A single question in the spec.
    - text: markdown with LaTeX (inline \( ... \) and block \[ ... \]).
    - type: 'free' | 'mcq' | 'short' | 'proof' | 'calc' (freeform; renderer won't rely on it too much).
    - options: only for MCQ; exactly 4 items if present.
    - correct: for MCQ: 'A'|'B'|'C'|'D' or 0..3
    """
    id: str
    type: str = "free"
    marks: Optional[int] = None
    text: str

    options: Optional[List[str]] = None
    correct: Optional[str | int] = None

    # Optional assets tied to this question (usually images)
    assets: Optional[List[Asset]] = None

    @field_validator("options", mode="before")
    def _strip_empty_options(cls, v):
        if not v:
            return v
        v2 = [str(x).strip() for x in v if str(x).strip()]
        return v2 or None


class Section(BaseModel):
    title: str
    questions: List[Question]


class AnswerItem(BaseModel):
    """Answer key entry."""
    id: str
    answer: str
    workings: Optional[str] = None  # markdown / LaTeX friendly


class MockSpec(BaseModel):
    """One complete mock exam + its key."""
    title: str = "Mock Exam Paper"
    instructions: Optional[str] = None
    sections: List[Section]
    answer_key: List[AnswerItem] = Field(default_factory=list)
    assets: List[Asset] = Field(default_factory=list)

    # convenience: map question IDs -> section index, etc. (not serialized)
    def _qid_set(self) -> set[str]:
        return {q.id for s in self.sections for q in s.questions}

    @field_validator("answer_key")
    def _answers_refer_to_existing_qids(cls, v, values):
        qids = set()
        try:
            sections = values.get("sections") or []
            for s in sections:
                for q in s.questions:
                    qids.add(q.id)
        except Exception:
            pass
        # Do not be strict; we just keep entries that reference valid IDs
        if qids and v:
            keep = [a for a in v if a.id in qids]
            return keep
        return v


class MockSet(RootModel):
    """Top-level payload from the model."""
    root: Dict[str, List[MockSpec]]  # {"mocks": [MockSpec, ...]}

    @property
    def mocks(self) -> List[MockSpec]:
        return self.root.get("mocks", [])


# ============================================================
# Prompting
# ============================================================
MOCKPAPER_SYSTEM = (
    "You are an expert university exam paper generator and formatter. "
    "You output STRICT JSON for each mock exam. "
    "Never include extra commentary, code fences, or explanations—only JSON."
)

def _difficulty_guidance(difficulty: str) -> str:
    d = (difficulty or "same").strip().lower()
    if d in {"same", "similar", "default"}:
        return "Keep difficulty the SAME as the reference."
    if d in {"easier", "easy"}:
        return "Make the paper EASIER: simpler numbers, fewer steps, guided wording."
    if d in {"harder", "hard"}:
        return "Make the paper HARDER: multi-step reasoning, less guidance, and careful distractors."
    return f"Adjust difficulty: {difficulty}."


def build_structured_prompt(paper_text: str, difficulty: str, num_mocks: int) -> str:
    """
    Ask for a structured JSON spec (mocks array).
    - Math must use \( ... \) for inline and \[ ... \] for block.
    - MCQ must have 4 options, labeled implicitly A-D by order.
    - Assets: supply image prompts only when beneficial (diagrams, plots).
    """
    return f"""
You must return a single JSON object of the form:

{{
  "mocks": [
    {{
      "title": "string",
      "instructions": "string (optional)",
      "sections": [
        {{
          "title": "string",
          "questions": [
            {{
              "id": "q1",
              "type": "free|mcq|short|proof|calc",
              "marks": 10,
              "text": "Markdown with LaTeX (inline \\( ... \\), block \\[ ... \\)). For MCQ, write the stem here, not the options.",
              "options": ["A", "B", "C", "D"],   // required for type 'mcq' (exactly 4)
              "correct": "A",                    // 'A'|'B'|'C'|'D' or 0..3 for 'mcq'; omit otherwise
              "assets": [                        // optional; include only when a diagram/plot helps
                 {{"question_id":"q1","kind":"image","prompt":"clean, high-contrast line diagram of ..."}}
              ]
            }}
          ]
        }}
      ],
      "answer_key": [
        {{
          "id": "q1",
          "answer": "Final answer as text/LaTeX.",
          "workings": "Optional derivation/justification in Markdown+LaTeX."
        }}
      ],
      "assets": [] // optional global assets (usually leave empty; prefer question.assets)
    }}
  ]
}}

Hard requirements:
- Produce exactly {num_mocks} mocks.
- Preserve number of sections, ordering, and approximate topical coverage of the reference.
- Keep marks labels and numbering ONLY if present in the reference; otherwise create clear IDs 'q1', 'q2', ...
- For MCQ: always 4 options; make distractors plausible and non-trivial. Put the correct label in 'correct'.
- Use LaTeX \( ... \) and \[ ... \] for math. Do not leak raw dollar signs.
- DO NOT include any headers/footers/page numbers/copyright.
- JSON ONLY—no backticks, no prose.

Difficulty:
{_difficulty_guidance(difficulty)}

Short reference excerpt (<=12k chars):
{paper_text[:12000]}
""".strip()


# ============================================================
# Helpers for robust JSON extraction
# ============================================================
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.S)

def _extract_json(s: str) -> str:
    """
    Extract the first {...} JSON object from a string. Tolerant of model adding prose.
    """
    # Remove code fences if present
    s2 = s.strip()
    if s2.startswith("```"):
        s2 = re.sub(r"^```[a-zA-Z]*\n?", "", s2)
        s2 = s2.rstrip("`").rstrip()
    m = _JSON_OBJECT_RE.search(s2)
    if m:
        return m.group(0)
    return s2  # last resort


def _json_loads_safe(s: str) -> Dict[str, Any]:
    try:
        return json.loads(s)
    except Exception:
        # extremely defensive: try to strip trailing commas and retry
        s2 = re.sub(r",\s*([}\]])", r"\1", s)
        return json.loads(s2)


# ============================================================
# Public: Structured generation
# ============================================================
def generate_mock_specs(
    paper_text: str,
    difficulty: str = "same",
    num_mocks: int = 1,
    model_name: str = OPENAI_DEFAULT_MODEL,
    api_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Generate mocks as structured JSON specs (list of dicts, validated).
    Each dict conforms to MockSpec (title, sections[], answer_key[], assets[]).

    Returns: List[Dict] with len == num_mocks
    """
    num_mocks = max(1, min(num_mocks, 3))
    client = configure_openai(api_key)
    prompt = build_structured_prompt(paper_text, difficulty, num_mocks)

    # Ask for JSON directly
    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": MOCKPAPER_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        response_format={"type": "json_object"},
    )

    raw = resp.choices[0].message.content.strip() if resp.choices else "{}"
    try:
        payload = _json_loads_safe(_extract_json(raw))
    except Exception as e:
        # If for any reason JSON mode errored or model leaked prose, raise a nice error
        raise RuntimeError(f"Model did not return valid JSON: {e}\n--- RAW ---\n{raw[:800]}")

    try:
        mockset = MockSet.model_validate(payload)
    except ValidationError as ve:
        # If validation fails, show a concise error and keep as much as possible
        raise RuntimeError(f"Structured spec failed validation: {ve}") from ve

    # Ensure count
    mocks = list(mockset.mocks)
    if len(mocks) < num_mocks:
        # pad with empty mocks to keep API contract
        while len(mocks) < num_mocks:
            mocks.append(
                MockSpec(sections=[Section(title="Section 1", questions=[])], answer_key=[])
            )
    return [m.model_dump() for m in mocks[:num_mocks]]


# ============================================================
# Back-compat: Flat text generator (paper, answers)
# ============================================================
LEGACY_SYSTEM = (
    "You are an expert exam paper generator. Produce two parts: "
    "first the exam paper (questions only), then the answer key (answers only). "
    "Do not mix answers into the paper."
)

def _build_legacy_prompt(paper_text: str, difficulty: str, num_mocks: int) -> str:
    return f"""
{LEGACY_SYSTEM}

Constraints to PRESERVE:
1. Keep the SAME number of sections and order as the reference.
2. Keep section headers verbatim.
3. Keep the SAME number of questions per section, and numbering style ONLY if present in the reference.
4. Preserve marks labels if present.
5. Preserve formatting (line breaks, math style, bullets).
6. Only include invigilator instructions if they appear in the reference.
7. DO NOT add page numbers, footers, headers, copyrights, or boilerplate.

Required variations:
- Change wording, numbers, datasets, constants, but keep topic and syllabus.
- Do NOT copy text verbatim.
- For MCQ: always 4 options (A)–(D).
- Answers ONLY in the answer key.

Difficulty:
{_difficulty_guidance(difficulty)}

Output format (STRICT):
For each of {num_mocks} mock exams:
### MOCK PAPER X
<full exam text, questions only>
### ANSWER KEY X
<full answer key text, answers only>

Reference exam (first 12k chars shown):
{paper_text[:12000]}
""".strip()


def _render_spec_to_text(spec: Dict[str, Any]) -> Tuple[str, str]:
    """
    Convert one structured spec to (paper_text, answer_key_text).
    - Paper: sections with questions (MCQ options shown as (A)-(D)).
    - Answer key: list id: answer (plus workings).
    """
    m = MockSpec.model_validate(spec)

    paper_lines: List[str] = []
    if m.title:
        paper_lines.append(m.title)
        paper_lines.append("")
    if m.instructions:
        paper_lines.append(m.instructions)
        paper_lines.append("")

    for s_idx, sec in enumerate(m.sections, start=1):
        paper_lines.append(f"{s_idx}. {sec.title}")
        for q in sec.questions:
            marks_str = f" ({q.marks} marks)" if q.marks else ""
            paper_lines.append(f"{q.id}. {q.text}{marks_str}")
            if q.options:
                labels = ["A", "B", "C", "D"]
                for i, opt in enumerate(q.options[:4]):
                    paper_lines.append(f"    ({labels[i]}) {opt}")
            paper_lines.append("")
        paper_lines.append("")

    ans_lines: List[str] = []
    ans_lines.append("Answer Key")
    for a in m.answer_key:
        ans_lines.append(f"{a.id}: {a.answer}")
        if a.workings:
            ans_lines.append(a.workings)
        ans_lines.append("")

    return "\n".join(paper_lines).strip(), "\n".join(ans_lines).strip()


def generate_mock_papers(
    paper_text: str,
    difficulty: str = "same",
    num_mocks: int = 1,
    model_name: str = OPENAI_DEFAULT_MODEL,
    api_key: Optional[str] = None,
) -> List[Tuple[str, str]]:
    """
    Generate 1–3 new mock exam papers (with answers).
    Returns a list of (paper_text, answer_key_text) tuples.

    Implementation notes:
    - First tries the structured JSON path (generate_mock_specs) for better MCQs/diagrams.
    - If anything fails (network/validation), it falls back to the original plain-text prompt.
    """
    # Preferred structured path
    try:
        specs = generate_mock_specs(
            paper_text=paper_text,
            difficulty=difficulty,
            num_mocks=num_mocks,
            model_name=model_name,
            api_key=api_key,
        )
        out: List[Tuple[str, str]] = []
        for spec in specs:
            out.append(_render_spec_to_text(spec))
        return out
    except Exception:
        # Fallback to legacy plain text
        client = configure_openai(api_key)
        prompt = _build_legacy_prompt(paper_text, difficulty, num_mocks)

        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "system", "content": LEGACY_SYSTEM},
                      {"role": "user", "content": prompt}],
            temperature=0.7,
        )
        raw = resp.choices[0].message.content.strip() if resp.choices else ""

        outputs: List[Tuple[str, str]] = []
        current_paper, current_answers = [], []
        mode = None

        for line in raw.splitlines():
            tag = line.strip().lower()
            if tag.startswith("### mock paper"):
                if current_paper or current_answers:
                    outputs.append(("\n".join(current_paper).strip(),
                                    "\n".join(current_answers).strip()))
                    current_paper, current_answers = [], []
                mode = "paper"
                continue
            if tag.startswith("### answer key"):
                mode = "answers"
                continue

            if mode == "paper":
                current_paper.append(line)
            elif mode == "answers":
                current_answers.append(line)

        if current_paper or current_answers:
            outputs.append(("\n".join(current_paper).strip(),
                            "\n".join(current_answers).strip()))

        while len(outputs) < max(1, min(num_mocks, 3)):
            outputs.append(("", ""))
        return outputs[:num_mocks]
