# backend/src/core/llm_mockgen.py
# -*- coding: utf-8 -*-
"""
OpenAI-driven mock exam generator (font-safe math).

- Prompts enforce ASCII-safe math (x^2, H2O, pi, theta).
- Unicode math symbols (π, √, ∑, ∫, etc.) are normalized into forms
  that your STIXTwoMath font can render.
- Supports decoding "U+xxxx" escapes into actual Unicode characters.
- Junk glyphs (■, ▮, █, etc.) are stripped to "*".
- MCQs: exactly 4 options, each on its own line (a.–d.) only if present in reference.
- Structured JSON output; legacy plain-text fallback.
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from pydantic import BaseModel, Field, RootModel, field_validator

OPENAI_DEFAULT_MODEL = "gpt-4o-mini"


# ============================================================
# OpenAI configuration
# ============================================================
def configure_openai(api_key: Optional[str] = None) -> OpenAI:
    key = (api_key or os.environ.get("OPENAI_API_KEY") or "").strip()
    if not key:
        raise RuntimeError(
            "OpenAI API key not found. Set OPENAI_API_KEY or pass api_key=..."
        )
    return OpenAI(api_key=key)


# ============================================================
# Unicode cleanup → font-safe
# ============================================================
def decode_unicode_escapes(text: str) -> str:
    """
    Decode sequences like 'U+03C0' into real Unicode characters.
    Works for all valid Unicode code points.
    """
    if not text:
        return text

    def repl(m):
        try:
            codepoint = int(m.group(1), 16)
            return chr(codepoint)
        except Exception:
            return m.group(0)

    return re.sub(r"U\+([0-9A-Fa-f]{4,6})", repl, text)


def normalize_unicode_math(text: str) -> str:
    """
    Normalize text into font-safe math.
    - Decode U+xxxx escapes into characters
    - Keep meaningful math symbols (π, θ, √, ∑, ∫, ∞, etc.)
    - Convert ambiguous ones to ASCII tokens (× → *, ÷ → /, etc.)
    - Replace junk blocks (■, ▮, █, etc.) with "*"
    - Strip combining marks
    """
    if not text:
        return text

    text = decode_unicode_escapes(text)
    norm = unicodedata.normalize("NFKD", text)
    out_chars = []

    replacements = {
        # multiplication/division
        "·": "*", "×": "*", "÷": "/", "⁄": "/",
        # dashes
        "–": "-", "−": "-", "—": "-",
        # greek
        "π": "π", "θ": "θ", "α": "α", "β": "β", "Δ": "Δ", "δ": "δ",
        # math ops
        "√": "√", "∑": "∑", "∫": "∫", "∞": "∞",
        # junk blocks
        "■": "*", "▮": "*", "█": "*", "▪": "*", "▫": "*",
        "◼": "*", "◾": "*", "◽": "*",
    }

    for ch in norm:
        if unicodedata.combining(ch):
            continue
        if ch in replacements:
            out_chars.append(replacements[ch])
        else:
            if 32 <= ord(ch) < 127:
                out_chars.append(ch)  # printable ASCII
            elif 0x2200 <= ord(ch) <= 0x22FF:
                out_chars.append(ch)  # Math operators block
            elif 0x03B1 <= ord(ch) <= 0x03C9:
                out_chars.append(ch)  # Greek lowercase
            elif 0x0391 <= ord(ch) <= 0x03A9:
                out_chars.append(ch)  # Greek uppercase
            else:
                out_chars.append("*")  # safe fallback

    return "".join(out_chars)


# ============================================================
# Structured output schema
# ============================================================
class Asset(BaseModel):
    question_id: str
    kind: str = Field("image")
    prompt: str


class Question(BaseModel):
    id: str
    type: str = "free"
    marks: Optional[int] = None
    text: str
    options: Optional[List[str]] = None
    correct: Optional[str | int] = None
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
    id: str
    answer: str
    workings: Optional[str] = None


class MockSpec(BaseModel):
    title: str = "Mock Exam Paper"
    instructions: Optional[str] = None
    sections: List[Section]
    answer_key: List[AnswerItem] = Field(default_factory=list)
    assets: List[Asset] = Field(default_factory=list)

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
        if qids and v:
            keep = [a for a in v if a.id in qids]
            return keep
        return v


class MockSet(RootModel):
    root: Dict[str, List[MockSpec]]

    @property
    def mocks(self) -> List[MockSpec]:
        return self.root.get("mocks", [])


# ============================================================
# Prompting
# ============================================================
MOCKPAPER_SYSTEM = (
    "You are an expert university exam paper generator and formatter. "
    "You output STRICT JSON for each mock exam. "
    "Never include commentary, code fences, or explanations—only JSON. "
    "All non-ASCII symbols must be output as U+xxxx escapes."
)


def _difficulty_guidance(difficulty: str) -> str:
    d = (difficulty or "same").strip().lower()
    if d in {"same", "similar", "default"}:
        return "Keep difficulty the SAME as the reference."
    if d in {"easier", "easy"}:
        return "Make the paper EASIER."
    if d in {"harder", "hard"}:
        return "Make the paper HARDER."
    return f"Adjust difficulty: {difficulty}."


def build_structured_prompt(paper_text: str, difficulty: str, num_mocks: int) -> str:
    """
    Structured JSON spec.
    - Preserve whether a question is free-response or MCQ.
    - MCQ: exactly 4 options (a.–d.) only if present in reference.
    - Free-response: no options, only text.
    - Math must remain plain ASCII-safe (x^2, H2O, pi, theta).
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
              "type": "free",
              "marks": 5,
              "text": "ASCII-safe math only (x^2, H2O, pi, theta). No LaTeX or Unicode."
            }},
            {{
              "id": "q2",
              "type": "mcq",
              "marks": 5,
              "text": "ASCII-safe math only. No answers here.",
              "options": [
                "a. First option",
                "b. Second option",
                "c. Third option",
                "d. Fourth option"
              ],
              "correct": "b"
            }}
          ]
        }}
      ],
      "answer_key": [
        {{
          "id": "q1",
          "answer": "Final ASCII-safe answer.",
          "workings": "ASCII-safe derivation."
        }},
        {{
          "id": "q2",
          "answer": "b",
          "workings": "Explanation if needed."
        }}
      ]
    }}
  ]
}}

Hard requirements:
- Produce exactly {num_mocks} mocks.
- Preserve the section structure and question types of the reference:
  * If the reference had MCQs, include them with 4 options (a.–d.).
  * If the reference had only open-ended questions, do NOT invent MCQs.
- Math must be ASCII-safe only (x^2, H2O, pi, theta).
- NO Unicode superscripts/subscripts, NO LaTeX.
- Every question MUST appear in answer_key with an answer.
- Do NOT include answers, solutions, or hints inside "questions".
- Do NOT use Markdown tables. If a table is needed, output as plain text rows in pipe-delimited format, e.g. "|col1|col2|col3|".
- JSON ONLY—no prose.

Difficulty:
{_difficulty_guidance(difficulty)}

Reference excerpt (<=12k chars):
{paper_text[:12000]}
""".strip()


# ============================================================
# Helpers for JSON extraction
# ============================================================
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.S)


def _extract_json(s: str) -> str:
    s2 = s.strip()
    if s2.startswith("```"):
        s2 = re.sub(r"^```[a-zA-Z]*\n?", "", s2)
        s2 = s2.rstrip("`").rstrip()
    m = _JSON_OBJECT_RE.search(s2)
    if m:
        return m.group(0)
    return s2


def _json_loads_safe(s: str) -> Dict[str, Any]:
    try:
        return json.loads(s)
    except Exception:
        s2 = re.sub(r",\s*([}\]])", r"\1", s)
        return json.loads(s2)


# ============================================================
# Post-parse ensure: complete answer key
# ============================================================
def _normalize_mcq_correct_label(correct: Any) -> Optional[str]:
    if correct is None:
        return None
    if isinstance(correct, int):
        if 0 <= correct <= 3:
            return "abcd"[correct]
        if 1 <= correct <= 4:
            return "abcd"[correct - 1]
        return None
    c = str(correct).strip().lower()
    if c in {"a", "b", "c", "d"}:
        return c
    if c.isdigit():
        n = int(c)
        if 0 <= n <= 3:
            return "abcd"[n]
        if 1 <= n <= 4:
            return "abcd"[n - 1]
    m = re.match(r"^\(?([abcd])\)?\.?$", c)
    if m:
        return m.group(1)
    m = re.match(r"^\(?([1-4])\)?\.?$", c)
    if m:
        return "abcd"[int(m.group(1)) - 1]
    return None


def _ensure_complete_answer_key(m: MockSpec) -> None:
    existing = {a.id for a in m.answer_key}
    ak = list(m.answer_key)

    for sec in m.sections:
        for q in sec.questions:
            if q.id in existing:
                continue
            if q.options:
                lab = _normalize_mcq_correct_label(q.correct)
                if lab is None:
                    ak.append(AnswerItem(id=q.id, answer="[missing]", workings=None))
                else:
                    ak.append(AnswerItem(id=q.id, answer=lab, workings=None))
            else:
                ak.append(AnswerItem(id=q.id, answer="[missing]", workings=None))
            existing.add(q.id)

    m.answer_key = ak


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
    num_mocks = max(1, min(num_mocks, 3))
    client = configure_openai(api_key)
    prompt = build_structured_prompt(paper_text, difficulty, num_mocks)

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
    payload = _json_loads_safe(_extract_json(raw))
    mockset = MockSet.model_validate(payload)

    mocks = list(mockset.mocks)
    for m in mocks:
        _ensure_complete_answer_key(m)

    if len(mocks) < num_mocks:
        while len(mocks) < num_mocks:
            empty = MockSpec(sections=[Section(title="Section 1", questions=[])], answer_key=[])
            _ensure_complete_answer_key(empty)
            mocks.append(empty)

    return [m.model_dump() for m in mocks[:num_mocks]]


# ============================================================
# Legacy fallback
# ============================================================
LEGACY_SYSTEM = (
    "You are an expert exam paper generator. "
    "Produce two parts: exam paper and answer key. "
    "Math must be ASCII-safe (x^2, H2O, pi, theta). "
    "Never use Unicode superscripts/subscripts or LaTeX. "
    "If the reference had only free-response questions, do not invent MCQs."
)


def _build_legacy_prompt(paper_text: str, difficulty: str, num_mocks: int) -> str:
    return f"""
{LEGACY_SYSTEM}

Constraints:
- Preserve sections, headers, question counts, numbering, and marks.
- Preserve question types: MCQs only if present in reference, otherwise free-response.
- ASCII-safe math only. No Unicode, no LaTeX.
- MCQs: 4 options, each on its own line a.–d.
- Provide answers for ALL questions.
- Do NOT include answers inside questions.
- Do NOT use Markdown tables. Use plain text rows in pipe-delimited format (e.g. "|col1|col2|").

Difficulty:
{_difficulty_guidance(difficulty)}

Output format (STRICT):
For each of {num_mocks} mock exams:
### MOCK PAPER X
<questions>
### ANSWER KEY X
<answers covering EVERY question>

Reference exam (<=12k chars):
{paper_text[:12000]}
""".strip()


# ============================================================
# Renderer: Convert spec → text
# ============================================================
def _render_spec_to_text(spec: Dict[str, Any]) -> Tuple[str, str]:
    m = MockSpec.model_validate(spec)

    paper_lines: List[str] = []
    if m.title:
        paper_lines.append(normalize_unicode_math(m.title))
        paper_lines.append("")
    if m.instructions:
        paper_lines.append(normalize_unicode_math(m.instructions))
        paper_lines.append("")

    for s_idx, sec in enumerate(m.sections, start=1):
        paper_lines.append(normalize_unicode_math(f"{s_idx}. {sec.title}"))
        for q in sec.questions:
            marks_str = f" ({q.marks} marks)" if q.marks else ""
            q_text = normalize_unicode_math(f"{q.id}. {q.text}{marks_str}")
            q_text = re.sub(r"(Answer\s*:.*|Correct\s*:.*)$", "", q_text, flags=re.I).strip()
            paper_lines.append(q_text)
            if q.options:
                for opt in q.options[:4]:
                    paper_lines.append(normalize_unicode_math(opt))
            paper_lines.append("")
        paper_lines.append("")

    ans_lines: List[str] = []
    ans_lines.append("Answer Key")
    for a in m.answer_key:
        ans_lines.append(f"{a.id}: {normalize_unicode_math(a.answer)}")
        if a.workings:
            ans_lines.append(normalize_unicode_math(a.workings))
        ans_lines.append("")

    return "\n".join(paper_lines).strip(), "\n".join(ans_lines).strip()


# ============================================================
# Public wrapper
# ============================================================
def generate_mock_papers(
    paper_text: str,
    difficulty: str = "same",
    num_mocks: int = 1,
    model_name: str = OPENAI_DEFAULT_MODEL,
    api_key: Optional[str] = None,
) -> List[Tuple[str, str]]:
    try:
        specs = generate_mock_specs(
            paper_text=paper_text,
            difficulty=difficulty,
            num_mocks=num_mocks,
            model_name=model_name,
            api_key=api_key,
        )
        return [_render_spec_to_text(spec) for spec in specs]
    except Exception:
        client = configure_openai(api_key)
        prompt = _build_legacy_prompt(paper_text, difficulty, num_mocks)

        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": LEGACY_SYSTEM},
                {"role": "user", "content": prompt},
            ],
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
                    outputs.append((
                        "\n".join(current_paper).strip(),
                        "\n".join(current_answers).strip(),
                    ))
                    current_paper, current_answers = [], []
                mode = "paper"
                continue
            if tag.startswith("### answer key"):
                mode = "answers"
                continue
            if mode == "paper":
                current_paper.append(normalize_unicode_math(line))
            elif mode == "answers":
                current_answers.append(normalize_unicode_math(line))

        if current_paper or current_answers:
            outputs.append((
                "\n".join(current_paper).strip(),
                "\n".join(current_answers).strip(),
            ))

        while len(outputs) < max(1, min(num_mocks, 3)):
            outputs.append(("", ""))
        return outputs[:num_mocks]
