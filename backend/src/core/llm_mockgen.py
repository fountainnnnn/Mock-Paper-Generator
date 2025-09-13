# backend/src/core/openai_qg.py

from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import os, re, json as _json
from openai import OpenAI

OPENAI_DEFAULT_MODEL = "gpt-4o-mini"

# ------------------------------------------------------------
# OpenAI configuration
# ------------------------------------------------------------
def configure_openai(api_key: Optional[str] = None) -> OpenAI:
    """
    Configure OpenAI client with a required API key.

    Precedence:
      1) function arg `api_key`
      2) env var OPENAI_API_KEY
    """
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OpenAI API key not found. Set it via environment variable OPENAI_API_KEY or pass api_key in the request."
        )
    return OpenAI(api_key=key)


# ------------------------------------------------------------
# Prompting
# ------------------------------------------------------------
MOCKPAPER_SYSTEM = (
    "You are an expert university exam paper generator and formatter. "
    "You will be given the full text of one or more sample exam papers. "
    "Your task is to produce NEW mock exam papers that preserve the structure and format, "
    "cover the same syllabus topics, and keep the target difficulty, while changing all concrete values. "
    "For each generated exam, you must also produce a separate ANSWER KEY document."
)

def _difficulty_guidance(difficulty: str) -> str:
    d = (difficulty or "same").strip().lower()
    if d in {"same", "similar", "default"}:
        return (
            "Keep the overall difficulty the SAME as the reference: same cognitive level, "
            "comparable multi-step depth, and similar distribution of easy/medium/hard items."
        )
    if d in {"easier", "easy"}:
        return (
            "Make the paper EASIER: reduce algebraic complexity, reduce number of steps, "
            "use more guided wording, and avoid edge-case traps, while keeping learning objectives intact."
        )
    if d in {"harder", "hard"}:
        return (
            "Make the paper HARDER: introduce more multi-step reasoning, combine two related concepts, "
            "use less-guided wording, and vary numbers to require careful calculation, without going off-syllabus."
        )
    return f"Difficulty target: {difficulty}. Calibrate question complexity accordingly while staying on the same syllabus."


def build_mockpaper_prompt(paper_text: str, difficulty: str, num_mocks: int) -> str:
    """
    A carefully engineered prompt that forces strict structure mirroring,
    topic fidelity, and controlled novelty in values/context.
    """
    return f"""
{MOCKPAPER_SYSTEM}

Constraints and invariants to PRESERVE exactly:
1) Preserve the NUMBER OF SECTIONS and their ORDER as in the reference.
2) Preserve SECTION HEADERS verbatim (e.g., "Section 1: MCQ", "Section 2: Open ended", "Section 3: Close ended"), including punctuation and casing.
3) Preserve the NUMBER OF QUESTIONS per section and the numbering style (Q1, Q2, ...).
4) Preserve any global instructions header, time limits, marks scheme per section, and any per-question marks labels if present.
5) Preserve formatting conventions used in the reference (line breaks, bulleting, equation style such as LaTeX/inline math, tables if any).
6) Stay strictly within the same syllabus scope and topic coverage.

Required variations to APPLY:
A) Do NOT copy wording. Reframe each question to test the same concept with DIFFERENT numeric values, symbols, names, or application context.
B) If a question contains datasets, equations, or constants, replace them with new ones that still yield a clean, solvable problem of comparable length.
C) For MCQs, always output 4 options labelled (A) (B) (C) (D), with exactly one correct answer and three plausible distractors.
D) If a question is short-answer or essay, provide only the question text in the exam paper. The answer goes ONLY into the answer key.

Difficulty calibration:
{_difficulty_guidance(difficulty)}

Failure cases to avoid:
- Do not add or remove sections.
- Do not change section headers or their order.
- Do not change the count of questions in any section.
- Do not drift off-topic or introduce out-of-syllabus material.
- Do not include answers inside the exam paper itself (except in the separate answer key).
- Do not include meta commentary or notes.

Output format:
- Generate {num_mocks} complete mock exam sets.
- For EACH exam set, output TWO parts in strict order:
  1) "### MOCK PAPER X" followed by the full exam text (with no answers).
  2) "### ANSWER KEY X" followed by the full answer key for that exam.

Reference exam paper (truncated to 12k chars if long):
{paper_text[:12000]}

Now generate {num_mocks} new mock exam(s) with mirrored structure and fresh content, each paired with its own answer key.
""".strip()


# ------------------------------------------------------------
# API call
# ------------------------------------------------------------
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
    """
    if num_mocks < 1:
        num_mocks = 1
    if num_mocks > 3:
        num_mocks = 3

    client = configure_openai(api_key)
    prompt = build_mockpaper_prompt(paper_text, difficulty, num_mocks)

    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": MOCKPAPER_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
    )
    raw = resp.choices[0].message.content.strip() if resp.choices else ""

    # --- Parse into (paper, answers) pairs
    outputs: List[Tuple[str, str]] = []
    current_paper, current_answers = [], []
    mode = None
    idx = 0

    for line in raw.splitlines():
        if line.strip().startswith("### MOCK PAPER"):
            if mode == "answers" and current_paper and current_answers:
                outputs.append(("\n".join(current_paper).strip(), "\n".join(current_answers).strip()))
                current_paper, current_answers = [], []
            mode = "paper"
            idx += 1
            continue
        elif line.strip().startswith("### ANSWER KEY"):
            mode = "answers"
            continue

        if mode == "paper":
            current_paper.append(line)
        elif mode == "answers":
            current_answers.append(line)

    # append last
    if current_paper and current_answers:
        outputs.append(("\n".join(current_paper).strip(), "\n".join(current_answers).strip()))

    return outputs[:num_mocks]
