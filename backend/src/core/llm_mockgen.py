# backend/src/core/openai_qg.py

from typing import Any, Dict, List, Optional, Tuple
import os
from openai import OpenAI

OPENAI_DEFAULT_MODEL = "gpt-4o-mini"

# ------------------------------------------------------------
# OpenAI configuration
# ------------------------------------------------------------
def configure_openai(api_key: Optional[str] = None) -> OpenAI:
    """
    Configure OpenAI client with a required API key.
    """
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OpenAI API key not found. Set OPENAI_API_KEY env var or pass api_key."
        )
    return OpenAI(api_key=key)

# ------------------------------------------------------------
# Prompting
# ------------------------------------------------------------
MOCKPAPER_SYSTEM = (
    "You are an expert university exam paper generator. "
    "You strictly produce two parts for each mock set: "
    "first the exam paper (questions only), then the answer key (answers only). "
    "You must not mix answers into the exam paper. "
    "Never include numbering unless it already exists in the reference. "
    "Never output footers, headers, page numbers, copyright notices, confidentiality labels, or boilerplate text. "
    "Focus only on exam questions and answer key content."
)

def _difficulty_guidance(difficulty: str) -> str:
    d = (difficulty or "same").strip().lower()
    if d in {"same", "similar", "default"}:
        return "Keep difficulty the SAME as the reference."
    if d in {"easier", "easy"}:
        return "Make the paper EASIER: simpler numbers, fewer steps, guided wording."
    if d in {"harder", "hard"}:
        return "Make the paper HARDER: multi-step reasoning, less guidance, trickier numbers."
    return f"Adjust difficulty: {difficulty}."

def build_mockpaper_prompt(paper_text: str, difficulty: str, num_mocks: int) -> str:
    return f"""
{MOCKPAPER_SYSTEM}

Constraints to PRESERVE:
1. Keep the SAME number of sections and order as the reference.
2. Keep section headers verbatim.
3. Keep the SAME number of questions per section, and numbering style ONLY if present in the reference.
4. Preserve marks labels if present.
5. Preserve formatting (line breaks, math style, bullets).
6. Only include invigilator instructions if they appear in the reference.
7. DO NOT add page numbers, footers, headers, copyrights, or confidentiality notes.

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
<full exam text, questions only, without any extra boilerplate>
### ANSWER KEY X
<full answer key text, answers only, without any extra boilerplate>

Reference exam (first 12k chars shown):
{paper_text[:12000]}
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
    num_mocks = max(1, min(num_mocks, 3))
    client = configure_openai(api_key)
    prompt = build_mockpaper_prompt(paper_text, difficulty, num_mocks)

    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "system", "content": MOCKPAPER_SYSTEM},
                  {"role": "user", "content": prompt}],
        temperature=0.7,
    )
    raw = resp.choices[0].message.content.strip() if resp.choices else ""

    # --- Parse output into pairs
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

    while len(outputs) < num_mocks:
        outputs.append(("", ""))
    return outputs[:num_mocks]
