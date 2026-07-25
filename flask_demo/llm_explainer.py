"""
Plain-language interaction explanations via Groq.

Given a drug pair, its risk tier, and the model's top predicted side-effect
labels (which can be technical/obscure -- e.g. "hyperalimentation"), this
asks Groq for two things:
  1. why_it_matters -- a few short bullets on general clinical significance
     (e.g. "may increase bleeding risk"), not obscure adverse-event labels.
  2. side_effects -- each input label paired with a one-sentence plain-
     language explanation of what it means.

This is explicitly NOT a diagnosis or dosing tool: the prompt tells the
model to stay descriptive/general and non-prescriptive, and the app
degrades gracefully (falls back to raw labels, no crash) if GROQ_API_KEY
isn't set or the call fails, so a missing key never breaks the demo.
"""
import json
import os
from typing import Dict, List, Tuple

import requests

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
# Fast + cheap model, plenty for a few short bullet points.
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")

SYSTEM_PROMPT = (
    "You are a clinical-safety writing assistant. You help translate predicted "
    "drug-drug interaction outputs from a machine-learning model into plain "
    "language for a general audience. You are not diagnosing, prescribing, or "
    "giving personalized medical advice -- you are only explaining, in general "
    "terms, what predicted interaction labels typically mean and why a "
    "combination might be worth discussing with a pharmacist or doctor.\n\n"
    "Respond with ONLY a JSON object, no preamble, no markdown fences, in "
    "exactly this shape:\n"
    '{"why_it_matters": ["...", "...", "..."], '
    '"side_effects": [{"name": "...", "plain_explanation": "..."}]}\n\n'
    "Rules:\n"
    "- why_it_matters: 2-4 short bullets on the GENERAL clinical significance "
    "of combining these two medicines (e.g. increased bleeding risk, added "
    "liver or kidney stress, reduced effectiveness of one drug, increased "
    "drowsiness/sedation). Base these on the kinds of side effects listed, "
    "but describe the underlying concern, not the raw label.\n"
    "- side_effects: one entry per input label, SAME ORDER, giving its plain "
    "-language name and a single-sentence explanation of what it means.\n"
    "- Do not give dosing instructions, timing advice, or tell the user what "
    "to do -- only explain the risk in plain terms and let them know to check "
    "with a professional.\n"
    "- Do not invent side effects that were not in the input list."
)


def explain_interaction(
    drug_a_name: str,
    drug_b_name: str,
    top_relations: List[Tuple[str, float]],
    tier_label: str = "",
) -> Dict:
    """top_relations: list of (label, probability) tuples, already trimmed
    to the top few by the caller (matches what the UI displays)."""
    if not GROQ_API_KEY:
        return _fallback(top_relations, reason="GROQ_API_KEY is not set on the server.")

    relation_list = ", ".join(
        f"{name} (model probability {prob:.2f})" for name, prob in top_relations
    )
    user_prompt = (
        f"Drug pair: {drug_a_name} + {drug_b_name}\n"
        f"Model-flagged risk tier: {tier_label or 'unspecified'}\n"
        f"Top predicted side-effect labels: {relation_list}"
    )

    try:
        resp = requests.post(
            GROQ_API_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.3,
                "max_tokens": 500,
            },
            timeout=15,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()

        # Groq/Llama sometimes wraps JSON in ```json fences despite instructions.
        if content.startswith("```"):
            content = content.strip("`")
            if content.lower().startswith("json"):
                content = content[4:]
        parsed = json.loads(content.strip())

        why = parsed.get("why_it_matters") or []
        effects = parsed.get("side_effects") or []
        if not why or not effects:
            raise ValueError("Groq response missing expected fields")

        return {"why_it_matters": why, "side_effects": effects, "source": "groq"}

    except Exception as e:
        print(f"[llm_explainer] Groq call failed, falling back to raw labels: {e}")
        return _fallback(top_relations, reason="AI explanation unavailable right now.")


def _fallback(top_relations: List[Tuple[str, float]], reason: str) -> Dict:
    """Keeps the feature usable (just less readable) if Groq is unreachable
    or no API key is configured, instead of erroring the whole request out."""
    return {
        "why_it_matters": [
            "This combination was flagged by the model as worth reviewing "
            "with a pharmacist or doctor before taking together."
        ],
        "side_effects": [{"name": name, "plain_explanation": None} for name, _ in top_relations],
        "source": "fallback",
        "note": reason,
    }