# src/llm_extract.py

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import requests

DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

SYSTEM_PROMPT = (
    "You extract structured clinical data from Italian ADI home-visit notes.\n"
    "Return ONLY valid JSON with this structure:\n"
    '{ "clinical": { "reason_for_visit": null|string, "follow_up": null|string, '
    '"interventions": [], "vitals": { "blood_pressure_systolic": null|number, '
    '"blood_pressure_diastolic": null|number, "heart_rate": null|number, '
    '"temperature": null|number, "spo2": null|number } }, '
    '"coding": { "problems_normalized": [] } }\n'
    "Rules:\n"
    "- Do NOT invent data.\n"
    "- Do NOT confuse dates with blood pressure.\n"
    "- Use null if missing.\n"
    "- Output must be strict JSON (no commentary, no markdown).\n"
)

REPAIR_PROMPT = (
    "You returned INVALID JSON. Fix it.\n"
    "Return ONLY strict JSON with the exact required structure.\n"
    "Do not add any text.\n"
)

def _extract_json_object(text: str) -> str:
    """
    Best-effort extraction of a JSON object if the model adds extra text.
    """
    t = (text or "").strip()
    if "{" in t and "}" in t:
        t = t[t.find("{") : t.rfind("}") + 1]
    return t.strip()


def _call_ollama(prompt: str, model: str, base_url: str, timeout_s: int) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1},
    }
    try:
        r = requests.post(f"{base_url}/api/generate", json=payload, timeout=timeout_s)
    except requests.RequestException as e:
        raise RuntimeError(
            f"Could not reach Ollama at {base_url}.\n"
            "Make sure Ollama is running (e.g. `ollama serve`).\n"
            f"Error: {e}"
        )

    if r.status_code != 200:
        if "model" in r.text.lower() and "not" in r.text.lower():
            raise RuntimeError(
                f"Ollama returned {r.status_code}: {r.text}\n"
                f"Model '{model}' may not be installed.\n"
                f"Try: ollama pull {model}\n"
                "Or set OLLAMA_MODEL to a model you have."
            )
        raise RuntimeError(f"Ollama error {r.status_code}: {r.text}")

    data = r.json()
    return (data.get("response") or "").strip()


def llm_extract(
    text: str,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_OLLAMA_URL,
    timeout_s: int = 90,
    return_raw: bool = False,
    max_retries: int = 1,
) -> Dict[str, Any] | Tuple[Dict[str, Any], str]:
    """
    Local LLM extraction using Ollama.

    - Requests strict JSON.
    - Extracts JSON object if model adds extra text.
    - Retries once with a repair prompt if JSON is invalid.
    - Optionally returns raw model output (return_raw=True).
    """
    prompt = f"{SYSTEM_PROMPT}\n\nTEXT:\n{text}\n\nJSON ONLY:"
    raw = _call_ollama(prompt=prompt, model=model, base_url=base_url, timeout_s=timeout_s)
    raw_json_candidate = _extract_json_object(raw)

    def _try_parse(s: str) -> Optional[Dict[str, Any]]:
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return None

    parsed = _try_parse(raw_json_candidate)

    # Retry with repair if needed
    attempts = 0
    while parsed is None and attempts < max_retries:
        attempts += 1
        repair = (
            f"{REPAIR_PROMPT}\n"
            f"Original required structure:\n{SYSTEM_PROMPT}\n\n"
            f"TEXT:\n{text}\n\n"
            f"Your invalid output:\n{raw}\n\n"
            "JSON ONLY:"
        )
        raw = _call_ollama(prompt=repair, model=model, base_url=base_url, timeout_s=timeout_s)
        raw_json_candidate = _extract_json_object(raw)
        parsed = _try_parse(raw_json_candidate)

    if parsed is None:
        os.makedirs("reports", exist_ok=True)
        with open("reports/llm_raw_output_last.txt", "w", encoding="utf-8") as f:
            f.write(raw)
        raise RuntimeError(
            "Local LLM returned invalid JSON after retries. "
            "Raw output saved to reports/llm_raw_output_last.txt."
        )

    return (parsed, raw) if return_raw else parsed