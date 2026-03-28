from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from flask import Flask, jsonify, render_template, request

from src.voice_input import transcribe_audio
from src.run_pipeline import PREPROCESS

app = Flask(__name__)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_URL = "http://localhost:11434/api/generate"


def _safe_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        return value if value else None
    return str(value).strip() or None


def _extract_json_object(text: str) -> Dict[str, Any]:
    if not text:
        return {}

    text = text.strip()

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {}

    candidate = match.group(0)
    try:
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            return obj
    except Exception:
        return {}

    return {}


def extract_blood_pressure(text: str) -> Optional[str]:
    m = re.search(r"\b(\d{2,3})\s*[/\-]\s*(\d{2,3})\b", text)
    if m:
        sys_val = int(m.group(1))
        dia_val = int(m.group(2))
        if 70 <= sys_val <= 260 and 30 <= dia_val <= 150 and sys_val > dia_val:
            return f"{sys_val}/{dia_val}"
    return None


def extract_heart_rate(text: str) -> Optional[str]:
    patterns = [
        r"\bfc\s*[:=]?\s*(\d{2,3})\b",
        r"\bfrequenza\s*cardiaca\s*[:=]?\s*(\d{2,3})\b",
        r"\bhr\s*[:=]?\s*(\d{2,3})\b",
        r"\b(\d{2,3})\s*bpm\b",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            val = int(m.group(1))
            if 30 <= val <= 220:
                return str(val)
    return None


def extract_temperature(text: str) -> Optional[str]:
    patterns = [
        r"\btemp(?:eratura)?\s*[:=]?\s*(\d{2}[.,]\d)\b",
        r"\b(\d{2}[.,]\d)\s*°\s*c\b",
        r"\b(\d{2}[.,]\d)\s*°c\b",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            value = m.group(1).replace(",", ".")
            try:
                temp = float(value)
                if 33.0 <= temp <= 42.5:
                    return value
            except ValueError:
                pass
    return None


def extract_spo2(text: str) -> Optional[str]:
    patterns = [
        r"\bspo2\s*[:=]?\s*(\d{2,3})\s*%?\b",
        r"\bsaturazione\s*[:=]?\s*(\d{2,3})\s*%?\b",
        r"\bsat\.?\s*[:=]?\s*(\d{2,3})\s*%?\b",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            val = int(m.group(1))
            if 50 <= val <= 100:
                return str(val)
    return None


def infer_reason_for_visit(text: str) -> Optional[str]:
    t = text.lower()

    reason_rules = [
        (["dolore toracico"], "dolore toracico"),
        (["dolore lombare", "lombalgia"], "dolore lombare"),
        (["dolore addominale"], "dolore addominale"),
        (["dispnea", "affanno"], "dispnea"),
        (["febbre"], "febbre"),
        (["medicazione", "ferita", "lesione", "ulcera", "piaga"], "medicazione e controllo lesione"),
        (["controllo parametri", "pressione", "pressione arteriosa", "frequenza cardiaca", "spo2", "saturazione"], "controllo parametri"),
        (["valutazione generale"], "valutazione generale"),
        (["dolore"], "valutazione dolore e controllo parametri"),
        (["terapia", "somministrazione farmaco", "farmaco"], "controllo terapia e somministrazione farmaco"),
    ]

    for keywords, label in reason_rules:
        if any(k in t for k in keywords):
            return label

    return None


def infer_follow_up(text: str) -> Optional[str]:
    t = text.lower()

    patterns = [
        r"(follow[- ]?up\s+tra\s+\d+\s+\w+)",
        r"(rivalutazione\s+tra\s+\d+\s+\w+)",
        r"(controllo\s+tra\s+\d+\s+\w+)",
        r"(tra\s+\d+\s+giorni)",
        r"(tra\s+\d+\s+settimane)",
        r"(tra\s+\d+\s+mesi)",
        r"(nei\s+prossimi\s+\d+\s+\w+)",
        r"(nelle\s+prossime\s+\d+\s+\w+)",
        r"(entro\s+\d+\s+\w+)",
    ]

    for pat in patterns:
        m = re.search(pat, t, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip()

    return None


def infer_interventions(text: str) -> List[str]:
    t = text.lower()
    out: List[str] = []

    if any(k in t for k in ["valutazione generale", "valutato", "eseguita valutazione"]):
        out.append("valutazione generale")

    if any(k in t for k in ["medicazione", "ferita", "lesione", "ulcera", "piaga"]):
        out.append("medicazione")

    if any(k in t for k in ["farmaco", "somministrazione", "somministrato"]):
        out.append("somministrazione farmaco")

    if any(k in t for k in ["pressione", "fc", "frequenza cardiaca", "spo2", "temperatura", "saturazione"]):
        out.append("monitoraggio parametri vitali")

    return list(dict.fromkeys(out))


def infer_critical_issues(text: str) -> List[str]:
    t = text.lower()
    issues: List[str] = []

    if any(k in t for k in ["dolore toracico", "dispnea", "desaturazione"]):
        issues.append("possibile instabilità clinica")

    if any(k in t for k in ["caduta", "post-caduta", "post caduta"]):
        issues.append("caduta recente")

    return list(dict.fromkeys(issues))


def call_llm_extract(text: str) -> Dict[str, Any]:
    prompt = f"""
You are a clinical assistant specialized in ADI (Assistenza Domiciliare Integrata).

Your task is to extract structured medical information from a home-care visit note.

IMPORTANT RULES:
- Return ONLY valid JSON
- No explanation
- No markdown
- No code fences
- Use concise clinical wording
- If information is missing, use null
- Do not invent data

Return exactly this JSON schema:
{{
  "reason_for_visit": null,
  "anamnesis_brief": null,
  "vitals": {{
    "blood_pressure": null,
    "heart_rate": null,
    "temperature": null,
    "spo2": null
  }},
  "follow_up": null,
  "interventions": [],
  "critical_issues": []
}}

EXTRACTION RULES:
- reason_for_visit: MUST be extracted even if implicit.
  Look for symptoms, complaints, or purpose of visit.
  Examples:
  - "dolore toracico"
  - "controllo parametri"
  - "valutazione generale"
  - "medicazione e controllo lesione"
  If nothing is clearly stated, infer the most likely reason from context.
- anamnesis_brief: short summary of symptoms or clinical context
- vitals.blood_pressure: format like "120/80" if present
- vitals.heart_rate: numeric value if present
- vitals.temperature: value if present
- vitals.spo2: oxygen saturation if present
- follow_up: next step, next visit, monitoring plan, or reassessment if mentioned
- interventions: list of actions performed
- critical_issues: list of urgent or clinically relevant issues

Visit note:
{text}
""".strip()

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0,
                    "num_predict": 300,
                },
            },
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        raw_output = data.get("response", "")
    except Exception as e:
        return {"_llm_error": f"Ollama API error: {e}"}

    parsed = _extract_json_object(raw_output)
    if not parsed:
        return {"_llm_error": "Model output could not be parsed as JSON."}

    vitals = parsed.get("vitals") or {}

    interventions = parsed.get("interventions")
    if not isinstance(interventions, list):
        interventions = [str(interventions)] if interventions else []

    critical_issues = parsed.get("critical_issues")
    if not isinstance(critical_issues, list):
        critical_issues = [str(critical_issues)] if critical_issues else []

    return {
        "reason_for_visit": _safe_str(parsed.get("reason_for_visit")),
        "anamnesis_brief": _safe_str(parsed.get("anamnesis_brief")),
        "vitals": {
            "blood_pressure": _safe_str(vitals.get("blood_pressure")),
            "heart_rate": _safe_str(vitals.get("heart_rate")),
            "temperature": _safe_str(vitals.get("temperature")),
            "spo2": _safe_str(vitals.get("spo2")),
        },
        "follow_up": _safe_str(parsed.get("follow_up")),
        "interventions": [str(x).strip() for x in interventions if str(x).strip()],
        "critical_issues": [str(x).strip() for x in critical_issues if str(x).strip()],
    }


def hybrid_extract(text: str) -> Dict[str, Any]:
    llm = call_llm_extract(text)
    llm_vitals = llm.get("vitals", {}) if isinstance(llm, dict) else {}

    bp_rule = extract_blood_pressure(text)
    hr_rule = extract_heart_rate(text)
    temp_rule = extract_temperature(text)
    spo2_rule = extract_spo2(text)

    reason_rule = infer_reason_for_visit(text)
    follow_rule = infer_follow_up(text)
    interventions_rule = infer_interventions(text)
    critical_rule = infer_critical_issues(text)

    reason = llm.get("reason_for_visit") or reason_rule
    anamnesis = llm.get("anamnesis_brief")

    vitals = {
        "blood_pressure": llm_vitals.get("blood_pressure") or bp_rule,
        "heart_rate": llm_vitals.get("heart_rate") or hr_rule,
        "temperature": llm_vitals.get("temperature") or temp_rule,
        "spo2": llm_vitals.get("spo2") or spo2_rule,
    }

    follow_up = llm.get("follow_up") or follow_rule

    interventions = llm.get("interventions", []) or []
    interventions = list(dict.fromkeys(interventions + interventions_rule))

    critical_issues = llm.get("critical_issues", []) or []
    critical_issues = list(dict.fromkeys(critical_issues + critical_rule))

    return {
        "reason_for_visit": reason,
        "anamnesis_brief": anamnesis,
        "vitals": vitals,
        "follow_up": follow_up,
        "interventions": interventions,
        "critical_issues": critical_issues,
        "_llm_error": llm.get("_llm_error"),
    }


def build_output(extracted: Dict[str, Any]) -> Dict[str, Any]:
    warnings: List[str] = []
    missing_fields: List[str] = []

    if not extracted.get("reason_for_visit"):
        missing_fields.append("clinical.reason_for_visit")

    if not extracted.get("follow_up"):
        warnings.append("No follow-up plan detected")

    vitals = extracted.get("vitals", {})
    if not any(vitals.values()):
        warnings.append("No vital signs recorded in note")

    if extracted.get("_llm_error"):
        warnings.append(extracted["_llm_error"])

    return {
        "meta": {
            "visit_datetime": datetime.now().isoformat(timespec="seconds"),
            "operator_role": "infermiere",
            "model": OLLAMA_MODEL,
            "extraction_mode": "hybrid",
        },
        "clinical": {
            "reason_for_visit": extracted.get("reason_for_visit"),
            "anamnesis_brief": extracted.get("anamnesis_brief"),
            "vitals": {
                "blood_pressure": vitals.get("blood_pressure"),
                "heart_rate": vitals.get("heart_rate"),
                "temperature": vitals.get("temperature"),
                "spo2": vitals.get("spo2"),
            },
            "follow_up": extracted.get("follow_up"),
            "interventions": extracted.get("interventions", []),
            "critical_issues": extracted.get("critical_issues", []),
        },
        "quality": {
            "missing_mandatory_fields": missing_fields,
            "warnings": warnings,
        },
    }


@app.route("/")
def home():
    return render_template("login.html")


@app.route("/register")
def register():
    return render_template("register.html")


@app.route("/assistant")
def assistant():
    return render_template("index.html")


@app.route("/process_text", methods=["POST"])
def process_text():
    data = request.get_json(silent=True) or {}
    raw_text = (data.get("text") or "").strip()

    if not raw_text:
        return jsonify({"error": "No text provided"}), 400

    text = PREPROCESS(raw_text)
    extracted = hybrid_extract(text)
    output = build_output(extracted)

    return jsonify({
        "transcript": raw_text,
        "result": output
    })


@app.route("/process_audio", methods=["POST"])
def process_audio():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["audio"]
    if not audio_file.filename:
        return jsonify({"error": "Empty filename"}), 400

    save_path = UPLOAD_DIR / audio_file.filename
    audio_file.save(save_path)

    try:
        raw_transcript = transcribe_audio(str(save_path))
    except Exception as e:
        return jsonify({"error": f"Audio transcription failed: {e}"}), 500

    text = PREPROCESS(raw_transcript)
    extracted = hybrid_extract(text)
    output = build_output(extracted)

    return jsonify({
        "transcript": raw_transcript,
        "result": output
    })


if __name__ == "__main__":
    app.run(debug=True)