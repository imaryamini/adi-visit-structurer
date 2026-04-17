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
from src.normalize import normalize_interventions, normalize_problems, normalize_reason
from src.italian_numbers import italian_word_to_number, extract_number_from_text

app = Flask(__name__)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Faster local model
OLLAMA_MODEL = "mistral"
OLLAMA_URL = "http://localhost:11434/api/generate"

# If True => no LLM at all
RULE_ONLY_MODE = False

# If True => call LLM only when rule-based result is weak
SMART_LLM_MODE = True


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


def _normalize_text(text: str) -> str:
    text = text.lower()
    text = text.replace("é", "e").replace("è", "e")
    text = re.sub(r"[^\w\s/%.-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------
# Rule-based extraction
# ---------------------------

def extract_blood_pressure(text: str) -> Optional[str]:
    t = _normalize_text(text)

    # 1) classic numeric format: 130/80 or 130-80
    m = re.search(r"\b(\d{2,3})\s*[/\-]\s*(\d{2,3})\b", t)
    if m:
        sys_val = int(m.group(1))
        dia_val = int(m.group(2))
        if 70 <= sys_val <= 260 and 30 <= dia_val <= 150 and sys_val > dia_val:
            return f"{sys_val}/{dia_val}"

    # 2) speech-like numeric format: 130 su 80
    m = re.search(r"\b(\d{2,3})\s+su\s+(\d{2,3})\b", t)
    if m:
        sys_val = int(m.group(1))
        dia_val = int(m.group(2))
        if 70 <= sys_val <= 260 and 30 <= dia_val <= 150 and sys_val > dia_val:
            return f"{sys_val}/{dia_val}"

    # 3) spoken Italian words near "pressione"
    # examples:
    # "pressione centotrenta su ottanta"
    # "pressione cento trenta su ottanta"
    pressure_patterns = [
        r"pressione(?:\s+arteriosa)?\s+([a-z]+(?:\s+[a-z]+)?)\s+su\s+([a-z]+(?:\s+[a-z]+)?)",
        r"pa\s+([a-z]+(?:\s+[a-z]+)?)\s+su\s+([a-z]+(?:\s+[a-z]+)?)",
    ]

    for pat in pressure_patterns:
        m = re.search(pat, t)
        if m:
            left_raw = m.group(1).strip()
            right_raw = m.group(2).strip()

            sys_val = italian_word_to_number(left_raw.replace(" ", ""))
            if sys_val is None:
                sys_val = extract_number_from_text(left_raw, 70, 260)

            dia_val = italian_word_to_number(right_raw.replace(" ", ""))
            if dia_val is None:
                dia_val = extract_number_from_text(right_raw, 30, 150)

            if sys_val is not None and dia_val is not None:
                if 70 <= sys_val <= 260 and 30 <= dia_val <= 150 and sys_val > dia_val:
                    return f"{sys_val}/{dia_val}"

    return None


def extract_heart_rate(text: str) -> Optional[str]:
    t = _normalize_text(text)

    # digits
    patterns = [
        r"\bfc\s*[:=]?\s*(\d{2,3})\b",
        r"\bfrequenza\s*cardiaca\s*[:=]?\s*(\d{2,3})\b",
        r"\bhr\s*[:=]?\s*(\d{2,3})\b",
        r"\b(\d{2,3})\s*bpm\b",
        r"\b(\d{2,3})\s*battiti(?:\s+al\s+minuto)?\b",
    ]
    for pat in patterns:
        m = re.search(pat, t, flags=re.IGNORECASE)
        if m:
            val = int(m.group(1))
            if 30 <= val <= 220:
                return str(val)

    # spoken Italian words
    spoken_patterns = [
        r"frequenza\s+cardiaca\s+([a-z]+(?:\s+[a-z]+)?)",
        r"fc\s+([a-z]+(?:\s+[a-z]+)?)",
        r"([a-z]+(?:\s+[a-z]+)?)\s+battiti(?:\s+al\s+minuto)?",
    ]
    for pat in spoken_patterns:
        m = re.search(pat, t, flags=re.IGNORECASE)
        if m:
            raw = m.group(1).strip()
            val = italian_word_to_number(raw.replace(" ", ""))
            if val is None:
                val = extract_number_from_text(raw, 30, 220)
            if val is not None and 30 <= val <= 220:
                return str(val)

    return None


def extract_temperature(text: str) -> Optional[str]:
    t = _normalize_text(text)

    patterns = [
        r"\btemp(?:eratura)?\s*[:=]?\s*(\d{2}[.,]\d)\b",
        r"\b(\d{2}[.,]\d)\s*°\s*c\b",
        r"\b(\d{2}[.,]\d)\s*°c\b",
    ]
    for pat in patterns:
        m = re.search(pat, t, flags=re.IGNORECASE)
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
    t = _normalize_text(text)

    # digits
    patterns = [
        r"\bspo2\s*[:=]?\s*(\d{2,3})\s*%?\b",
        r"\bsaturazione\s*[:=]?\s*(\d{2,3})\s*%?\b",
        r"\bsat\.?\s*[:=]?\s*(\d{2,3})\s*%?\b",
    ]
    for pat in patterns:
        m = re.search(pat, t, flags=re.IGNORECASE)
        if m:
            val = int(m.group(1))
            if 50 <= val <= 100:
                return str(val)

    # spoken Italian words
    spoken_patterns = [
        r"spo2\s+([a-z]+(?:\s+[a-z]+)?)",
        r"saturazione\s+([a-z]+(?:\s+[a-z]+)?)",
        r"sat\.?\s+([a-z]+(?:\s+[a-z]+)?)",
    ]
    for pat in spoken_patterns:
        m = re.search(pat, t, flags=re.IGNORECASE)
        if m:
            raw = m.group(1).strip()
            val = italian_word_to_number(raw.replace(" ", ""))
            if val is None:
                val = extract_number_from_text(raw, 50, 100)
            if val is not None and 50 <= val <= 100:
                return str(val)

    return None


def infer_reason_for_visit(text: str) -> Optional[str]:
    t = _normalize_text(text)

    reason_rules = [
        (["tosse", "febbre", "dispnea"], "tosse, febbre e lieve dispnea", "all"),
        (["dolore toracico"], "dolore toracico", "any"),
        (["dolore lombare", "lombalgia"], "dolore lombare", "any"),
        (["dolore addominale"], "dolore addominale", "any"),
        (["dispnea", "affanno"], "dispnea", "any"),
        (["febbre"], "febbre", "any"),
        (["medicazione", "ferita", "lesione", "ulcera", "piaga"], "medicazione e controllo lesione", "any"),
        (["controllo parametri", "pressione", "pressione arteriosa", "frequenza cardiaca", "spo2", "saturazione"], "controllo parametri", "any"),
        (["valutazione generale"], "valutazione generale", "any"),
        (["terapia", "somministrazione farmaco", "farmaco"], "controllo terapia e somministrazione farmaco", "any"),
        (["caduta", "post-caduta", "post caduta", "caduta domestica"], "controllo post-caduta", "any"),
        (["dolore"], "valutazione dolore e controllo parametri", "any"),
    ]

    for keywords, label, mode in reason_rules:
        matched = all(k in t for k in keywords) if mode == "all" else any(k in t for k in keywords)
        if matched:
            return normalize_reason(label)

    return None


def infer_follow_up(text: str) -> Optional[str]:
    t = _normalize_text(text)

    patterns = [
        r"(follow[- ]?up\s+tra\s+\d+\s+\w+)",
        r"(rivalutazione\s+tra\s+\d+\s+\w+)",
        r"(controllo\s+tra\s+\d+\s+\w+)",
        r"(nuovo\s+controllo\s+tra\s+\d+\s+\w+)",
        r"(tra\s+\d+\s+giorni)",
        r"(tra\s+\d+\s+settimane)",
        r"(tra\s+\d+\s+mesi)",
        r"(nelle\s+prossime\s+\d+\s+\w+)",
        r"(nei\s+prossimi\s+\d+\s+\w+)",
        r"(entro\s+\d+\s+\w+)",
    ]

    for pat in patterns:
        m = re.search(pat, t, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip()

    return None


def infer_interventions(text: str) -> List[str]:
    t = _normalize_text(text)
    out: List[str] = []

    if any(k in t for k in ["valutazione generale", "valutazione clinica", "eseguita valutazione", "valutato", "ho fatto una valutazione generale"]):
        out.append("valutazione generale")

    if any(k in t for k in ["medicazione", "ferita", "lesione", "ulcera", "piaga"]):
        out.append("medicazione")

    if any(k in t for k in ["farmaco", "somministrazione", "somministrato", "terapia"]):
        out.append("somministrazione farmaco")

    if any(k in t for k in ["pressione", "fc", "frequenza cardiaca", "spo2", "saturazione", "temperatura", "parametri vitali", "monitoraggio parametri"]):
        out.append("monitoraggio parametri vitali")

    return normalize_interventions(out)


def infer_critical_issues(text: str, spo2: Optional[str] = None) -> List[str]:
    t = _normalize_text(text)
    issues: List[str] = []

    has_dyspnea = any(k in t for k in ["dispnea", "affanno"])

    spo2_val = None
    if spo2:
        try:
            spo2_val = int(spo2)
        except Exception:
            spo2_val = None

    if spo2_val is not None and spo2_val < 92:
        issues.append("possibile instabilita respiratoria")

    tachy_patterns = [
        r"\bfc\s*[:=]?\s*(1[1-9]\d|200)\b",
        r"\bfrequenza\s*cardiaca\s*[:=]?\s*(1[1-9]\d|200)\b",
        r"\bhr\s*[:=]?\s*(1[1-9]\d|200)\b",
        r"\b(1[1-9]\d|200)\s*bpm\b",
    ]
    has_tachy = any(re.search(p, t, flags=re.IGNORECASE) for p in tachy_patterns)

    if has_dyspnea and spo2_val is not None and spo2_val < 94:
        issues.append("possibile instabilita clinica")
    elif has_dyspnea and has_tachy:
        issues.append("possibile instabilita clinica")

    if any(k in t for k in ["caduta recente", "recente caduta", "post-caduta", "post caduta", "caduta domestica"]):
        issues.append("caduta recente")

    return list(dict.fromkeys(issues))


# ---------------------------
# LLM extraction
# ---------------------------

def call_llm_extract(text: str) -> Dict[str, Any]:
    prompt = f"""
Extract structured ADI clinical information from this note.

Return ONLY valid JSON.
No explanation.
No markdown.
No code fences.
Be concise.
If missing, use null.

Schema:
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

Clinical note:
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
                    "num_predict": 120,
                },
            },
            timeout=30,
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
        "reason_for_visit": normalize_reason(_safe_str(parsed.get("reason_for_visit"))),
        "anamnesis_brief": _safe_str(parsed.get("anamnesis_brief")),
        "vitals": {
            "blood_pressure": _safe_str(vitals.get("blood_pressure")),
            "heart_rate": _safe_str(vitals.get("heart_rate")),
            "temperature": _safe_str(vitals.get("temperature")),
            "spo2": _safe_str(vitals.get("spo2")),
        },
        "follow_up": _safe_str(parsed.get("follow_up")),
        "interventions": normalize_interventions([str(x).strip() for x in interventions if str(x).strip()]),
        "critical_issues": list(dict.fromkeys([str(x).strip() for x in critical_issues if str(x).strip()])),
    }


# ---------------------------
# Fast hybrid merge
# ---------------------------

def should_call_llm(rule_result: Dict[str, Any]) -> bool:
    if RULE_ONLY_MODE:
        return False

    if not SMART_LLM_MODE:
        return True

    vitals = rule_result.get("vitals", {}) or {}
    enough_vitals = sum(1 for v in vitals.values() if v) >= 2
    has_reason = bool(rule_result.get("reason_for_visit"))
    has_interventions = len(rule_result.get("interventions", [])) > 0

    return not (has_reason and enough_vitals and has_interventions)


def hybrid_extract(text: str) -> Dict[str, Any]:
    bp_rule = extract_blood_pressure(text)
    hr_rule = extract_heart_rate(text)
    temp_rule = extract_temperature(text)
    spo2_rule = extract_spo2(text)

    reason_rule = infer_reason_for_visit(text)
    follow_rule = infer_follow_up(text)
    interventions_rule = infer_interventions(text)
    problems = normalize_problems(text)

    rule_result = {
        "reason_for_visit": reason_rule or "valutazione generale",
        "anamnesis_brief": None,
        "vitals": {
            "blood_pressure": bp_rule,
            "heart_rate": hr_rule,
            "temperature": temp_rule,
            "spo2": spo2_rule,
        },
        "follow_up": follow_rule or "monitoraggio clinico secondo indicazioni",
        "interventions": interventions_rule,
        "critical_issues": infer_critical_issues(text, spo2_rule),
        "problems_normalized": problems,
        "_llm_error": None,
    }

    if not should_call_llm(rule_result):
        return rule_result

    llm = call_llm_extract(text)
    llm_vitals = llm.get("vitals", {}) if isinstance(llm, dict) else {}

    vitals = {
        "blood_pressure": rule_result["vitals"]["blood_pressure"] or llm_vitals.get("blood_pressure"),
        "heart_rate": rule_result["vitals"]["heart_rate"] or llm_vitals.get("heart_rate"),
        "temperature": rule_result["vitals"]["temperature"] or llm_vitals.get("temperature"),
        "spo2": rule_result["vitals"]["spo2"] or llm_vitals.get("spo2"),
    }

    reason = llm.get("reason_for_visit") or rule_result["reason_for_visit"]
    reason = normalize_reason(reason or "valutazione generale")

    anamnesis = llm.get("anamnesis_brief")
    follow_up = llm.get("follow_up") or rule_result["follow_up"]
    interventions = normalize_interventions((llm.get("interventions", []) or []) + rule_result["interventions"])
    critical_issues = rule_result["critical_issues"] or llm.get("critical_issues", []) or []

    return {
        "reason_for_visit": reason,
        "anamnesis_brief": anamnesis,
        "vitals": vitals,
        "follow_up": follow_up,
        "interventions": interventions,
        "critical_issues": critical_issues,
        "problems_normalized": problems,
        "_llm_error": llm.get("_llm_error"),
    }


# ---------------------------
# Output builder
# ---------------------------

def build_output(extracted: Dict[str, Any]) -> Dict[str, Any]:
    warnings: List[str] = []
    missing_fields: List[str] = []

    if not extracted.get("reason_for_visit"):
        missing_fields.append("clinical.reason_for_visit")

    vitals = extracted.get("vitals", {}) or {}
    present_vitals = [key for key, value in vitals.items() if value]

    if not present_vitals:
        warnings.append("No vital signs recorded in note")
    elif len(present_vitals) < 2:
        warnings.append(f"Only partial vital signs detected: {', '.join(present_vitals)}")

    if not extracted.get("interventions"):
        warnings.append("No interventions detected")

    if extracted.get("_llm_error"):
        warnings.append(extracted["_llm_error"])

    return {
        "meta": {
            "visit_datetime": datetime.now().isoformat(timespec="seconds"),
            "operator_role": "infermiere",
            "model": OLLAMA_MODEL if not RULE_ONLY_MODE else "rule-based only",
            "extraction_mode": "fast-hybrid" if not RULE_ONLY_MODE else "rule-only",
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
        "coding": {
            "problems_normalized": extracted.get("problems_normalized", []),
        },
        "quality": {
            "missing_mandatory_fields": missing_fields,
            "warnings": warnings,
        },
    }


# ---------------------------
# Routes
# ---------------------------

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
    finally:
        try:
            save_path.unlink(missing_ok=True)
        except Exception:
            pass

    text = PREPROCESS(raw_transcript)
    extracted = hybrid_extract(text)
    output = build_output(extracted)

    return jsonify({
        "transcript": raw_transcript,
        "result": output
    })


if __name__ == "__main__":
    app.run(debug=True, port=5001)