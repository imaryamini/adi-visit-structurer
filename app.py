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

OLLAMA_MODEL = "mistral"
OLLAMA_URL = "http://localhost:11434/api/generate"

RULE_ONLY_MODE = False
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
    text = re.sub(r"[^\w\s/%.\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_timing_days(text: str) -> Optional[int]:
    t = _normalize_text(text)

    num_map = {
        "uno": 1, "una": 1, "un": 1,
        "due": 2,
        "tre": 3,
        "quattro": 4,
        "cinque": 5,
        "sei": 6,
        "sette": 7,
        "otto": 8,
        "nove": 9,
        "dieci": 10,
        "undici": 11,
        "dodici": 12,
        "quattordici": 14,
        "quindici": 15,
        "venti": 20,
        "trenta": 30,
    }

    patterns = [
        (r"\btra\s+(\d+)\s+giorni\b", 1),
        (r"\bentro\s+(\d+)\s+giorni\b", 1),
        (r"\bnei\s+prossimi\s+(\d+)\s+giorni\b", 1),
        (r"\bnelle\s+prossime\s+(\d+)\s+settimane\b", 7),
        (r"\btra\s+(\d+)\s+settimane\b", 7),
        (r"\btra\s+(\d+)\s+mesi\b", 30),
        (r"\bentro\s+(\d+)\s+settimane\b", 7),
        (r"\bentro\s+(\d+)\s+mesi\b", 30),
    ]

    for pat, mult in patterns:
        m = re.search(pat, t)
        if m:
            return int(m.group(1)) * mult

    word_patterns = [
        (r"\btra\s+(uno|una|un|due|tre|quattro|cinque|sei|sette|otto|nove|dieci|undici|dodici|quattordici|quindici|venti|trenta)\s+giorni\b", 1),
        (r"\btra\s+(uno|una|un|due|tre|quattro|cinque|sei|sette|otto|nove|dieci)\s+settimane\b", 7),
        (r"\btra\s+(uno|una|un|due|tre)\s+mesi\b", 30),
        (r"\bentro\s+(uno|una|un|due|tre|quattro|cinque|sei|sette)\s+giorni\b", 1),
    ]

    for pat, mult in word_patterns:
        m = re.search(pat, t)
        if m:
            return num_map.get(m.group(1), 0) * mult

    if re.search(r"\bnei\s+prossimi\s+giorni\b", t):
        return 3
    if re.search(r"\ba\s+breve\b", t):
        return 3
    if re.search(r"\btra\s+qualche\s+giorno\b", t):
        return 3
    if re.search(r"\btra\s+una\s+settimana\b", t):
        return 7
    if re.search(r"\btra\s+due\s+settimane\b", t):
        return 14
    if re.search(r"\btra\s+un\s+mese\b", t):
        return 30

    return None


# ---------------------------
# Rule-based extraction
# ---------------------------

def extract_blood_pressure(text: str) -> Optional[str]:
    t = _normalize_text(text)

    def valid_bp(sys_val: int, dia_val: int) -> bool:
        return (
            70 <= sys_val <= 260
            and 30 <= dia_val <= 150
            and sys_val > dia_val
        )

    def parse_num(raw: str, min_v: int, max_v: int) -> Optional[int]:
        raw = raw.strip()
        if not raw:
            return None

        m = re.search(r"\b\d{2,3}\b", raw)
        if m:
            value = int(m.group(0))
            if min_v <= value <= max_v:
                return value

        value = italian_word_to_number(raw.replace(" ", ""))
        if value is None:
            value = extract_number_from_text(raw, min_v, max_v)

        if value is not None and min_v <= value <= max_v:
            return value

        return None

    m = re.search(r"\b(\d{2,3})\s*[/\-]\s*(\d{2,3})\b", t)
    if m:
        sys_val = int(m.group(1))
        dia_val = int(m.group(2))
        if valid_bp(sys_val, dia_val):
            return f"{sys_val}/{dia_val}"

    m = re.search(r"\b(\d{2,3})\s+su\s+(\d{2,3})\b", t)
    if m:
        sys_val = int(m.group(1))
        dia_val = int(m.group(2))
        if valid_bp(sys_val, dia_val):
            return f"{sys_val}/{dia_val}"

    m = re.search(r"\b([a-z]+(?:\s+[a-z]+)?)\s+su\s+([a-z]+(?:\s+[a-z]+)?)\b", t)
    if m:
        sys_val = parse_num(m.group(1).strip(), 70, 260)
        dia_val = parse_num(m.group(2).strip(), 30, 150)
        if sys_val is not None and dia_val is not None and valid_bp(sys_val, dia_val):
            return f"{sys_val}/{dia_val}"

    patterns = [
        r"pressione(?:\s+arteriosa)?\s+([a-z0-9]+(?:\s+[a-z0-9]+)?)\s+su\s+([a-z0-9]+(?:\s+[a-z0-9]+)?)",
        r"pa\s+([a-z0-9]+(?:\s+[a-z0-9]+)?)\s+su\s+([a-z0-9]+(?:\s+[a-z0-9]+)?)",
    ]
    for pat in patterns:
        m = re.search(pat, t)
        if m:
            sys_val = parse_num(m.group(1).strip(), 70, 260)
            dia_val = parse_num(m.group(2).strip(), 30, 150)
            if sys_val is not None and dia_val is not None and valid_bp(sys_val, dia_val):
                return f"{sys_val}/{dia_val}"

    systolic = None
    diastolic = None

    max_patterns = [
        r"(?:la\s+)?massima\s*(?:e|è|:)?\s*([a-z0-9]+(?:\s+[a-z0-9]+)?)",
        r"sistolica\s*(?:e|è|:)?\s*([a-z0-9]+(?:\s+[a-z0-9]+)?)",
    ]
    min_patterns = [
        r"(?:la\s+)?minima\s*(?:e|è|:)?\s*([a-z0-9]+(?:\s+[a-z0-9]+)?)",
        r"diastolica\s*(?:e|è|:)?\s*([a-z0-9]+(?:\s+[a-z0-9]+)?)",
    ]

    for pat in max_patterns:
        m = re.search(pat, t)
        if m:
            systolic = parse_num(m.group(1), 70, 260)
            if systolic is not None:
                break

    for pat in min_patterns:
        m = re.search(pat, t)
        if m:
            diastolic = parse_num(m.group(1), 30, 150)
            if diastolic is not None:
                break

    if systolic is not None and diastolic is not None and valid_bp(systolic, diastolic):
        return f"{systolic}/{diastolic}"

    rev_min = re.search(r"(?:la\s+)?minima\s*(?:e|è|:)?\s*([a-z0-9]+(?:\s+[a-z0-9]+)?)", t)
    rev_max = re.search(r"(?:la\s+)?massima\s*(?:e|è|:)?\s*([a-z0-9]+(?:\s+[a-z0-9]+)?)", t)

    if rev_min and rev_max:
        dia_val = parse_num(rev_min.group(1), 30, 150)
        sys_val = parse_num(rev_max.group(1), 70, 260)
        if sys_val is not None and dia_val is not None and valid_bp(sys_val, dia_val):
            return f"{sys_val}/{dia_val}"

    return None


def extract_heart_rate(text: str) -> Optional[str]:
    t = _normalize_text(text)

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

    patterns = [
        r"\bspo2\s*[:=]?\s*(\d{2,3})\s*%?\b",
        r"\bsat(?:urazione)?\.?\s*[:=]?\s*(\d{2,3})\s*%?\b",
        r"\bsaturazione(?:\s+di\s+ossigeno)?\s*[:=]?\s*(\d{2,3})\s*%?\b",
        r"\bo2\s*sat(?:uration)?\s*[:=]?\s*(\d{2,3})\s*%?\b",
        r"\b(\d{2,3})\s*%\s*(?:di\s+)?saturazione\b",
    ]
    for pat in patterns:
        m = re.search(pat, t, flags=re.IGNORECASE)
        if m:
            val = int(m.group(1))
            if 50 <= val <= 100:
                return str(val)

    spoken_patterns = [
        r"\bsaturazione\s+([a-z]+(?:\s+[a-z]+)?)\b",
        r"\bspo2\s+([a-z]+(?:\s+[a-z]+)?)\b",
        r"\bsat\.?\s+([a-z]+(?:\s+[a-z]+)?)\b",
        r"\bossigenazione\s+([a-z]+(?:\s+[a-z]+)?)\b",
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

    generic = re.finditer(r"\b(\d{2,3})\s*(?:%|per\s+cento)\b", t, flags=re.IGNORECASE)
    for m in generic:
        val = int(m.group(1))
        left = t[max(0, m.start() - 40):m.start()]
        right = t[m.end():min(len(t), m.end() + 40)]
        context = f"{left} {right}"
        if any(k in context for k in ["spo2", "sat", "saturazione", "ossigen"]):
            if 50 <= val <= 100:
                return str(val)

    return None


def infer_reason_for_visit(text: str) -> Optional[str]:
    t = _normalize_text(text)

    rules = [
        (["medicazione", "piaga"], "medicazione piaga da decubito", "all"),
        (["medicazione", "decubito"], "medicazione piaga da decubito", "all"),
        (["medicazione", "ulcera"], "medicazione e controllo lesione", "all"),
        (["medicazione", "ferita"], "medicazione e controllo lesione", "all"),
        (["medicazione", "lesione"], "medicazione e controllo lesione", "all"),
        (["controllo terapia"], "controllo terapia/somministrazione farmaco", "any"),
        (["somministrazione farmaco"], "controllo terapia/somministrazione farmaco", "any"),
        (["terapia"], "controllo terapia/somministrazione farmaco", "any"),
        (["farmaco"], "controllo terapia/somministrazione farmaco", "any"),
        (["dolore cronico"], "valutazione dolore cronico", "any"),
        (["dolore", "parametri"], "valutazione dolore e controllo parametri", "all"),
        (["dolore lombare"], "dolore lombare", "any"),
        (["lombalgia"], "dolore lombare", "any"),
        (["dolore addominale"], "dolore addominale", "any"),
        (["dolore toracico"], "dolore toracico", "any"),
        (["stanchezza", "appetito"], "stanchezza e scarso appetito", "all"),
        (["sintomi generali"], "riferiti sintomi generali", "any"),
        (["febbre", "tosse", "dispnea"], "tosse, febbre e lieve dispnea", "all"),
        (["dispnea"], "dispnea", "any"),
        (["febbre"], "febbre", "any"),
        (["caduta"], "controllo post-caduta", "any"),
        (
            ["controllo parametri", "pressione", "frequenza cardiaca", "spo2", "saturazione", "temperatura"],
            "controllo parametri",
            "any",
        ),
        (["valutazione generale"], "valutazione generale", "any"),
        (["controllo generale"], "controllo generale", "any"),
    ]

    for keywords, label, mode in rules:
        matched = all(k in t for k in keywords) if mode == "all" else any(k in t for k in keywords)
        if matched:
            return normalize_reason(label)

    return None


def infer_follow_up(text: str) -> Optional[Dict[str, Any]]:
    t = _normalize_text(text)

    if any(k in t for k in ["ricontatto telefonico", "contatto telefonico", "follow up telefonico", "controllo telefonico"]):
        return {"type": "ricontatto_telefonico", "timing_days": _extract_timing_days(t), "target": None}

    if any(k in t for k in ["ferita", "lesione", "ulcera", "piaga", "medicazione"]) and any(
        k in t for k in ["rivalutare", "ricontrollare", "controllo", "medicazione successiva"]
    ):
        return {"type": "controllo_ferita", "timing_days": _extract_timing_days(t)}

    follow_patterns = [
        r"\brivalutazione\b",
        r"\bricontrollo\b",
        r"\bcontrollo\b",
        r"\bnuova visita\b",
        r"\bfollow[- ]?up\b",
        r"\bmonitoraggio\b",
        r"\bda rivalutare\b",
        r"\bda ricontrollare\b",
        r"\ba breve\b",
        r"\bnei prossimi giorni\b",
        r"\bnelle prossime settimane\b",
        r"\btra qualche giorno\b",
        r"\btra una settimana\b",
        r"\btra due settimane\b",
        r"\btra un mese\b",
        r"\bentro\b",
        r"\bprossimo accesso\b",
        r"\bvisita successiva\b",
        r"\bprossima visita\b",
    ]

    if any(re.search(p, t) for p in follow_patterns):
        return {"type": "controllo", "timing_days": _extract_timing_days(t)}

    return None


def infer_interventions(text: str) -> List[str]:
    t = _normalize_text(text)
    out: List[str] = []

    if any(k in t for k in [
        "valutazione generale",
        "valutazione clinica",
        "eseguita valutazione",
        "valutato",
        "ho fatto una valutazione generale",
        "visita di controllo",
        "controllo generale",
    ]):
        out.append("valutazione generale")

    if any(k in t for k in [
        "medicazione",
        "eseguita medicazione",
        "controllo lesione",
        "ferita",
        "lesione",
        "ulcera",
        "piaga",
        "decubito",
    ]):
        out.append("medicazione")

    if any(k in t for k in [
        "farmaco",
        "somministrazione",
        "somministrato",
        "somministrata terapia",
        "terapia",
        "controllo terapia",
    ]):
        out.append("somministrazione farmaco")

    if any(k in t for k in [
        "pressione",
        "fc",
        "frequenza cardiaca",
        "spo2",
        "saturazione",
        "temperatura",
        "parametri vitali",
        "monitoraggio parametri",
        "monitoraggio dei parametri",
        "monitoraggio parametri vitali",
        "rilevati parametri",
        "controllo parametri",
    ]):
        out.append("monitoraggio parametri vitali")

    if any(k in t for k in [
        "educazione caregiver",
        "istruito il caregiver",
        "consigli al caregiver",
        "educazione sanitaria",
    ]):
        out.append("educazione caregiver")

    if any(k in t for k in [
        "controllo saturazione",
        "monitoraggio saturazione",
        "valutazione respiratoria",
        "controllo respiratorio",
    ]):
        out.append("monitoraggio respiratorio")

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

For follow_up:
- if possible return an object like:
  {{
    "type": "controllo",
    "timing_days": 7
  }}
- if timing is not clear, timing_days can be null

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
                    "num_predict": 140,
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

    follow_up = parsed.get("follow_up")
    if isinstance(follow_up, str):
        follow_up = {"type": follow_up.strip(), "timing_days": _extract_timing_days(follow_up)}
    elif isinstance(follow_up, dict):
        follow_up = {
            "type": _safe_str(follow_up.get("type")),
            "timing_days": follow_up.get("timing_days"),
            **({"target": _safe_str(follow_up.get("target"))} if follow_up.get("target") is not None else {}),
        }
    else:
        follow_up = None

    return {
        "reason_for_visit": normalize_reason(_safe_str(parsed.get("reason_for_visit"))),
        "anamnesis_brief": _safe_str(parsed.get("anamnesis_brief")),
        "vitals": {
            "blood_pressure": _safe_str(vitals.get("blood_pressure")),
            "heart_rate": _safe_str(vitals.get("heart_rate")),
            "temperature": _safe_str(vitals.get("temperature")),
            "spo2": _safe_str(vitals.get("spo2")),
        },
        "follow_up": follow_up,
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
    has_follow_up = bool(rule_result.get("follow_up"))

    return not (has_reason and enough_vitals and has_interventions and has_follow_up)


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
        "reason_for_visit": reason_rule,
        "anamnesis_brief": None,
        "vitals": {
            "blood_pressure": bp_rule,
            "heart_rate": hr_rule,
            "temperature": temp_rule,
            "spo2": spo2_rule,
        },
        "follow_up": follow_rule,
        "interventions": interventions_rule,
        "critical_issues": infer_critical_issues(text, spo2_rule),
        "problems_normalized": problems,
        "_llm_error": None,
    }

    if not should_call_llm(rule_result):
        if not rule_result["reason_for_visit"]:
            rule_result["reason_for_visit"] = normalize_reason("valutazione generale")
        if not rule_result["follow_up"]:
            rule_result["follow_up"] = {"type": "controllo", "timing_days": None}
        return rule_result

    llm = call_llm_extract(text)
    llm_vitals = llm.get("vitals", {}) if isinstance(llm, dict) else {}

    vitals = {
        "blood_pressure": rule_result["vitals"]["blood_pressure"] or llm_vitals.get("blood_pressure"),
        "heart_rate": rule_result["vitals"]["heart_rate"] or llm_vitals.get("heart_rate"),
        "temperature": rule_result["vitals"]["temperature"] or llm_vitals.get("temperature"),
        "spo2": rule_result["vitals"]["spo2"] or llm_vitals.get("spo2"),
    }

    reason = rule_result["reason_for_visit"] or llm.get("reason_for_visit")
    if not reason:
        reason = "valutazione generale"
    reason = normalize_reason(reason)

    anamnesis = llm.get("anamnesis_brief")
    follow_up = rule_result["follow_up"] or llm.get("follow_up") or {"type": "controllo", "timing_days": None}
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