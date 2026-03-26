import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import src.preprocess as preprocess_mod
import src.extract_rules as rules_mod
from src.normalize import normalize_problems
from src.quality import quality_check
from src.schema import coerce_llm_output
from src.problem_evidence import has_evidence

try:
    from src.llm_extract import llm_extract
except Exception:
    llm_extract = None


RAW_DIR = Path("data/synthetic/raw")
PRED_DIR = Path("data/synthetic/pred")
REPORTS_DIR = Path("reports")
EXAMPLES_DIR = REPORTS_DIR / "examples"
DEBUG_LOG_PATH = REPORTS_DIR / "hybrid_debug.log"
RUN_SUMMARY_PATH = REPORTS_DIR / "run_summary.json"

PIPELINE_VERSION = "0.4.0"
MAX_EXAMPLE_EXPORTS = 3

INTERVENTION_VOCAB = {
    "monitoraggio_parametri_vitali",
    "valutazione_generale",
    "consigli_alimentari",
    "educazione_alimentare",
    "medicazione",
    "somministrazione_farmaco",
    "monitoraggio_glicemia",
    "gestione_catetere",
    "gestione_stomia",
    "educazione_terapeutica",
}


def log_debug(message: str) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().isoformat(timespec="seconds")
    with DEBUG_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")


def _pick_callable(module, names: list[str]) -> Optional[Callable]:
    for n in names:
        fn = getattr(module, n, None)
        if callable(fn):
            return fn
    return None


def get_preprocess_fn() -> Callable[[str], str]:
    candidates = [
        "preprocess_text",
        "preprocess",
        "clean_text",
        "clean",
        "normalize_text",
        "prepare_text",
    ]
    fn = _pick_callable(preprocess_mod, candidates)
    if fn:
        return fn

    available = [x for x in dir(preprocess_mod) if not x.startswith("_")]
    raise ImportError(
        "Could not find a preprocess function in src/preprocess.py.\n"
        f"Tried: {candidates}\n"
        f"Available: {available}"
    )


PREPROCESS = get_preprocess_fn()

EXTRACT_REASON = _pick_callable(
    rules_mod,
    ["extract_reason", "extract_reason_for_visit", "reason_for_visit", "get_reason"],
)
EXTRACT_FOLLOW_UP = _pick_callable(
    rules_mod,
    ["extract_follow_up", "extract_followup", "follow_up", "get_follow_up"],
)
EXTRACT_INTERVENTIONS = _pick_callable(
    rules_mod,
    ["extract_interventions", "extract_actions", "interventions", "get_interventions"],
)
EXTRACT_BP = _pick_callable(
    rules_mod,
    ["extract_blood_pressure", "extract_bp", "blood_pressure", "get_bp"],
)
EXTRACT_HR = _pick_callable(
    rules_mod,
    ["extract_heart_rate", "extract_hr", "heart_rate", "get_hr"],
)
EXTRACT_TEMP = _pick_callable(
    rules_mod,
    ["extract_temperature", "extract_temp", "temperature", "get_temp"],
)
EXTRACT_SPO2 = _pick_callable(
    rules_mod,
    ["extract_spo2", "extract_saturation", "spo2", "saturation", "get_spo2"],
)
EXTRACT_DATETIME = _pick_callable(
    rules_mod,
    ["extract_datetime", "get_datetime", "extract_visit_datetime"],
)


_BP_CONTEXT = re.compile(r"\b(pa|pressione|press\.?|bp)\b", re.IGNORECASE)
_HR_CONTEXT = re.compile(r"\b(fc|frequenza\s*cardiaca|hr|bpm)\b", re.IGNORECASE)
_TEMP_CONTEXT = re.compile(r"\b(temp(?:eratura)?|t)\b", re.IGNORECASE)
_SPO2_CONTEXT = re.compile(r"\b(spo2|saturazione|sat\.?)\b", re.IGNORECASE)


def _to_float(s: str) -> Optional[float]:
    s = s.strip().replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def _parse_bp_fallback(text: str) -> Tuple[Optional[int], Optional[int]]:
    candidates = []
    for m in re.finditer(r"(\d{2,3})\s*[/\-]\s*(\d{2,3})", text):
        s1 = int(m.group(1))
        s2 = int(m.group(2))
        if 70 <= s1 <= 260 and 30 <= s2 <= 150 and s1 > s2:
            start = max(0, m.start() - 25)
            end = min(len(text), m.end() + 25)
            window = text[start:end]
            score = 2 if _BP_CONTEXT.search(window) else 1
            candidates.append((score, s1, s2))

    if not candidates:
        return None, None

    candidates.sort(key=lambda x: (-x[0], x[1]))
    _, sys_val, dia_val = candidates[0]
    return sys_val, dia_val


def _parse_hr_fallback(text: str) -> Optional[int]:
    candidates = []
    patterns = [
        r"\bfc\s*[:=]?\s*(\d{2,3})\b",
        r"\bhr\s*[:=]?\s*(\d{2,3})\b",
        r"\b(\d{2,3})\s*bpm\b",
    ]
    for pat in patterns:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            val = int(m.group(1))
            if 30 <= val <= 220:
                start = max(0, m.start() - 25)
                end = min(len(text), m.end() + 25)
                window = text[start:end]
                score = 2 if _HR_CONTEXT.search(window) else 1
                candidates.append((score, val))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (-x[0], x[1]))
    return candidates[0][1]


def _parse_temp_fallback(text: str) -> Optional[float]:
    candidates = []
    patterns = [
        r"\btemp(?:eratura)?\s*[:=]?\s*(\d{2}[.,]\d)\b",
        r"\bt\s*[:=]?\s*(\d{2}[.,]\d)\b",
        r"\b(\d{2}[.,]\d)\s*°\s*c\b",
        r"\b(\d{2}[.,]\d)\s*°c\b",
    ]
    for pat in patterns:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            val = _to_float(m.group(1))
            if val is not None and 33.0 <= val <= 42.5:
                start = max(0, m.start() - 25)
                end = min(len(text), m.end() + 25)
                window = text[start:end]
                score = 2 if _TEMP_CONTEXT.search(window) else 1
                candidates.append((score, val))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (-x[0], x[1]))
    return candidates[0][1]


def _parse_spo2_fallback(text: str) -> Optional[int]:
    candidates = []
    patterns = [
        r"\bspo2\s*[:=]?\s*(\d{2,3})\s*%?\b",
        r"\bsaturazione\s*[:=]?\s*(\d{2,3})\s*%?\b",
        r"\bsat\.?\s*[:=]?\s*(\d{2,3})\s*%?\b",
    ]
    for pat in patterns:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            val = int(m.group(1))
            if 50 <= val <= 100:
                start = max(0, m.start() - 25)
                end = min(len(text), m.end() + 25)
                window = text[start:end]
                score = 2 if _SPO2_CONTEXT.search(window) else 1
                candidates.append((score, val))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (-x[0], x[1]))
    return candidates[0][1]


def extract_vitals_wrapper(text: str) -> Dict[str, Any]:
    vitals: Dict[str, Any] = {
        "blood_pressure_systolic": None,
        "blood_pressure_diastolic": None,
        "heart_rate": None,
        "temperature": None,
        "spo2": None,
    }

    if EXTRACT_BP:
        bp = EXTRACT_BP(text)
        if isinstance(bp, dict):
            vitals["blood_pressure_systolic"] = bp.get("blood_pressure_systolic") or bp.get("systolic")
            vitals["blood_pressure_diastolic"] = bp.get("blood_pressure_diastolic") or bp.get("diastolic")
        elif isinstance(bp, (tuple, list)) and len(bp) >= 2:
            vitals["blood_pressure_systolic"] = bp[0]
            vitals["blood_pressure_diastolic"] = bp[1]

    if EXTRACT_HR:
        vitals["heart_rate"] = EXTRACT_HR(text)

    if EXTRACT_TEMP:
        vitals["temperature"] = EXTRACT_TEMP(text)

    if EXTRACT_SPO2:
        vitals["spo2"] = EXTRACT_SPO2(text)

    if vitals["blood_pressure_systolic"] is None or vitals["blood_pressure_diastolic"] is None:
        sys_val, dia_val = _parse_bp_fallback(text)
        if vitals["blood_pressure_systolic"] is None:
            vitals["blood_pressure_systolic"] = sys_val
        if vitals["blood_pressure_diastolic"] is None:
            vitals["blood_pressure_diastolic"] = dia_val

    if vitals["heart_rate"] is None:
        vitals["heart_rate"] = _parse_hr_fallback(text)

    if vitals["temperature"] is None:
        vitals["temperature"] = _parse_temp_fallback(text)

    if vitals["spo2"] is None:
        vitals["spo2"] = _parse_spo2_fallback(text)

    return vitals


def build_base_record(record_id: str, mode: str, model: str) -> Dict[str, Any]:
    return {
        "meta": {
            "record_id": record_id,
            "template_type": ["diario_clinico"],
            "visit_datetime": None,
            "operator_role": "infermiere",
            "extraction_mode": mode,
            "llm_model": model if mode in {"llm", "hybrid"} else None,
            "pipeline_version": PIPELINE_VERSION,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        },
        "patient": {"patient_id": f"SYNTH-{record_id}", "age": None, "sex": None},
        "clinical": {
            "reason_for_visit": None,
            "anamnesis_brief": [],
            "vitals": {
                "blood_pressure_systolic": None,
                "blood_pressure_diastolic": None,
                "heart_rate": None,
                "temperature": None,
                "spo2": None,
            },
            "consciousness": None,
            "mobility": None,
            "interventions": [],
            "critical_issues": [],
            "follow_up": None,
        },
        "coding": {
            "problems_normalized": [],
            "problems_suspects": [],
        },
        "quality": {"missing_mandatory_fields": [], "warnings": []},
    }


def _normalize_reason(reason: Optional[str]) -> Optional[str]:
    if not reason:
        return None
    r = str(reason).strip().lower()
    r = r.replace("+", " e ")
    r = re.sub(r"\s+", " ", r).strip()

    replacements = {
        "monitoraggio segni vitali": "controllo parametri",
        "monitoraggio parametri vitali": "controllo parametri",
        "controllo parametri vitali": "controllo parametri",
        "controllo e medicazione lesione": "medicazione e controllo lesione",
        "controllo lesione": "medicazione e controllo lesione",
        "controllo ferita": "medicazione e controllo lesione",
        "medicazione ferita": "medicazione e controllo lesione",
        "medicazione lesione": "medicazione e controllo lesione",
        "rivalutazione del dolore": "rivalutazione dolore",
        "controllo terapia": "controllo terapia e somministrazione farmaco",
        "somministrazione terapia": "controllo terapia e somministrazione farmaco",
    }
    return replacements.get(r, r)


def _normalize_interventions(interventions: list[str], text: str, reason: Optional[str], vitals: dict) -> list[str]:
    t = (text or "").lower()
    r = (reason or "").lower()
    out: list[str] = []

    synonym_map = {
        "controllo_parametri_vitali": "monitoraggio_parametri_vitali",
        "monitoraggio_parametri_vitali": "monitoraggio_parametri_vitali",
        "valutazione": "valutazione_generale",
        "controllo generale": "valutazione_generale",
        "valutazione generale": "valutazione_generale",
        "somministrato farmaco": "somministrazione_farmaco",
        "somministrazione farmaco": "somministrazione_farmaco",
        "terapia": "educazione_terapeutica",
        "catetere": "gestione_catetere",
        "stomia": "gestione_stomia",
        "glicemia": "monitoraggio_glicemia",
    }

    for item in interventions or []:
        low = str(item).strip().lower()
        mapped = synonym_map.get(low, low)
        if mapped in INTERVENTION_VOCAB:
            out.append(mapped)

    has_any_vital = any(vitals.get(k) is not None for k in [
        "blood_pressure_systolic",
        "blood_pressure_diastolic",
        "heart_rate",
        "temperature",
        "spo2",
    ])
    if has_any_vital:
        out.append("monitoraggio_parametri_vitali")

    if any(k in t for k in ["medicazione", "ferita", "lesione", "piaga", "ulcera", "decubito"]) or "lesione" in r:
        out.append("medicazione")

    if any(k in t for k in ["farmaco", "somministrazione", "somministrato"]) or "farmaco" in r or "terapia" in r:
        out.append("somministrazione_farmaco")

    if any(k in t for k in ["terapia", "aderenza terapeutica", "caregiver", "istruito", "educazione"]) and "somministrazione_farmaco" not in out:
        out.append("educazione_terapeutica")

    if "catetere" in t or "catetere" in r:
        out.append("gestione_catetere")

    if "stomia" in t or "stomia" in r:
        out.append("gestione_stomia")

    if "glicemia" in t:
        out.append("monitoraggio_glicemia")

    if not out:
        out.append("valutazione_generale")

    out = list(dict.fromkeys(out))
    return [x for x in out if x in INTERVENTION_VOCAB]


def _infer_critical_issues(text: str) -> list[str]:
    t = (text or "").lower()
    issues = []
    if any(k in t for k in ["caduta recente", "post-caduta", "post caduta"]):
        issues.append("caduta_recente")
    if any(k in t for k in ["dispnea importante", "desaturazione", "spo2 88", "spo2 89", "spo2 90"]):
        issues.append("instabilita_respiratoria")
    return issues


def apply_rules(text: str, rec: Dict[str, Any]) -> None:
    vitals = extract_vitals_wrapper(text)

    reason = EXTRACT_REASON(text) if EXTRACT_REASON else None
    follow_up = EXTRACT_FOLLOW_UP(text) if EXTRACT_FOLLOW_UP else None

    interventions = []
    if EXTRACT_INTERVENTIONS:
        try:
            interventions = EXTRACT_INTERVENTIONS(text, vitals=vitals, reason=reason) or []
        except TypeError:
            interventions = EXTRACT_INTERVENTIONS(text) or []

    rec["meta"]["visit_datetime"] = EXTRACT_DATETIME(text) if EXTRACT_DATETIME else None
    rec["clinical"]["reason_for_visit"] = _normalize_reason(reason)
    rec["clinical"]["follow_up"] = follow_up
    rec["clinical"]["vitals"] = vitals
    rec["clinical"]["interventions"] = _normalize_interventions(
        interventions=interventions,
        text=text,
        reason=rec["clinical"]["reason_for_visit"],
        vitals=vitals,
    )
    rec["clinical"]["critical_issues"] = _infer_critical_issues(text)
    rec["coding"]["problems_normalized"] = normalize_problems(text) or []


def apply_llm(text: str, rec: Dict[str, Any], model: str) -> str:
    if llm_extract is None:
        raise RuntimeError("LLM extraction requested but src/llm_extract.py could not be imported.")

    out, llm_raw = llm_extract(text=text, model=model, return_raw=True)
    out = coerce_llm_output(out)

    rec["meta"]["visit_datetime"] = (
        out.get("meta", {}).get("visit_datetime")
        or (EXTRACT_DATETIME(text) if EXTRACT_DATETIME else None)
    )

    rec["clinical"]["reason_for_visit"] = _normalize_reason(out["clinical"].get("reason_for_visit"))
    rec["clinical"]["follow_up"] = out["clinical"].get("follow_up")

    rule_vitals = extract_vitals_wrapper(text)
    llm_vitals = out["clinical"].get("vitals", {}) or {}
    merged_vitals = {
        "blood_pressure_systolic": llm_vitals.get("blood_pressure_systolic") if llm_vitals.get("blood_pressure_systolic") is not None else rule_vitals.get("blood_pressure_systolic"),
        "blood_pressure_diastolic": llm_vitals.get("blood_pressure_diastolic") if llm_vitals.get("blood_pressure_diastolic") is not None else rule_vitals.get("blood_pressure_diastolic"),
        "heart_rate": llm_vitals.get("heart_rate") if llm_vitals.get("heart_rate") is not None else rule_vitals.get("heart_rate"),
        "temperature": llm_vitals.get("temperature") if llm_vitals.get("temperature") is not None else rule_vitals.get("temperature"),
        "spo2": llm_vitals.get("spo2") if llm_vitals.get("spo2") is not None else rule_vitals.get("spo2"),
    }
    rec["clinical"]["vitals"] = merged_vitals

    rec["clinical"]["interventions"] = _normalize_interventions(
        interventions=out["clinical"].get("interventions", []),
        text=text,
        reason=rec["clinical"]["reason_for_visit"],
        vitals=merged_vitals,
    )
    rec["clinical"]["critical_issues"] = _infer_critical_issues(text)

    probs = out["coding"].get("problems_normalized", [])
    final: list[str] = []
    suspects: list[str] = []
    for p in probs:
        if has_evidence(p, text):
            final.append(p)
        else:
            suspects.append(p)

    rule_probs = normalize_problems(text) or []
    final = sorted(set(final) | set(rule_probs))
    rec["coding"]["problems_normalized"] = final
    rec["coding"]["problems_suspects"] = sorted(set(suspects))

    if rec["clinical"]["reason_for_visit"] is None and EXTRACT_REASON:
        rec["clinical"]["reason_for_visit"] = _normalize_reason(EXTRACT_REASON(text))
    if rec["clinical"]["follow_up"] is None and EXTRACT_FOLLOW_UP:
        rec["clinical"]["follow_up"] = EXTRACT_FOLLOW_UP(text)

    return llm_raw


def apply_hybrid(text: str, rec: Dict[str, Any], model: str) -> str:
    if llm_extract is None:
        apply_rules(text, rec)
        return ""

    out, llm_raw = llm_extract(text=text, model=model, return_raw=True)
    out = coerce_llm_output(out)

    rule_vitals = extract_vitals_wrapper(text)
    rule_reason = EXTRACT_REASON(text) if EXTRACT_REASON else None
    rule_follow = EXTRACT_FOLLOW_UP(text) if EXTRACT_FOLLOW_UP else None

    rec["meta"]["visit_datetime"] = (
        out.get("meta", {}).get("visit_datetime")
        or (EXTRACT_DATETIME(text) if EXTRACT_DATETIME else None)
    )

    llm_reason = _normalize_reason(out["clinical"].get("reason_for_visit"))
    rec["clinical"]["reason_for_visit"] = llm_reason or _normalize_reason(rule_reason)

    rec["clinical"]["follow_up"] = out["clinical"].get("follow_up") or rule_follow
    rec["clinical"]["vitals"] = rule_vitals

    llm_interventions = out["clinical"].get("interventions", [])
    rule_interventions = []
    if EXTRACT_INTERVENTIONS:
        try:
            rule_interventions = EXTRACT_INTERVENTIONS(
                text,
                vitals=rule_vitals,
                reason=rec["clinical"]["reason_for_visit"],
            ) or []
        except TypeError:
            rule_interventions = EXTRACT_INTERVENTIONS(text) or []

    merged_interventions = list(dict.fromkeys((llm_interventions or []) + (rule_interventions or [])))
    rec["clinical"]["interventions"] = _normalize_interventions(
        interventions=merged_interventions,
        text=text,
        reason=rec["clinical"]["reason_for_visit"],
        vitals=rule_vitals,
    )
    rec["clinical"]["critical_issues"] = _infer_critical_issues(text)

    rule_probs = set(normalize_problems(text) or [])
    llm_probs = out["coding"].get("problems_normalized", []) or []
    suspects: list[str] = []
    for p in llm_probs:
        if has_evidence(p, text):
            rule_probs.add(p)
        else:
            suspects.append(p)

    rec["coding"]["problems_normalized"] = sorted(rule_probs)
    rec["coding"]["problems_suspects"] = sorted(set(suspects))

    return llm_raw


def postprocess_record(rec: Dict[str, Any], text: str) -> None:
    if rec["meta"].get("visit_datetime") is None and EXTRACT_DATETIME:
        rec["meta"]["visit_datetime"] = EXTRACT_DATETIME(text)

    reason = rec["clinical"].get("reason_for_visit")
    rec["clinical"]["reason_for_visit"] = _normalize_reason(reason)

    vitals = rec["clinical"].get("vitals", {}) or {}
    rec["clinical"]["interventions"] = _normalize_interventions(
        rec["clinical"].get("interventions", []),
        text=text,
        reason=rec["clinical"].get("reason_for_visit"),
        vitals=vitals,
    )

    if rec["clinical"].get("follow_up") is None and EXTRACT_FOLLOW_UP:
        rec["clinical"]["follow_up"] = EXTRACT_FOLLOW_UP(text)

    rec["clinical"]["critical_issues"] = list(dict.fromkeys(rec["clinical"].get("critical_issues", []) + _infer_critical_issues(text)))


def run_quality_check(rec: Dict[str, Any], text: str) -> Dict[str, Any]:
    try:
        return quality_check(rec, text)  # type: ignore
    except TypeError:
        return quality_check(rec)  # type: ignore


def save_prediction(record_id: str, rec: Dict[str, Any]) -> Path:
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PRED_DIR / f"{record_id}.json"
    out_path.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def save_example(record_id: str, rec: Dict[str, Any], raw_text: str, processed_text: str) -> None:
    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    example_payload = {
        "record_id": record_id,
        "raw_text": raw_text,
        "processed_text": processed_text,
        "prediction": rec,
    }
    out_path = EXAMPLES_DIR / f"{record_id}_example.json"
    out_path.write_text(json.dumps(example_payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-llm", action="store_true", help="Use LLM extraction only")
    parser.add_argument("--hybrid", action="store_true", help="Use hybrid extraction")
    parser.add_argument("--model", default="llama3.1:8b", help="LLM model name (Ollama)")
    args = parser.parse_args()

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    mode = "hybrid" if args.hybrid else "llm" if args.use_llm else "rules"

    records_total = 0
    records_ok = 0
    records_failed = 0
    exported_examples = 0
    failures: list[dict[str, str]] = []

    DEBUG_LOG_PATH.write_text("", encoding="utf-8")

    raw_files = sorted(RAW_DIR.glob("ADI-*.txt"))
    if not raw_files:
        raise FileNotFoundError(f"No input files found in {RAW_DIR.resolve()}")

    log_debug(f"Pipeline started in mode={mode}, model={args.model}, files={len(raw_files)}")

    for txt_path in raw_files:
        records_total += 1
        record_id = txt_path.stem

        try:
            raw = txt_path.read_text(encoding="utf-8")
            text = PREPROCESS(raw)

            rec = build_base_record(record_id, mode=mode, model=args.model)

            if args.hybrid:
                _ = apply_hybrid(text, rec, args.model)
            elif args.use_llm:
                _ = apply_llm(text, rec, args.model)
            else:
                apply_rules(text, rec)

            postprocess_record(rec, text)

            q = run_quality_check(rec, text)
            rec["quality"]["missing_mandatory_fields"] = q.get(
                "missing_mandatory_fields",
                q.get("missing_fields", []),
            )
            rec["quality"]["warnings"] = q.get("warnings", [])

            out_path = save_prediction(record_id, rec)
            print(f"Wrote {out_path}")
            log_debug(f"SUCCESS {record_id} -> {out_path}")

            if exported_examples < MAX_EXAMPLE_EXPORTS:
                save_example(record_id, rec, raw_text=raw, processed_text=text)
                exported_examples += 1

            records_ok += 1

        except Exception as e:
            records_failed += 1
            failures.append({"record_id": record_id, "error": str(e)})
            log_debug(f"ERROR {record_id}: {e}")
            print(f"Error on {record_id}: {e}")

    mode_payload = {
        "use_llm": args.use_llm,
        "hybrid": args.hybrid,
        "model": args.model,
        "pipeline_version": PIPELINE_VERSION,
    }
    (REPORTS_DIR / "llm_mode.json").write_text(
        json.dumps(mode_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "mode": mode,
        "model": args.model if mode in {"llm", "hybrid"} else None,
        "pipeline_version": PIPELINE_VERSION,
        "raw_dir": str(RAW_DIR),
        "pred_dir": str(PRED_DIR),
        "examples_dir": str(EXAMPLES_DIR),
        "records_total": records_total,
        "records_ok": records_ok,
        "records_failed": records_failed,
        "examples_exported": exported_examples,
        "failures": failures,
    }
    RUN_SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nRun summary")
    print(f"- total records: {records_total}")
    print(f"- successful: {records_ok}")
    print(f"- failed: {records_failed}")
    print(f"- examples exported: {exported_examples}")
    print(f"- summary file: {RUN_SUMMARY_PATH}")
    print(f"- debug log: {DEBUG_LOG_PATH}")


if __name__ == "__main__":
    main()