# src/run_pipeline.py

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

# LLM extractor (Ollama-based in src/llm_extract.py)
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

PIPELINE_VERSION = "0.3.0"
MAX_EXAMPLE_EXPORTS = 3


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
        f"Available: {available}\n"
        "Fix: rename your preprocess function to 'preprocess_text' or add a wrapper."
    )


PREPROCESS = get_preprocess_fn()

# ---- Rule extractors (robust resolution) ----

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

# ---------------------------
# Robust vitals fallback regex
# ---------------------------

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
    """
    Parse BP like:
      - 130/80
      - 130-80
      - PA 130/80
      - pressione 135-80
    Avoid dates by requiring plausible BP ranges.
    """
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

    # Rules first
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

    # Fallbacks if needed
    sys_missing = vitals["blood_pressure_systolic"] is None
    dia_missing = vitals["blood_pressure_diastolic"] is None
    if sys_missing or dia_missing:
        sys_val, dia_val = _parse_bp_fallback(text)
        if sys_missing:
            vitals["blood_pressure_systolic"] = sys_val
        if dia_missing:
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


def apply_rules(text: str, rec: Dict[str, Any]) -> None:
    if EXTRACT_REASON:
        rec["clinical"]["reason_for_visit"] = EXTRACT_REASON(text)
    if EXTRACT_FOLLOW_UP:
        rec["clinical"]["follow_up"] = EXTRACT_FOLLOW_UP(text)
    if EXTRACT_INTERVENTIONS:
        rec["clinical"]["interventions"] = EXTRACT_INTERVENTIONS(text) or []
    rec["clinical"]["vitals"] = extract_vitals_wrapper(text)
    rec["coding"]["problems_normalized"] = normalize_problems(text) or []


def apply_llm(text: str, rec: Dict[str, Any], model: str, record_id: str) -> str:
    if llm_extract is None:
        raise RuntimeError("LLM extraction requested but src/llm_extract.py could not be imported.")

    out, llm_raw = llm_extract(text=text, model=model, return_raw=True)
    out = coerce_llm_output(out)

    rec["clinical"]["reason_for_visit"] = out["clinical"].get("reason_for_visit")
    rec["clinical"]["follow_up"] = out["clinical"].get("follow_up")
    rec["clinical"]["interventions"] = out["clinical"].get("interventions", [])
    rec["clinical"]["vitals"] = out["clinical"].get("vitals", rec["clinical"]["vitals"])

    probs = out["coding"].get("problems_normalized", [])
    final: list[str] = []
    suspects: list[str] = []
    for p in probs:
        if has_evidence(p, text):
            final.append(p)
        else:
            suspects.append(p)
    rec["coding"]["problems_normalized"] = sorted(set(final))
    rec["coding"]["problems_suspects"] = sorted(set(suspects))

    return llm_raw


def apply_hybrid(text: str, rec: Dict[str, Any], model: str, record_id: str) -> str:
    if llm_extract is None:
        raise RuntimeError("Hybrid requested but src/llm_extract.py could not be imported.")

    out, llm_raw = llm_extract(text=text, model=model, return_raw=True)
    out = coerce_llm_output(out)

    rec["clinical"]["reason_for_visit"] = out["clinical"].get("reason_for_visit")
    rec["clinical"]["follow_up"] = out["clinical"].get("follow_up")
    rec["clinical"]["interventions"] = out["clinical"].get("interventions", [])

    # Rules for vitals
    rec["clinical"]["vitals"] = extract_vitals_wrapper(text)

    # Problems: rules first
    rule_probs = normalize_problems(text) or []
    final_set = set(rule_probs)

    llm_probs = out["coding"].get("problems_normalized", []) or []
    suspects: list[str] = []
    for p in llm_probs:
        if p in final_set:
            continue
        if has_evidence(p, text):
            final_set.add(p)
        else:
            suspects.append(p)

    rec["coding"]["problems_normalized"] = sorted(final_set)
    rec["coding"]["problems_suspects"] = sorted(set(suspects))

    return llm_raw


def normalize_reason(reason: Optional[str]) -> Optional[str]:
    if not reason:
        return None
    r = reason.strip().lower()
    r = r.replace("+", " e ")
    r = re.sub(r"\s+", " ", r).strip()
    r = re.sub(r"\bdx\b", "destro", r)
    r = re.sub(r"\bsx\b", "sinistro", r)
    r = re.sub(r"\bda giorni\b", "", r).strip()
    r = re.sub(r"\s+", " ", r).strip()
    return r


def normalize_follow_up(fu: Optional[str]) -> Optional[str]:
    if fu is None:
        return None
    s = fu.strip().lower()
    s = re.sub(r"\s+", " ", s)

    m = re.match(r"^(?:programmato controllo )?(?:tra )?(\d+)\s*giorni$", s)
    if m:
        n = m.group(1)
        return f"programmato controllo tra {n} giorni"

    if s in {"nuovo controllo", "programmato nuovo controllo"}:
        return "programmato nuovo controllo"

    return s


def follow_up_to_struct(fu: Optional[str]) -> Any:
    if fu is None:
        return None
    s = fu.strip().lower()
    s = re.sub(r"\s+", " ", s)

    m = re.match(r"^programmato controllo tra (\d+)\s*giorni$", s)
    if m:
        return {"type": "controllo", "timing_days": int(m.group(1))}

    if s in {"programmato nuovo controllo", "nuovo controllo"}:
        return {"type": "controllo", "timing_days": None}

    if "ricontatto" in s and "telefon" in s and "caregiver" in s:
        return {"type": "ricontatto_telefonico", "target": "caregiver", "timing_days": None}

    if "ricontatto" in s and "telefon" in s:
        return {"type": "ricontatto_telefonico", "target": None, "timing_days": None}

    return s


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

INTERVENTION_SYNONYMS = {
    "controllo parametri": "monitoraggio_parametri_vitali",
    "controllo parametri vitali": "monitoraggio_parametri_vitali",
    "rilevati parametri": "monitoraggio_parametri_vitali",
    "rilevazione parametri": "monitoraggio_parametri_vitali",
    "monitoraggio parametri": "monitoraggio_parametri_vitali",
    "monitoraggio parametri vitali": "monitoraggio_parametri_vitali",
    "controllo generale": "valutazione_generale",
    "valutazione generale": "valutazione_generale",
    "valutazione": "valutazione_generale",
    "consigli alimentari": "consigli_alimentari",
    "educazione alimentare": "educazione_alimentare",
    "medicazione avanzata": "medicazione",
    "medicazione lesione": "medicazione",
    "medicazione piaga": "medicazione",
    "cambio medicazione": "medicazione",
    "somministrato farmaco": "somministrazione_farmaco",
    "somministrazione farmaco": "somministrazione_farmaco",
    "terapia": "educazione_terapeutica",
    "catetere": "gestione_catetere",
    "stomia": "gestione_stomia",
    "glicemia": "monitoraggio_glicemia",
}


def normalize_interventions(interventions: list[str], text: str, reason: Optional[str]) -> list[str]:
    t = (text or "").lower()
    r = (reason or "").lower()
    out: list[str] = []

    for it in interventions or []:
        low = str(it).strip().lower()
        mapped = INTERVENTION_SYNONYMS.get(low, low)
        if mapped in INTERVENTION_VOCAB:
            out.append(mapped)

    if ("medicazione" in t) or ("medicazione" in r) or ("piaga" in t) or ("lesione" in t) or ("decubito" in t):
        out.append("medicazione")

    if ("glicemia" in t) or ("diabete" in t):
        out.append("monitoraggio_glicemia")

    out = list(dict.fromkeys(out))
    return [x for x in out if x in INTERVENTION_VOCAB]


def postprocess_record(rec: Dict[str, Any], text: str) -> None:
    rec["clinical"]["reason_for_visit"] = normalize_reason(rec["clinical"].get("reason_for_visit"))
    rec["clinical"]["follow_up"] = normalize_follow_up(rec["clinical"].get("follow_up"))

    if rec["clinical"]["reason_for_visit"] is None and EXTRACT_REASON:
        rec["clinical"]["reason_for_visit"] = normalize_reason(EXTRACT_REASON(text))

    if rec["clinical"]["follow_up"] is None and "nuovo controllo" in (text or "").lower():
        rec["clinical"]["follow_up"] = "programmato nuovo controllo"

    rec["clinical"]["interventions"] = normalize_interventions(
        rec["clinical"].get("interventions", []),
        text=text,
        reason=rec["clinical"].get("reason_for_visit"),
    )

    vitals = rec.get("clinical", {}).get("vitals", {}) or {}
    has_any_vital = any(
        vitals.get(k) is not None
        for k in ["blood_pressure_systolic", "blood_pressure_diastolic", "heart_rate", "temperature", "spo2"]
    )

    if has_any_vital and "monitoraggio_parametri_vitali" not in rec["clinical"]["interventions"]:
        rec["clinical"]["interventions"].insert(0, "monitoraggio_parametri_vitali")

    rec["clinical"]["follow_up"] = follow_up_to_struct(rec["clinical"].get("follow_up"))


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
    parser.add_argument("--hybrid", action="store_true", help="Use hybrid extraction (recommended)")
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
                _ = apply_hybrid(text, rec, args.model, record_id=record_id)
            elif args.use_llm:
                _ = apply_llm(text, rec, args.model, record_id=record_id)
            else:
                apply_rules(text, rec)

            postprocess_record(rec, text)

            q = run_quality_check(rec, text)
            rec["quality"]["missing_mandatory_fields"] = q.get("missing_fields", [])
            rec["quality"]["warnings"] = q.get("warnings", [])

            out_path = save_prediction(record_id, rec)
            print(f"Wrote {out_path}")
            log_debug(f"SUCCESS {record_id} -> {out_path}")

            if exported_examples < MAX_EXAMPLE_EXPORTS:
                save_example(record_id, rec, raw_text=raw, processed_text=text)
                exported_examples += 1
                log_debug(f"EXAMPLE_EXPORTED {record_id}")

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