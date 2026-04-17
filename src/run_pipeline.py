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

PIPELINE_VERSION = "0.4.1"
MAX_EXAMPLE_EXPORTS = 3

INTERVENTION_VOCAB = {
    "monitoraggio_parametri_vitali",
    "valutazione_generale",
    "medicazione",
    "somministrazione_farmaco",
    "monitoraggio_glicemia",
    "gestione_catetere",
    "gestione_stomia",
    "gestione_ossigenoterapia",
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
    ]
    fn = _pick_callable(preprocess_mod, candidates)
    if fn:
        return fn

    raise ImportError("No preprocess function found in src/preprocess.py")


PREPROCESS = get_preprocess_fn()

EXTRACT_REASON = _pick_callable(rules_mod, ["extract_reason", "extract_reason_for_visit"])
EXTRACT_FOLLOW_UP = _pick_callable(rules_mod, ["extract_follow_up"])
EXTRACT_INTERVENTIONS = _pick_callable(rules_mod, ["extract_interventions"])
EXTRACT_BP = _pick_callable(rules_mod, ["extract_bp"])
EXTRACT_HR = _pick_callable(rules_mod, ["extract_hr"])
EXTRACT_TEMP = _pick_callable(rules_mod, ["extract_temp"])
EXTRACT_SPO2 = _pick_callable(rules_mod, ["extract_spo2"])
EXTRACT_DATETIME = _pick_callable(rules_mod, ["extract_datetime"])


def extract_vitals_wrapper(text: str) -> Dict[str, Any]:
    return {
        "blood_pressure_systolic": (EXTRACT_BP(text)[0] if EXTRACT_BP else None),
        "blood_pressure_diastolic": (EXTRACT_BP(text)[1] if EXTRACT_BP else None),
        "heart_rate": EXTRACT_HR(text) if EXTRACT_HR else None,
        "temperature": EXTRACT_TEMP(text) if EXTRACT_TEMP else None,
        "spo2": EXTRACT_SPO2(text) if EXTRACT_SPO2 else None,
    }


def build_base_record(record_id: str, mode: str, model: str) -> Dict[str, Any]:
    return {
        "meta": {
            "record_id": record_id,
            "visit_datetime": None,
            "operator_role": "infermiere",
            "extraction_mode": mode,
            "llm_model": model if mode != "rules" else None,
            "pipeline_version": PIPELINE_VERSION,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        },
        "clinical": {
            "reason_for_visit": None,
            "vitals": {},
            "interventions": [],
            "follow_up": None,
            "critical_issues": [],
        },
        "coding": {"problems_normalized": [], "problems_suspects": []},
        "quality": {"missing_mandatory_fields": [], "warnings": []},
    }


def _normalize_interventions(interventions, text, reason, vitals):
    out = set(interventions or [])

    if any(vitals.values()):
        out.add("monitoraggio_parametri_vitali")

    if "lesione" in (reason or ""):
        out.add("medicazione")

    if "farmaco" in (reason or ""):
        out.add("somministrazione_farmaco")

    if not out:
        out.add("valutazione_generale")

    return list(out & INTERVENTION_VOCAB)


def apply_rules(text: str, rec: Dict[str, Any]):
    vitals = extract_vitals_wrapper(text)

    rec["meta"]["visit_datetime"] = EXTRACT_DATETIME(text) if EXTRACT_DATETIME else None
    rec["clinical"]["reason_for_visit"] = EXTRACT_REASON(text) if EXTRACT_REASON else None
    rec["clinical"]["follow_up"] = EXTRACT_FOLLOW_UP(text) if EXTRACT_FOLLOW_UP else None
    rec["clinical"]["vitals"] = vitals

    interventions = EXTRACT_INTERVENTIONS(text, vitals=vitals) if EXTRACT_INTERVENTIONS else []
    rec["clinical"]["interventions"] = _normalize_interventions(interventions, text, rec["clinical"]["reason_for_visit"], vitals)

    rec["coding"]["problems_normalized"] = normalize_problems(text) or []


def apply_llm(text: str, rec: Dict[str, Any], model: str):
    out, _ = llm_extract(text=text, model=model, return_raw=True)
    out = coerce_llm_output(out)

    rec["meta"]["visit_datetime"] = out.get("meta", {}).get("visit_datetime")
    rec["clinical"] = out.get("clinical", {})
    rec["coding"] = out.get("coding", {})


def apply_hybrid(text: str, rec: Dict[str, Any], model: str):
    apply_rules(text, rec)
    if llm_extract:
        out, _ = llm_extract(text=text, model=model, return_raw=True)
        out = coerce_llm_output(out)

        if not rec["clinical"]["reason_for_visit"]:
            rec["clinical"]["reason_for_visit"] = out["clinical"].get("reason_for_visit")

        rec["coding"]["problems_normalized"] = list(set(
            rec["coding"]["problems_normalized"] +
            out["coding"].get("problems_normalized", [])
        ))


def run_quality_check(rec):
    return quality_check(rec)


def save_prediction(record_id: str, rec: Dict[str, Any]):
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    path = PRED_DIR / f"{record_id}.json"
    path.write_text(json.dumps(rec, indent=2, ensure_ascii=False))
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument("--hybrid", action="store_true")
    parser.add_argument("--model", default="llama3.1:8b")
    args = parser.parse_args()

    mode = "hybrid" if args.hybrid else "llm" if args.use_llm else "rules"

    raw_files = list(RAW_DIR.glob("ADI-*.txt"))

    for txt in raw_files:
        record_id = txt.stem
        text = PREPROCESS(txt.read_text())

        rec = build_base_record(record_id, mode, args.model)

        if mode == "rules":
            apply_rules(text, rec)
        elif mode == "llm":
            apply_llm(text, rec, args.model)
        else:
            apply_hybrid(text, rec, args.model)

        rec["quality"] = run_quality_check(rec)

        save_prediction(record_id, rec)

        print(f"Processed {record_id}")


if __name__ == "__main__":
    main()