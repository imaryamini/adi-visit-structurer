from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPORTS_DIR = Path("reports")
GOLD_DIR = Path("data/synthetic/gold")
PRED_DIR = Path("data/synthetic/pred")


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_get(dct: Dict[str, Any], *keys: str, default=None):
    cur = dct
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def canonical_reason(label: str | None) -> str | None:
    if not label:
        return None

    t = str(label).strip().lower()

    mapping = {
        "controllo parametri": "controllo_parametri",
        "controllo terapia e somministrazione farmaco": "controllo_terapia",
        "controllo terapia/somministrazione farmaco": "controllo_terapia",
        "controllo pressione e rivalutazione terapia": "controllo_terapia",
        "monitoraggio segni vitali e verifica terapia": "controllo_terapia",

        "medicazione e controllo lesione": "medicazione_lesione",
        "medicazione/controllo lesione": "medicazione_lesione",
        "medicazione lesione da pressione": "medicazione_lesione",
        "medicazione piaga da decubito": "medicazione_lesione",

        "valutazione dolore cronico": "rivalutazione_dolore",
        "valutazione dolore e controllo parametri": "rivalutazione_dolore",
        "rivalutazione dolore": "rivalutazione_dolore",
        "dolore al ginocchio destro": "rivalutazione_dolore",
        "dolore lombare": "rivalutazione_dolore",
        "dolore toracico": "rivalutazione_dolore",
        "dolore addominale": "rivalutazione_dolore",

        "controllo e gestione catetere": "controllo_catetere",
        "controllo e gestione stomia": "controllo_stomia",

        "controllo respiratorio e gestione ossigenoterapia": "controllo_respiratorio",
        "dispnea": "controllo_respiratorio",
        "tosse, febbre e lieve dispnea": "controllo_respiratorio",

        "riferiti sintomi generali": "sintomi_generali",
        "stanchezza e scarso appetito": "sintomi_generali",
        "febbre": "sintomi_generali",

        "educazione caregiver e controllo generale": "controllo_generale",
        "controllo generale": "controllo_generale",
        "valutazione generale": "controllo_generale",
        "valutazione delle condizioni generali": "controllo_generale",

        "rivalutazione caduta recente": "caduta",
        "controllo post-caduta": "caduta",
    }

    if t in mapping:
        return mapping[t]

    if any(x in t for x in ["lesione", "ferita", "medicazione", "decubito", "piaga", "ulcera"]):
        return "medicazione_lesione"
    if "catetere" in t:
        return "controllo_catetere"
    if "stomia" in t:
        return "controllo_stomia"
    if any(x in t for x in ["ossigenoterapia", "respiratorio", "dispnea", "affanno", "tosse"]):
        return "controllo_respiratorio"
    if "caduta" in t:
        return "caduta"
    if any(x in t for x in ["terapia", "farmaco", "somministrazione"]):
        return "controllo_terapia"
    if any(x in t for x in ["parametri", "segni vitali", "pressione", "frequenza cardiaca", "spo2", "saturazione"]):
        return "controllo_parametri"
    if any(x in t for x in ["dolore", "algia"]):
        return "rivalutazione_dolore"
    if any(x in t for x in ["astenia", "inappetenza", "nausea", "capogiro", "appetito", "stanchezza", "febbre", "sintomi generali"]):
        return "sintomi_generali"
    if any(x in t for x in ["generali", "controllo generale", "valutazione generale", "rivalutazione clinica"]):
        return "controllo_generale"

    return t


def canonical_problem(label: str | None) -> str | None:
    if not label:
        return None

    t = str(label).strip().lower()

    mapping = {
        "dolore": "dolore_cronico",
        "dolore_cronico": "dolore_cronico",
        "dolore cronico": "dolore_cronico",
        "dolore_generico": "dolore_cronico",

        "lesione": "lesione_da_pressione",
        "lesione_cutanea": "lesione_da_pressione",
        "lesione_da_pressione": "lesione_da_pressione",
        "lesione da pressione": "lesione_da_pressione",
        "piaga da decubito": "lesione_da_pressione",
        "decubito": "lesione_da_pressione",
        "ferita": "lesione_da_pressione",
        "ulcera": "lesione_da_pressione",

        "caduta": "caduta",
        "caduta_recente": "caduta",
        "rischio_caduta": "rischio_caduta",

        "ipertensione": "ipertensione",
        "bpco": "bpco",
        "dispnea": "dispnea",
        "affanno": "dispnea",
        "scompenso_cardiaco": "scompenso_cardiaco",
        "diabete_tipo_2": "diabete_tipo_2",
        "malnutrizione": "malnutrizione",
        "disidratazione": "disidratazione",
        "astenia": "astenia",
        "nausea": "nausea",
        "capogiro": "capogiro",
        "febbre": "febbre",
    }

    if t in mapping:
        return mapping[t]

    if "dolore" in t:
        return "dolore_cronico"
    if any(x in t for x in ["lesione", "ferita", "ulcera", "decubito", "piaga"]):
        return "lesione_da_pressione"
    if "caduta" in t:
        return "caduta"
    if "pressione" in t or "ipertension" in t:
        return "ipertensione"

    if "dispnea" in t or "affanno" in t:
        return "dispnea"
    if "bpco" in t:
        return "bpco"

    if "scompenso" in t:
        return "scompenso_cardiaco"
    if "diabete" in t or "glicemia" in t:
        return "diabete_tipo_2"
    if "appetito" in t or "inappetenza" in t:
        return "malnutrizione"
    if "disidrat" in t:
        return "disidratazione"
    if "astenia" in t or "stanchezza" in t or "debolezza" in t:
        return "astenia"
    if "nausea" in t:
        return "nausea"
    if "capogiro" in t or "vertigin" in t:
        return "capogiro"
    if "febbre" in t:
        return "febbre"

    return t


def canonicalize_problem_list(items: Iterable[str]) -> List[str]:
    out = []
    for item in items or []:
        c = canonical_problem(item)
        if c:
            out.append(c)
    return sorted(set(out))


def _extract_timing_days_from_text(text: str) -> Optional[int]:
    if not text:
        return None

    t = str(text).strip().lower()

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

    digit_patterns = [
        (r"\btra\s+(\d+)\s+giorni\b", 1),
        (r"\bentro\s+(\d+)\s+giorni\b", 1),
        (r"\bnei\s+prossimi\s+(\d+)\s+giorni\b", 1),
        (r"\bnelle\s+prossime\s+(\d+)\s+settimane\b", 7),
        (r"\btra\s+(\d+)\s+settimane\b", 7),
        (r"\btra\s+(\d+)\s+mesi\b", 30),
        (r"\bentro\s+(\d+)\s+settimane\b", 7),
        (r"\bentro\s+(\d+)\s+mesi\b", 30),
    ]

    for pat, mult in digit_patterns:
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
    if re.search(r"\bprossima\s+settimana\b", t):
        return 7
    if re.search(r"\bsettimana\s+prossima\b", t):
        return 7

    return None


def normalize_follow_up(fu) -> Optional[Dict[str, Any]]:
    if fu is None:
        return None

    if isinstance(fu, dict):
        type_value = fu.get("type")
        type_norm = str(type_value).strip().lower() if type_value is not None else None
        timing_days = fu.get("timing_days")

        if timing_days is None:
            timing_days = _extract_timing_days_from_text(json.dumps(fu, ensure_ascii=False))

        if type_norm:
            if "telefon" in type_norm:
                return {"type": "ricontatto_telefonico", "timing_days": timing_days, "target": None}
            if any(x in type_norm for x in ["ferita", "lesione", "medicazione", "piaga", "ulcera"]):
                return {"type": "controllo_ferita", "timing_days": timing_days}
            if any(x in type_norm for x in ["controllo", "rivalutazione", "monitoraggio", "ricontrollo", "follow-up", "follow up"]):
                return {"type": "controllo", "timing_days": timing_days}

        return {"type": type_norm, "timing_days": timing_days}

    if isinstance(fu, str):
        s = fu.strip().lower()
        timing_days = _extract_timing_days_from_text(s)

        if "telefon" in s:
            return {"type": "ricontatto_telefonico", "timing_days": timing_days, "target": None}

        if any(x in s for x in ["ferita", "lesione", "medicazione", "piaga", "ulcera"]):
            if any(x in s for x in ["controllo", "rivalutazione", "ricontrollo", "programmato", "previsto", "follow"]):
                return {"type": "controllo_ferita", "timing_days": timing_days}

        if any(x in s for x in ["controllo", "rivalutazione", "monitoraggio", "ricontrollo", "follow-up", "follow up"]):
            return {"type": "controllo", "timing_days": timing_days}

        if any(x in s for x in ["a breve", "nei prossimi giorni", "tra qualche giorno", "prossima settimana", "settimana prossima"]):
            return {"type": "controllo", "timing_days": timing_days}

        return {"type": s, "timing_days": timing_days}

    return None


def follow_up_equal(gold_fu, pred_fu) -> bool:
    g = normalize_follow_up(gold_fu)
    p = normalize_follow_up(pred_fu)

    if g is None and p is None:
        return True
    if g is None or p is None:
        return False

    if g.get("type") != p.get("type"):
        return False

    g_days = g.get("timing_days")
    p_days = p.get("timing_days")

    if g_days is None and p_days is None:
        return True
    if g_days is None or p_days is None:
        return False

    return abs(int(g_days) - int(p_days)) <= 3


def normalize_vitals_for_compare(vitals):
    vitals = vitals or {}

    systolic = vitals.get("blood_pressure_systolic")
    diastolic = vitals.get("blood_pressure_diastolic")

    if systolic is None or diastolic is None:
        bp = vitals.get("blood_pressure")
        if isinstance(bp, str) and "/" in bp:
            left, right = bp.split("/", 1)
            left = left.strip()
            right = right.strip()
            systolic = systolic if systolic is not None else left
            diastolic = diastolic if diastolic is not None else right

    return {
        "blood_pressure_systolic": None if systolic in ("", None) else str(systolic),
        "blood_pressure_diastolic": None if diastolic in ("", None) else str(diastolic),
        "heart_rate": None if vitals.get("heart_rate") in ("", None) else str(vitals.get("heart_rate")),
        "temperature": None if vitals.get("temperature") in ("", None) else str(vitals.get("temperature")),
        "spo2": None if vitals.get("spo2") in ("", None) else str(vitals.get("spo2")),
    }


def vitals_equal(gold_v, pred_v) -> bool:
    keys = [
        "blood_pressure_systolic",
        "blood_pressure_diastolic",
        "heart_rate",
        "temperature",
        "spo2",
    ]
    gold_v = normalize_vitals_for_compare(gold_v)
    pred_v = normalize_vitals_for_compare(pred_v)

    for k in keys:
        if gold_v.get(k) != pred_v.get(k):
            return False
    return True


def f1_for_multilabel(gold_items, pred_items) -> Tuple[float, float, float]:
    gold_set = set(gold_items or [])
    pred_set = set(pred_items or [])

    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def macro_metric_over_records(pairs, gold_getter, pred_getter) -> Tuple[float, float, float]:
    precisions = []
    recalls = []
    f1s = []

    for gold, pred in pairs:
        p, r, f1 = f1_for_multilabel(gold_getter(gold), pred_getter(pred))
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)

    n = len(pairs) or 1
    return sum(precisions) / n, sum(recalls) / n, sum(f1s) / n


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    gold_files = {p.stem: p for p in GOLD_DIR.glob("ADI-*.json")}
    pred_files = {p.stem: p for p in PRED_DIR.glob("ADI-*.json")}

    common_ids = sorted(set(gold_files) & set(pred_files))
    gold_only = sorted(set(gold_files) - set(pred_files))
    pred_only = sorted(set(pred_files) - set(gold_files))

    pairs = []
    for rid in common_ids:
        gold = load_json(gold_files[rid])
        pred = load_json(pred_files[rid])
        pairs.append((gold, pred))

    n = len(pairs)

    reason_correct = 0
    follow_up_correct = 0
    vitals_correct = 0
    reason_errors = []
    follow_up_errors = []
    vitals_errors = []

    for gold, pred in pairs:
        rid = safe_get(gold, "meta", "record_id", default="UNKNOWN")

        gold_reason = canonical_reason(safe_get(gold, "clinical", "reason_for_visit"))
        pred_reason = canonical_reason(safe_get(pred, "clinical", "reason_for_visit"))

        if gold_reason == pred_reason:
            reason_correct += 1
        else:
            reason_errors.append({
                "record_id": rid,
                "gold_reason_raw": safe_get(gold, "clinical", "reason_for_visit"),
                "pred_reason_raw": safe_get(pred, "clinical", "reason_for_visit"),
                "gold_reason_canonical": gold_reason,
                "pred_reason_canonical": pred_reason,
            })

        gold_fu = safe_get(gold, "clinical", "follow_up")
        pred_fu = safe_get(pred, "clinical", "follow_up")

        if follow_up_equal(gold_fu, pred_fu):
            follow_up_correct += 1
        else:
            follow_up_errors.append({
                "record_id": rid,
                "gold_follow_up_raw": gold_fu,
                "pred_follow_up_raw": pred_fu,
                "gold_follow_up_normalized": normalize_follow_up(gold_fu),
                "pred_follow_up_normalized": normalize_follow_up(pred_fu),
            })

        gold_vitals = safe_get(gold, "clinical", "vitals", default={})
        pred_vitals = safe_get(pred, "clinical", "vitals", default={})

        if vitals_equal(gold_vitals, pred_vitals):
            vitals_correct += 1
        else:
            vitals_errors.append({
                "record_id": rid,
                "gold_vitals_normalized": normalize_vitals_for_compare(gold_vitals),
                "pred_vitals_normalized": normalize_vitals_for_compare(pred_vitals),
            })

    reason_acc = reason_correct / n if n else 0.0
    follow_up_acc = follow_up_correct / n if n else 0.0
    vitals_acc = vitals_correct / n if n else 0.0

    int_p, int_r, int_f1 = macro_metric_over_records(
        pairs,
        lambda g: safe_get(g, "clinical", "interventions", default=[]),
        lambda p: safe_get(p, "clinical", "interventions", default=[]),
    )

    prob_p, prob_r, prob_f1 = macro_metric_over_records(
        pairs,
        lambda g: canonicalize_problem_list(safe_get(g, "coding", "problems_normalized", default=[])),
        lambda p: canonicalize_problem_list(safe_get(p, "coding", "problems_normalized", default=[])),
    )

    metrics = {
        "records_evaluated": n,
        "reason_for_visit_accuracy": round(reason_acc, 4),
        "follow_up_accuracy": round(follow_up_acc, 4),
        "vitals_exact_match_rate": round(vitals_acc, 4),
        "interventions_macro_precision": round(int_p, 4),
        "interventions_macro_recall": round(int_r, 4),
        "interventions_macro_f1": round(int_f1, 4),
        "problems_macro_precision": round(prob_p, 4),
        "problems_macro_recall": round(prob_r, 4),
        "problems_macro_f1": round(prob_f1, 4),
        "gold_only_ids": gold_only,
        "pred_only_ids": pred_only,
        "reason_mismatches_sample": reason_errors[:15],
        "follow_up_mismatches_sample": follow_up_errors[:15],
        "vitals_mismatches_sample": vitals_errors[:15],
    }

    metrics_path = REPORTS_DIR / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"Wrote {metrics_path}")
    print("\nEvaluation summary")
    print(f"- records evaluated: {n}")
    print(f"- reason_for_visit accuracy: {reason_acc:.2f}")
    print(f"- follow_up accuracy: {follow_up_acc:.2f}")
    print(f"- vitals exact match rate: {vitals_acc:.2f}")
    print(f"- interventions macro F1: {int_f1:.4f} (P={int_p:.4f}, R={int_r:.4f})")
    print(f"- problems macro F1: {prob_f1:.4f} (P={prob_p:.4f}, R={prob_r:.4f})")

    if gold_only or pred_only:
        print("\nDataset alignment warnings")
        print(f"- gold only ids: {len(gold_only)}")
        print(f"- pred only ids: {len(pred_only)}")


if __name__ == "__main__":
    main()