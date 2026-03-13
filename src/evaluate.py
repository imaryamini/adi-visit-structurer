#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


VITAL_KEYS = [
    "blood_pressure_systolic",
    "blood_pressure_diastolic",
    "heart_rate",
    "temperature",
    "spo2",
]


def load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def normalize_scalar(x: Any) -> Any:
    if x is None:
        return None

    if isinstance(x, bool):
        return x

    if isinstance(x, (int, float)):
        return x

    if isinstance(x, dict):
        return {k: normalize_scalar(v) for k, v in sorted(x.items())}

    s = str(x).strip().lower()

    if s == "":
        return None

    # Try numeric normalization
    try:
        if "." in s or "," in s:
            return float(s.replace(",", "."))
        return int(s)
    except ValueError:
        pass

    return s


def safe_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, dict):
        return json.dumps(normalize_scalar(x), ensure_ascii=False, sort_keys=True)
    return str(normalize_scalar(x) if normalize_scalar(x) is not None else "").strip().lower()


def safe_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(normalize_scalar(i)).strip().lower() for i in x if normalize_scalar(i) is not None]
    val = normalize_scalar(x)
    return [str(val).strip().lower()] if val is not None else []


def set_metrics(pred: set[str], gold: set[str]) -> Tuple[float, float, float]:
    tp = len(pred & gold)
    fp = len(pred - gold)
    fn = len(gold - pred)

    precision = tp / (tp + fp) if (tp + fp) else (1.0 if len(gold) == 0 else 0.0)
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def normalize_vital_value(x: Any) -> Any:
    if x is None:
        return None

    if isinstance(x, str):
        s = x.strip().replace(",", ".")
        try:
            if "." in s:
                return float(s)
            return int(s)
        except ValueError:
            return s.lower()

    return x


def vitals_exact_match(pred_v: Dict[str, Any], gold_v: Dict[str, Any]) -> bool:
    for k in VITAL_KEYS:
        if normalize_vital_value(pred_v.get(k)) != normalize_vital_value(gold_v.get(k)):
            return False
    return True


def diff_vitals(pred_v: Dict[str, Any], gold_v: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    diffs: Dict[str, Dict[str, Any]] = {}
    for k in VITAL_KEYS:
        pv = normalize_vital_value(pred_v.get(k))
        gv = normalize_vital_value(gold_v.get(k))
        if pv != gv:
            diffs[k] = {"pred": pv, "gold": gv}
    return diffs


def round4(x: float) -> float:
    return round(x, 4)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate ADI predictions vs gold.")
    ap.add_argument("--gold_dir", default="data/synthetic/gold", help="Gold JSON folder")
    ap.add_argument("--pred_dir", default="data/synthetic/pred", help="Prediction JSON folder")
    ap.add_argument("--pattern", default="ADI-*.json", help="Filename pattern")
    ap.add_argument("--out", default="reports/metrics.json", help="Output metrics path")
    args = ap.parse_args()

    gold_dir = Path(args.gold_dir)
    pred_dir = Path(args.pred_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    gold_files = sorted(gold_dir.glob(args.pattern))
    pred_files = sorted(pred_dir.glob(args.pattern))

    gold_ids = {p.stem for p in gold_files}
    pred_ids = {p.stem for p in pred_files}
    common_ids = sorted(gold_ids & pred_ids)

    gold_only = sorted(gold_ids - pred_ids)
    pred_only = sorted(pred_ids - gold_ids)

    if not common_ids:
        metrics = {
            "summary": {
                "n_records": 0,
                "text_field_accuracy": {
                    "clinical.reason_for_visit": 0.0,
                    "clinical.follow_up": 0.0,
                },
                "vitals_exact_match_rate": 0.0,
                "list_f1_macro": {
                    "clinical.interventions": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
                    "coding.problems_normalized": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
                },
            },
            "per_record": {},
            "dataset_alignment": {
                "gold_only_ids": gold_only,
                "pred_only_ids": pred_only,
            },
            "debug": {
                "gold_dir": str(gold_dir),
                "pred_dir": str(pred_dir),
                "n_gold": len(gold_files),
                "n_pred": len(pred_files),
                "common_ids": 0,
            },
        }
        out_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote {out_path}")
        print("No matching prediction/gold pairs found.")
        return

    reason_ok = 0
    follow_ok = 0
    vitals_ok = 0

    int_p_sum = int_r_sum = int_f1_sum = 0.0
    prob_p_sum = prob_r_sum = prob_f1_sum = 0.0

    per_record: Dict[str, Any] = {}

    reason_mismatches: List[str] = []
    follow_mismatches: List[str] = []
    vitals_mismatches: List[str] = []

    for rid in common_ids:
        gold = load_json(gold_dir / f"{rid}.json")
        pred = load_json(pred_dir / f"{rid}.json")

        gold_cl = gold.get("clinical", {}) or {}
        pred_cl = pred.get("clinical", {}) or {}

        g_reason = safe_str(gold_cl.get("reason_for_visit"))
        p_reason = safe_str(pred_cl.get("reason_for_visit"))
        reason_match = (g_reason == p_reason)
        if reason_match:
            reason_ok += 1
        else:
            reason_mismatches.append(rid)

        g_follow = safe_str(gold_cl.get("follow_up"))
        p_follow = safe_str(pred_cl.get("follow_up"))
        follow_match = (g_follow == p_follow)
        if follow_match:
            follow_ok += 1
        else:
            follow_mismatches.append(rid)

        g_v = gold_cl.get("vitals", {}) or {}
        p_v = pred_cl.get("vitals", {}) or {}
        v_ok = vitals_exact_match(p_v, g_v)
        if v_ok:
            vitals_ok += 1
        else:
            vitals_mismatches.append(rid)

        g_int = set(safe_list(gold_cl.get("interventions")))
        p_int = set(safe_list(pred_cl.get("interventions")))
        ip, ir, if1 = set_metrics(p_int, g_int)
        int_p_sum += ip
        int_r_sum += ir
        int_f1_sum += if1

        gold_cod = gold.get("coding", {}) or {}
        pred_cod = pred.get("coding", {}) or {}
        g_prob = set(safe_list(gold_cod.get("problems_normalized")))
        p_prob = set(safe_list(pred_cod.get("problems_normalized")))
        pp, pr, pf1 = set_metrics(p_prob, g_prob)
        prob_p_sum += pp
        prob_r_sum += pr
        prob_f1_sum += pf1

        per_record[rid] = {
            "text_match": {
                "clinical.reason_for_visit": reason_match,
                "clinical.follow_up": follow_match,
            },
            "text_values": {
                "clinical.reason_for_visit": {"pred": p_reason, "gold": g_reason},
                "clinical.follow_up": {"pred": p_follow, "gold": g_follow},
            },
            "vitals_exact_match": v_ok,
            "vitals_diff": diff_vitals(p_v, g_v),
            "set_comparison": {
                "clinical.interventions": {
                    "pred": sorted(p_int),
                    "gold": sorted(g_int),
                    "missing_from_pred": sorted(g_int - p_int),
                    "extra_in_pred": sorted(p_int - g_int),
                },
                "coding.problems_normalized": {
                    "pred": sorted(p_prob),
                    "gold": sorted(g_prob),
                    "missing_from_pred": sorted(g_prob - p_prob),
                    "extra_in_pred": sorted(p_prob - g_prob),
                },
            },
            "f1_macro": {
                "clinical.interventions": {
                    "precision": round4(ip),
                    "recall": round4(ir),
                    "f1": round4(if1),
                },
                "coding.problems_normalized": {
                    "precision": round4(pp),
                    "recall": round4(pr),
                    "f1": round4(pf1),
                },
            },
        }

    n = len(common_ids)

    metrics = {
        "summary": {
            "n_records": n,
            "text_field_accuracy": {
                "clinical.reason_for_visit": round4(reason_ok / n),
                "clinical.follow_up": round4(follow_ok / n),
            },
            "vitals_exact_match_rate": round4(vitals_ok / n),
            "list_f1_macro": {
                "clinical.interventions": {
                    "precision": round4(int_p_sum / n),
                    "recall": round4(int_r_sum / n),
                    "f1": round4(int_f1_sum / n),
                },
                "coding.problems_normalized": {
                    "precision": round4(prob_p_sum / n),
                    "recall": round4(prob_r_sum / n),
                    "f1": round4(prob_f1_sum / n),
                },
            },
        },
        "error_analysis": {
            "reason_mismatches": reason_mismatches,
            "follow_up_mismatches": follow_mismatches,
            "vitals_mismatches": vitals_mismatches,
        },
        "per_record": per_record,
        "dataset_alignment": {
            "gold_only_ids": gold_only,
            "pred_only_ids": pred_only,
        },
        "debug": {
            "gold_dir": str(gold_dir),
            "pred_dir": str(pred_dir),
            "n_gold": len(gold_files),
            "n_pred": len(pred_files),
            "common_ids": n,
        },
    }

    out_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")

    print("\nEvaluation summary")
    print(f"- records evaluated: {n}")
    print(f"- reason_for_visit accuracy: {round4(reason_ok / n)}")
    print(f"- follow_up accuracy: {round4(follow_ok / n)}")
    print(f"- vitals exact match rate: {round4(vitals_ok / n)}")
    print(
        f"- interventions macro F1: {round4(int_f1_sum / n)} "
        f"(P={round4(int_p_sum / n)}, R={round4(int_r_sum / n)})"
    )
    print(
        f"- problems macro F1: {round4(prob_f1_sum / n)} "
        f"(P={round4(prob_p_sum / n)}, R={round4(prob_r_sum / n)})"
    )

    if gold_only or pred_only:
        print("\nDataset alignment warnings")
        print(f"- gold only ids: {len(gold_only)}")
        print(f"- pred only ids: {len(pred_only)}")


if __name__ == "__main__":
    main()