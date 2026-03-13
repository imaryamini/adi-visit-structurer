# src/schema.py

from typing import Any, Dict, List


def _str_or_none(x: Any) -> str | None:
    if isinstance(x, str):
        s = x.strip()
        return s if s else None
    return None


def _num_or_none(x: Any) -> int | float | None:
    if isinstance(x, (int, float)):
        return x
    return None


def _list_of_str(x: Any) -> list[str]:
    if not isinstance(x, list):
        return []
    out: list[str] = []
    for i in x:
        s = str(i).strip()
        if s:
            out.append(s)
    return out


def coerce_llm_output(out: Any) -> Dict[str, Any]:
    """
    Coerce arbitrary parsed JSON into the expected shape.
    Drops unknown structures, ensures nested keys exist, coerces types.
    """
    if not isinstance(out, dict):
        out = {}

    clinical = out.get("clinical")
    clinical = clinical if isinstance(clinical, dict) else {}

    coding = out.get("coding")
    coding = coding if isinstance(coding, dict) else {}

    vitals = clinical.get("vitals")
    vitals = vitals if isinstance(vitals, dict) else {}

    coerced = {
        "clinical": {
            "reason_for_visit": _str_or_none(clinical.get("reason_for_visit")),
            "follow_up": _str_or_none(clinical.get("follow_up")),
            "interventions": _list_of_str(clinical.get("interventions")),
            "vitals": {
                "blood_pressure_systolic": _num_or_none(vitals.get("blood_pressure_systolic")),
                "blood_pressure_diastolic": _num_or_none(vitals.get("blood_pressure_diastolic")),
                "heart_rate": _num_or_none(vitals.get("heart_rate")),
                "temperature": _num_or_none(vitals.get("temperature")),
                "spo2": _num_or_none(vitals.get("spo2")),
            },
        },
        "coding": {
            "problems_normalized": _list_of_str(coding.get("problems_normalized"))
        },
    }
    return coerced