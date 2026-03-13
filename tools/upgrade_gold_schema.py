import json
from pathlib import Path

GOLD_DIR = Path("data/synthetic/gold")

DEFAULT_VITALS = {
    "blood_pressure_systolic": None,
    "blood_pressure_diastolic": None,
    "heart_rate": None,
    "temperature": None,
    "spo2": None,
}

def upgrade_one(path: Path):
    rid = path.stem  # ADI-0009
    data = json.loads(path.read_text(encoding="utf-8"))

    # If already full schema, skip
    if all(k in data for k in ["meta", "patient", "clinical", "coding", "quality"]):
        return False

    clinical = data.get("clinical", {})
    coding = data.get("coding", {})

    # Make sure vitals has all keys
    vitals = clinical.get("vitals", {})
    merged_vitals = dict(DEFAULT_VITALS)
    if isinstance(vitals, dict):
        merged_vitals.update(vitals)
    clinical["vitals"] = merged_vitals

    upgraded = {
        "meta": {
            "record_id": rid,
            "template_type": ["diario_clinico"],
            "visit_datetime": None,          # we keep None if unknown; you can fill later from raw if you want
            "operator_role": "infermiere",
        },
        "patient": {
            "patient_id": f"SYNTH-{rid}",
            "age": None,
            "sex": None,
        },
        "clinical": {
            "reason_for_visit": clinical.get("reason_for_visit"),
            "anamnesis_brief": clinical.get("anamnesis_brief", []),
            "vitals": clinical["vitals"],
            "interventions": clinical.get("interventions", []),
            "critical_issues": clinical.get("critical_issues", []),
            "follow_up": clinical.get("follow_up"),
        },
        "coding": {
            "problems_normalized": coding.get("problems_normalized", []),
            **({"risk_flags": coding["risk_flags"]} if "risk_flags" in coding else {}),
        },
        "quality": {
            "missing_mandatory_fields": [],
            "warnings": [],
        }
    }

    path.write_text(json.dumps(upgraded, ensure_ascii=False, indent=2), encoding="utf-8")
    return True

def main():
    changed = 0
    for p in GOLD_DIR.glob("ADI-*.json"):
        if upgrade_one(p):
            changed += 1
    print(f"Upgraded {changed} gold files.")

if __name__ == "__main__":
    main()
    