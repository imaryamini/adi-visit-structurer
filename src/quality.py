# src/quality.py

MANDATORY_FIELDS = [
    "meta.visit_datetime",
    "meta.operator_role",
    "clinical.reason_for_visit",
    # interventions is not mandatory: some notes can be very short/minimal
]


def quality_check(output: dict) -> dict:
    missing = []
    warnings = []

    def get_path(d, path):
        cur = d
        for part in path.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return None
            cur = cur[part]
        return cur

    def has_value(value):
        return value not in (None, "", [])

    def normalize_vitals(vitals: dict) -> dict:
        vitals = vitals or {}

        systolic = vitals.get("blood_pressure_systolic")
        diastolic = vitals.get("blood_pressure_diastolic")
        combined = vitals.get("blood_pressure")

        if (systolic in [None, ""] or diastolic in [None, ""]) and isinstance(combined, str) and "/" in combined:
            left, right = combined.split("/", 1)
            left = left.strip()
            right = right.strip()
            if systolic in [None, ""]:
                systolic = left or None
            if diastolic in [None, ""]:
                diastolic = right or None

        if not has_value(combined) and has_value(systolic) and has_value(diastolic):
            combined = f"{systolic}/{diastolic}"

        return {
            "blood_pressure": combined,
            "blood_pressure_systolic": systolic,
            "blood_pressure_diastolic": diastolic,
            "heart_rate": vitals.get("heart_rate"),
            "temperature": vitals.get("temperature"),
            "spo2": vitals.get("spo2"),
        }

    # Mandatory fields
    for path in MANDATORY_FIELDS:
        v = get_path(output, path)
        if v is None or v == "" or v == []:
            missing.append(path)

    clinical = output.get("clinical", {}) or {}
    interventions = clinical.get("interventions", []) or []
    vit = normalize_vitals(clinical.get("vitals", {}) or {})
    follow_up = clinical.get("follow_up")
    reason = clinical.get("reason_for_visit")

    any_vital = any(
        has_value(vit.get(k))
        for k in [
            "blood_pressure",
            "blood_pressure_systolic",
            "blood_pressure_diastolic",
            "heart_rate",
            "temperature",
            "spo2",
        ]
    )

    # Warning: no interventions extracted
    if interventions == []:
        warnings.append("No interventions extracted")

    # Warning: no vitals extracted at all
    if not any_vital:
        warnings.append("No vital signs extracted")

    # Consistency warning: vitals monitoring present but no vitals extracted
    if "monitoraggio_parametri_vitali" in interventions or "monitoraggio parametri vitali" in interventions:
        if not any_vital:
            warnings.append("Vitals monitoring present but no vitals extracted")

    # More specific soft warnings
    if reason == "controllo parametri" and not has_value(vit.get("blood_pressure")):
        warnings.append("Reason suggests parameter monitoring but blood pressure is missing")

    if reason == "controllo parametri" and not has_value(vit.get("heart_rate")):
        warnings.append("Reason suggests parameter monitoring but heart rate is missing")

    # Soft warning: follow-up missing
    if follow_up in [None, "", []]:
        warnings.append("Follow-up not specified")

    return {
        "missing_mandatory_fields": missing,
        "warnings": warnings,
    }