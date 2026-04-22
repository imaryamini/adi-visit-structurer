from __future__ import annotations

from typing import Iterable, List, Optional


def _clean(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


REASON_MAP = {
    # pain / symptoms
    "dolore toracico": "dolore toracico",
    "dolore lombare": "dolore lombare",
    "lombalgia": "dolore lombare",
    "dolore addominale": "dolore addominale",
    "dispnea": "dispnea",
    "affanno": "dispnea",
    "febbre": "febbre",
    "tosse febbre e lieve dispnea": "tosse, febbre e lieve dispnea",
    "tosse, febbre e lieve dispnea": "tosse, febbre e lieve dispnea",
    "valutazione dolore cronico": "valutazione dolore cronico",
    "valutazione dolore e controllo parametri": "valutazione dolore e controllo parametri",
    "rivalutazione dolore": "rivalutazione dolore",
    "stanchezza e scarso appetito": "stanchezza e scarso appetito",
    "riferiti sintomi generali": "riferiti sintomi generali",

    # parameter monitoring
    "controllo parametri": "controllo parametri",
    "controllo parametri vitali": "controllo parametri",
    "monitoraggio parametri": "controllo parametri",
    "monitoraggio parametri vitali": "controllo parametri",
    "monitoraggio dei parametri vitali": "controllo parametri",

    # general visit
    "valutazione generale": "valutazione generale",
    "controllo generale": "controllo generale",
    "valutazione clinica generale": "valutazione generale",

    # wound / dressing
    "medicazione": "medicazione e controllo lesione",
    "controllo lesione": "medicazione e controllo lesione",
    "medicazione lesione": "medicazione e controllo lesione",
    "medicazione e controllo lesione": "medicazione e controllo lesione",
    "medicazione piaga da decubito": "medicazione piaga da decubito",

    # therapy
    "somministrazione terapia": "controllo terapia e somministrazione farmaco",
    "somministrazione farmaco": "controllo terapia e somministrazione farmaco",
    "controllo terapia": "controllo terapia e somministrazione farmaco",
    "controllo terapia/somministrazione farmaco": "controllo terapia e somministrazione farmaco",
    "controllo terapia e somministrazione farmaco": "controllo terapia e somministrazione farmaco",

    # fall
    "recente caduta domestica": "controllo post-caduta",
    "caduta recente": "controllo post-caduta",
    "controllo post caduta": "controllo post-caduta",
    "controllo post-caduta": "controllo post-caduta",
}


INTERVENTION_MAP = {
    # general
    "valutazione generale": "valutazione generale",
    "valutazione clinica": "valutazione generale",
    "valutazione clinica generale": "valutazione generale",
    "controllo generale": "valutazione generale",

    # vitals
    "monitoraggio parametri": "monitoraggio parametri vitali",
    "monitoraggio parametri vitali": "monitoraggio parametri vitali",
    "monitoraggio dei parametri": "monitoraggio parametri vitali",
    "monitoraggio dei parametri vitali": "monitoraggio parametri vitali",
    "controllo parametri": "monitoraggio parametri vitali",
    "controllo parametri vitali": "monitoraggio parametri vitali",
    "rilevati parametri": "monitoraggio parametri vitali",

    # meds
    "somministrazione terapia": "somministrazione farmaco",
    "somministrazione farmaco": "somministrazione farmaco",
    "somministrata terapia": "somministrazione farmaco",
    "terapia": "somministrazione farmaco",
    "controllo terapia": "somministrazione farmaco",

    # wound care
    "medicazione": "medicazione",
    "medicazione lesione": "medicazione",
    "controllo lesione": "medicazione",

    # caregiver / education
    "educazione caregiver": "educazione caregiver",
    "educazione sanitaria": "educazione caregiver",
    "consigli al caregiver": "educazione caregiver",

    # respiratory
    "monitoraggio respiratorio": "monitoraggio respiratorio",
    "controllo respiratorio": "monitoraggio respiratorio",
    "valutazione respiratoria": "monitoraggio respiratorio",

    # weaker aliases
    "valutazione dolore": "valutazione generale",
}


PROBLEM_MAP = {
    "dolore toracico": "dolore_toracico",
    "dispnea": "dispnea",
    "affanno": "dispnea",
    "febbre": "febbre",
    "ferita": "lesione_cutanea",
    "lesione": "lesione_cutanea",
    "ulcera": "lesione_cutanea",
    "piaga": "lesione_cutanea",
    "decubito": "lesione_cutanea",
    "caduta": "caduta_recente",
    "caduta recente": "caduta_recente",
    "lombalgia": "dolore_lombare",
    "dolore lombare": "dolore_lombare",
    "dolore addominale": "dolore_addominale",
    "ipertensione": "ipertensione",
    "bpco": "bpco",
    "scompenso cardiaco": "scompenso_cardiaco",
    "diabete": "diabete_tipo_2",
    "diabete tipo 2": "diabete_tipo_2",
    "astenia": "astenia",
    "nausea": "nausea",
    "capogiro": "capogiro",
    "inappetenza": "malnutrizione",
    "scarso appetito": "malnutrizione",
    "ridotto appetito": "malnutrizione",
    "disidratazione": "disidratazione",
}


def _unique_keep_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        key = _clean(item)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def normalize_reason(reason: Optional[str]) -> Optional[str]:
    if not reason:
        return None

    key = _clean(reason)
    if not key:
        return None

    if key in REASON_MAP:
        return REASON_MAP[key]

    if any(x in key for x in ["lesione", "ferita", "ulcera", "piaga", "medicazione", "decubito"]):
        if "decubito" in key or "piaga" in key:
            return "medicazione piaga da decubito"
        return "medicazione e controllo lesione"

    if "dolore lombare" in key or "lombalgia" in key:
        return "dolore lombare"
    if "dolore toracico" in key:
        return "dolore toracico"
    if "dolore addominale" in key:
        return "dolore addominale"
    if "dolore" in key and "parametri" in key:
        return "valutazione dolore e controllo parametri"
    if "dolore" in key:
        return "rivalutazione dolore"

    if "dispnea" in key or "affanno" in key:
        return "dispnea"
    if "febbre" in key and "tosse" in key:
        return "tosse, febbre e lieve dispnea"
    if "febbre" in key:
        return "febbre"

    if "caduta" in key:
        return "controllo post-caduta"

    if any(x in key for x in ["terapia", "farmaco", "somministrazione"]):
        return "controllo terapia e somministrazione farmaco"

    if any(x in key for x in ["parametri", "pressione", "frequenza cardiaca", "spo2", "saturazione"]):
        return "controllo parametri"

    if any(x in key for x in ["stanchezza", "astenia", "scarso appetito", "inappetenza", "ridotto appetito", "sintomi generali"]):
        return "riferiti sintomi generali"

    if "controllo generale" in key:
        return "controllo generale"
    if "valutazione generale" in key:
        return "valutazione generale"

    return key


def normalize_interventions(interventions: Optional[Iterable[str]]) -> List[str]:
    if not interventions:
        return []

    normalized: List[str] = []

    for item in interventions:
        key = _clean(str(item))
        if not key:
            continue

        mapped = INTERVENTION_MAP.get(key)

        if mapped is None:
            if any(x in key for x in ["parametri", "pressione", "frequenza cardiaca", "spo2", "saturazione", "temperatura"]):
                mapped = "monitoraggio parametri vitali"
            elif any(x in key for x in ["medicazione", "ferita", "lesione", "ulcera", "piaga"]):
                mapped = "medicazione"
            elif any(x in key for x in ["somministrazione", "farmaco", "terapia"]):
                mapped = "somministrazione farmaco"
            elif any(x in key for x in ["caregiver", "educazione sanitaria", "istruito"]):
                mapped = "educazione caregiver"
            elif any(x in key for x in ["respiratorio", "saturazione", "dispnea"]):
                mapped = "monitoraggio respiratorio"
            elif any(x in key for x in ["valutazione", "controllo generale"]):
                mapped = "valutazione generale"
            else:
                mapped = key

        normalized.append(mapped)

    normalized = _unique_keep_order(normalized)

    final: List[str] = []
    has_vitals = "monitoraggio parametri vitali" in normalized
    has_general = "valutazione generale" in normalized
    has_respiratory = "monitoraggio respiratorio" in normalized

    for item in normalized:
        if has_vitals and item in {"monitoraggio parametri", "controllo parametri", "controllo parametri vitali"}:
            continue
        if has_general and item in {"valutazione clinica", "controllo generale"}:
            continue
        if has_respiratory and item in {"controllo respiratorio", "valutazione respiratoria"}:
            continue
        final.append(item)

    return _unique_keep_order(final)


def normalize_problems(text_or_items) -> List[str]:
    """
    Supports either:
    - a free-text string
    - a list of raw problem labels
    """
    if text_or_items is None:
        return []

    found: List[str] = []

    if isinstance(text_or_items, str):
        t = _clean(text_or_items)

        for raw, mapped in PROBLEM_MAP.items():
            if raw in t:
                found.append(mapped)

        if (
            "dolore" in t
            and "dolore_lombare" not in found
            and "dolore_toracico" not in found
            and "dolore_addominale" not in found
        ):
            found.append("dolore_generico")

        if "dispnea" in t or "affanno" in t:
            if "dispnea" not in found:
                found.append("dispnea")

        if "stanchezza" in t or "astenia" in t or "debolezza" in t:
            if "astenia" not in found:
                found.append("astenia")

        if "scarso appetito" in t or "inappetenza" in t or "ridotto appetito" in t:
            if "malnutrizione" not in found:
                found.append("malnutrizione")

        if "febbre" in t and "febbre" not in found:
            found.append("febbre")

        return _unique_keep_order(found)

    for item in text_or_items:
        key = _clean(str(item))
        if not key:
            continue

        mapped = PROBLEM_MAP.get(key)
        if mapped is None:
            if "dolore lombare" in key or "lombalgia" in key:
                mapped = "dolore_lombare"
            elif "dolore toracico" in key:
                mapped = "dolore_toracico"
            elif "dolore addominale" in key:
                mapped = "dolore_addominale"
            elif "dolore" in key:
                mapped = "dolore_generico"
            elif any(x in key for x in ["ferita", "lesione", "ulcera", "piaga", "decubito"]):
                mapped = "lesione_cutanea"
            elif "caduta" in key:
                mapped = "caduta_recente"
            elif "dispnea" in key or "affanno" in key:
                mapped = "dispnea"
            elif "febbre" in key:
                mapped = "febbre"
            elif "ipertension" in key or "pressione alta" in key:
                mapped = "ipertensione"
            elif "bpco" in key:
                mapped = "bpco"
            elif "scompenso" in key:
                mapped = "scompenso_cardiaco"
            elif "diabete" in key or "glicemia" in key:
                mapped = "diabete_tipo_2"
            elif "astenia" in key or "stanchezza" in key or "debolezza" in key:
                mapped = "astenia"
            elif "nausea" in key:
                mapped = "nausea"
            elif "capogiro" in key or "vertigine" in key:
                mapped = "capogiro"
            elif "inappetenza" in key or "scarso appetito" in key or "ridotto appetito" in key:
                mapped = "malnutrizione"
            elif "disidrata" in key:
                mapped = "disidratazione"
            else:
                mapped = key

        found.append(mapped)

    return _unique_keep_order(found)