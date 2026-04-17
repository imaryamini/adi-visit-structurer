from __future__ import annotations

from typing import Iterable, List, Optional


def _clean(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


REASON_MAP = {
    "dolore toracico": "dolore toracico",
    "dolore lombare": "dolore lombare",
    "lombalgia": "dolore lombare",
    "dolore addominale": "dolore addominale",
    "dispnea": "dispnea",
    "affanno": "dispnea",
    "febbre": "febbre",
    "controllo parametri": "controllo parametri",
    "controllo parametri vitali": "controllo parametri",
    "monitoraggio parametri": "controllo parametri",
    "monitoraggio parametri vitali": "controllo parametri",
    "valutazione generale": "valutazione generale",
    "medicazione": "medicazione e controllo lesione",
    "controllo lesione": "medicazione e controllo lesione",
    "medicazione e controllo lesione": "medicazione e controllo lesione",
    "somministrazione terapia": "controllo terapia e somministrazione farmaco",
    "somministrazione farmaco": "controllo terapia e somministrazione farmaco",
    "controllo terapia": "controllo terapia e somministrazione farmaco",
    "recente caduta domestica": "controllo post-caduta",
    "caduta recente": "controllo post-caduta",
    "controllo post-caduta": "controllo post-caduta",
}


INTERVENTION_MAP = {
    "valutazione generale": "valutazione generale",
    "valutazione clinica": "valutazione generale",
    "monitoraggio parametri": "monitoraggio parametri vitali",
    "monitoraggio parametri vitali": "monitoraggio parametri vitali",
    "controllo parametri": "monitoraggio parametri vitali",
    "controllo parametri vitali": "monitoraggio parametri vitali",
    "somministrazione terapia": "somministrazione farmaco",
    "somministrazione farmaco": "somministrazione farmaco",
    "terapia": "somministrazione farmaco",
    "medicazione": "medicazione",
    "medicazione lesione": "medicazione",
    "valutazione dolore": "valutazione generale",
}


PROBLEM_MAP = {
    "dolore toracico": "dolore_toracico",
    "dispnea": "dispnea",
    "febbre": "febbre",
    "ferita": "lesione_cutanea",
    "lesione": "lesione_cutanea",
    "ulcera": "lesione_cutanea",
    "piaga": "lesione_cutanea",
    "caduta": "caduta_recente",
    "caduta recente": "caduta_recente",
    "lombalgia": "dolore_lombare",
    "dolore lombare": "dolore_lombare",
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
    return REASON_MAP.get(key, key)


def normalize_interventions(interventions: Optional[Iterable[str]]) -> List[str]:
    if not interventions:
        return []

    normalized: List[str] = []
    for item in interventions:
        key = _clean(str(item))
        if not key:
            continue
        mapped = INTERVENTION_MAP.get(key, key)
        normalized.append(mapped)

    normalized = _unique_keep_order(normalized)

    # extra conservative merge:
    # if monitoraggio parametri vitali exists, remove weaker variants
    final: List[str] = []
    has_vitals = "monitoraggio parametri vitali" in normalized
    for item in normalized:
        if has_vitals and item in {"monitoraggio parametri", "controllo parametri"}:
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
        return _unique_keep_order(found)

    for item in text_or_items:
        key = _clean(str(item))
        if not key:
            continue
        found.append(PROBLEM_MAP.get(key, key))

    return _unique_keep_order(found)