from difflib import SequenceMatcher
from src.resources.problem_lexicon import PROBLEM_VOCAB, SYNONYM_MAP

FUZZY_THRESHOLD = 0.90


def _contains_any(text: str, phrases: list[str]) -> bool:
    t = (text or "").lower()
    return any(p in t for p in phrases)


def _partial_ratio(phrase: str, text: str) -> float:
    phrase = (phrase or "").lower().strip()
    text = (text or "").lower().strip()

    if not phrase or not text:
        return 0.0

    if phrase in text:
        return 1.0

    if len(text) <= len(phrase):
        return SequenceMatcher(None, phrase, text).ratio()

    best = 0.0
    window = len(phrase)

    for i in range(0, len(text) - window + 1):
        chunk = text[i:i + window]
        score = SequenceMatcher(None, phrase, chunk).ratio()
        if score > best:
            best = score

    return best


def normalize_problems(text: str) -> list[str]:
    if not text:
        return []

    t = text.lower()
    found = set()

    # exact synonym mapping
    for phrase, norm in SYNONYM_MAP.items():
        if phrase in t:
            found.add(norm)

    # fuzzy synonym mapping
    for phrase, norm in SYNONYM_MAP.items():
        if norm in found:
            continue
        score = _partial_ratio(phrase, t)
        if score >= FUZZY_THRESHOLD:
            found.add(norm)

    # safe clinical rules
    if _contains_any(t, ["dolore", "algia", "nrs", "vas", "dolore cronico", "dolore al ginocchio"]):
        found.add("dolore_cronico")

    if _contains_any(t, ["caduta", "caduto", "scivolato", "post-caduta", "post caduta", "trauma"]):
        found.add("caduta")

    if _contains_any(t, [
        "piaga",
        "ferita",
        "ulcera",
        "decubito",
        "lesione da pressione",
        "lesione locale",
        "lesione cutanea",
    ]):
        found.add("lesione_da_pressione")

    if _contains_any(t, ["glicemia", "diabete", "dm2", "diabetico", "diabetica"]):
        found.add("diabete_tipo_2")

    if _contains_any(t, ["pressione alta", "ipertensione", "iperteso", "ipertesa", "pressione arteriosa elevata"]):
        found.add("ipertensione")

    if _contains_any(t, ["dispnea", "bpco", "bronchite cronica", "fiato corto", "quadro respiratorio"]):
        found.add("bpco")

    if _contains_any(t, ["scompenso", "insufficienza cardiaca", "edemi declivi", "ortopnea"]):
        found.add("scompenso_cardiaco")

    if _contains_any(t, [
        "scarso appetito",
        "inappetenza",
        "ridotto appetito",
        "mangia poco",
        "non mangia",
        "appetito ridotto",
    ]):
        found.add("malnutrizione")

    if _contains_any(t, ["disidratazione", "poca idratazione", "assunzione di liquidi ridotta", "beve poco"]):
        found.add("disidratazione")

    if _contains_any(t, ["rischio caduta", "instabilità posturale", "deambulazione incerta"]):
        found.add("rischio_caduta")

    if _contains_any(t, ["astenia", "debolezza generale", "stanchezza"]):
        found.add("astenia")

    if _contains_any(t, ["nausea"]):
        found.add("nausea")

    if _contains_any(t, ["capogiro", "vertigine", "vertigini"]):
        found.add("capogiro")

    # keep only allowed vocab
    found = {label for label in found if label in PROBLEM_VOCAB}
    return sorted(found)