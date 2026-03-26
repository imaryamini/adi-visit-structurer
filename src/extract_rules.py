import re
from datetime import datetime
from typing import Any, Optional


def _clean_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _sentences(text: str) -> list[str]:
    text = (text or "").replace("\n", ". ")
    parts = re.split(r"[.;]\s+|\n+", text)
    return [_clean_spaces(p) for p in parts if _clean_spaces(p)]


def _contains_any(text: str, patterns: list[str]) -> bool:
    t = (text or "").lower()
    return any(re.search(p, t, flags=re.IGNORECASE) for p in patterns)


def extract_datetime(text: str) -> Optional[str]:
    patterns = [
        r"(\d{1,2}/\d{1,2}/\d{4})\s*(?:ore|alle)?\s*(\d{1,2}:\d{2})",
        r"(\d{1,2}-\d{1,2}-\d{4})\s*(?:ore|alle)?\s*(\d{1,2}:\d{2})",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if not m:
            continue
        date_part = m.group(1).replace("-", "/")
        time_part = m.group(2)
        try:
            dt = datetime.strptime(f"{date_part} {time_part}", "%d/%m/%Y %H:%M")
            return dt.isoformat()
        except Exception:
            continue
    return None


def extract_bp(text: str):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    bp_patterns = [
        re.compile(r"\bPA\s*[:=]?\s*(\d{2,3})\s*/\s*(\d{2,3})\b", re.IGNORECASE),
        re.compile(r"\bPA\s*[:=]?\s*(\d{2,3})\s*-\s*(\d{2,3})\b", re.IGNORECASE),
        re.compile(r"\bpressione(?: arteriosa)?(?: sistolica/diastolica)?\s*[:=]?\s*(\d{2,3})\s*/\s*(\d{2,3})\b", re.IGNORECASE),
        re.compile(r"\bpressione arteriosa\s*(\d{2,3})/(\d{2,3})\b", re.IGNORECASE),
        re.compile(r"\b(\d{2,3})\s*/\s*(\d{2,3})\s*(?:mmhg)?\b", re.IGNORECASE),
        re.compile(r"\b(\d{2,3})\s*-\s*(\d{2,3})\s*(?:mmhg)?\b", re.IGNORECASE),
    ]

    verbose_pat = re.compile(
        r"pressione arteriosa(?: di)?\s*(\d{2,3})\s*mmhg\s*\(sistolica\)\s*e\s*(\d{2,3})\s*mmhg\s*\(diastolica\)",
        re.IGNORECASE,
    )

    for line in lines:
        m = verbose_pat.search(line)
        if m:
            sys_v = int(m.group(1))
            dia_v = int(m.group(2))
            if 70 <= sys_v <= 250 and 40 <= dia_v <= 150 and sys_v > dia_v:
                return sys_v, dia_v

        for pat in bp_patterns:
            m = pat.search(line)
            if m:
                sys_v = int(m.group(1))
                dia_v = int(m.group(2))
                if 70 <= sys_v <= 250 and 40 <= dia_v <= 150 and sys_v > dia_v:
                    return sys_v, dia_v

    return None, None


def extract_hr(text: str) -> Optional[int]:
    patterns = [
        r"\bFC\s*[:=]?\s*(\d{2,3})\b",
        r"\bHR\s*[:=]?\s*(\d{2,3})\b",
        r"\bfrequenza cardiaca(?: di)?\s*(\d{2,3})\b",
        r"\b(\d{2,3})\s*bpm\b",
        r"\b(\d{2,3})\s*battiti al minuto\b",
    ]
    for p in patterns:
        m = re.search(p, text, flags=re.IGNORECASE)
        if m:
            v = int(m.group(1))
            if 30 <= v <= 220:
                return v
    return None


def extract_temp(text: str) -> Optional[float]:
    patterns = [
        r"\btemperatura(?: corporea)?\s*[:=]?\s*([0-9]{1,2}[.,][0-9])\b",
        r"\btemp\s*[:=]?\s*([0-9]{1,2}[.,][0-9])\b",
        r"\bT\s*[:=]?\s*([0-9]{1,2}[.,][0-9])\b",
        r"\b([0-9]{1,2}[.,][0-9])\s*°\s*C\b",
        r"\b([0-9]{1,2}[.,][0-9])\s*°C\b",
    ]
    for p in patterns:
        m = re.search(p, text, flags=re.IGNORECASE)
        if m:
            val = float(m.group(1).replace(",", "."))
            if 30.0 <= val <= 43.0:
                return val
    return None


def extract_spo2(text: str) -> Optional[int]:
    patterns = [
        r"\bSpO2\s*[:=]?\s*(\d{2,3})\s*%?\b",
        r"\bSatO2\s*[:=]?\s*(\d{2,3})\s*%?\b",
        r"\bsaturazione(?: di ossigeno)?\s*[:=]?\s*(\d{2,3})\s*%?\b",
        r"\bsat\.?\s*[:=]?\s*(\d{2,3})\s*%?\b",
    ]
    for p in patterns:
        m = re.search(p, text, flags=re.IGNORECASE)
        if m:
            v = int(m.group(1))
            if 50 <= v <= 100:
                return v
    return None


def _reason_from_keywords(text: str) -> Optional[str]:
    t = (text or "").lower()

    # Most specific first
    if "lesione da pressione" in t:
        return "medicazione lesione da pressione"

    if any(k in t for k in ["piaga da decubito", "lesione da decubito", "decubito"]):
        return "medicazione piaga da decubito"

    if "dolore al ginocchio destro" in t:
        return "dolore al ginocchio destro"

    if "dolore cronico" in t:
        return "valutazione dolore cronico"

    if any(k in t for k in ["stanchezza", "debolezza generale", "scarso appetito", "ridotto appetito"]):
        return "stanchezza e scarso appetito"

    if any(k in t for k in ["monitoraggio dei segni vitali", "segni vitali"]) and any(
        k in t for k in ["verifica della terapia", "verifica terapia", "terapia"]
    ):
        return "monitoraggio segni vitali e verifica terapia"

    if "pressione arteriosa" in t and any(k in t for k in ["terapia", "farmaco", "somministrazione"]):
        return "controllo pressione e rivalutazione terapia"

    if any(k in t for k in ["catetere vescicale", "catetere", "presidio urinario", "vescicale"]):
        return "controllo e gestione catetere"

    if any(k in t for k in ["presidio stomale", "cute peristomale", "stomia", "colostomia", "ileostomia"]):
        return "controllo e gestione stomia"

    if any(k in t for k in ["ossigenoterapia", "o2 terapia", "o2", "controllo respiratorio", "rivalutazione respiratoria"]):
        return "controllo respiratorio e gestione ossigenoterapia"

    if any(k in t for k in ["caduta recente", "post-caduta", "caduta", "scivolato", "trauma recente"]):
        return "rivalutazione caduta recente"

    if any(k in t for k in [
        "cambio di medicazione",
        "cambio medicazione",
        "medicazione",
        "valutata la lesione",
        "rivalutazione della lesione",
        "controllo della lesione",
        "lesione già trattata",
        "lesione locale",
        "ferita",
        "ulcera",
        "piaga",
        "lesione",
    ]):
        return "medicazione/controllo lesione"

    if any(k in t for k in ["dolore", "algia", "sintomatologia algica", "nrs", "vas"]):
        if any(k2 in t for k2 in [
            "parametri vitali",
            "controlli dei parametri vitali",
            "monitoraggio dei parametri",
            "pressione arteriosa",
            "frequenza cardiaca",
            "temperatura corporea",
        ]):
            return "valutazione dolore e controllo parametri"
        return "rivalutazione dolore"

    if any(k in t for k in ["astenia", "inappetenza", "nausea", "capogiro", "sintomi aspecifici", "sintomatologia generale"]):
        return "riferiti sintomi generali"

    if any(k in t for k in ["educazione del caregiver", "istruzione del caregiver", "supporto al caregiver", "caregiver", "familiare presente", "familiare"]):
        return "educazione caregiver e controllo generale"

    if any(k in t for k in [
        "monitoraggio dei parametri vitali",
        "monitoraggio dei parametri",
        "controlli dei parametri vitali",
        "controllo dei parametri vitali",
        "controllo parametri vitali",
        "rilevazione dei parametri vitali",
        "rilevazione dei parametri",
        "rilevati i seguenti parametri",
        "parametri vitali",
        "controllo parametri",
    ]):
        return "controllo parametri"

    if any(k in t for k in ["valutazione delle condizioni generali", "condizioni generali"]):
        return "controllo generale"

    if any(k in t for k in ["controllo generale", "valutazione generale", "rivalutazione clinica", "accesso domiciliare di rivalutazione"]):
        return "controllo generale"

    return None


def extract_reason(text: str) -> Optional[str]:
    sents = _sentences(text)
    lead_sentences = sents[:3]
    lead = " ".join(lead_sentences) if lead_sentences else text

    explicit_patterns = [
        r"\bAccesso domiciliare per\s+(.*?)(?:$|\.)",
        r"\bAccesso per\s+(.*?)(?:$|\.)",
        r"\bVisita ADI per\s+(.*?)(?:$|\.)",
        r"\bVisita per\s+(.*?)(?:$|\.)",
        r"\bVisita richiesta per\s+(.*?)(?:$|\.)",
        r"\bLa visita di oggi è stata occasionale per\s+(.*?)(?:$|\.)",
        r"\bLa visita di oggi è stata per\s+(.*?)(?:$|\.)",
        r"\bSottoposto a visita per\s+(.*?)(?:$|\.)",
        r"\bAccesso domiciliare di controllo generale per\s+(.*?)(?:$|\.)",
        r"\bAccesso domiciliare di\s+(.*?)(?:$|\.)",
        r"\bAccesso per rivalutazione\s+(.*?)(?:$|\.)",
        r"\bAccesso per monitoraggio\s+(.*?)(?:$|\.)",
    ]

    for p in explicit_patterns:
        m = re.search(p, lead, flags=re.IGNORECASE)
        if m:
            raw_reason = _clean_spaces(m.group(1).strip(" .,:;"))
            normalized = _reason_from_keywords(raw_reason)
            if normalized:
                return normalized

    for sent in lead_sentences:
        normalized = _reason_from_keywords(sent)
        if normalized:
            return normalized

    normalized = _reason_from_keywords(text)
    if normalized:
        return normalized

    return None


def extract_reason_for_visit(text: str) -> str | None:
    return extract_reason(text)


def _extract_days(text: str) -> Optional[int]:
    patterns = [
        r"\btra\s+(\d+)\s+giorni\b",
        r"\bentro\s+(\d+)\s+giorni\b",
        r"\bfra\s+(\d+)\s+giorni\b",
        r"\bentro\s+tre\s+giorni\b",
        r"\bentro\s+due\s+giorni\b",
        r"\bentro\s+una\s+settimana\b",
        r"\bprossima\s+settimana\b",
        r"\bsettimana\s+prossima\b",
    ]
    for p in patterns:
        m = re.search(p, text, flags=re.IGNORECASE)
        if m:
            matched = m.group(0).lower()
            if matched == "entro tre giorni":
                return 3
            if matched == "entro due giorni":
                return 2
            if matched in {"entro una settimana", "prossima settimana", "settimana prossima"}:
                return 7
            if m.lastindex:
                return int(m.group(1))
    return None


def extract_follow_up(text: str) -> Any:
    relevant_lines = []
    for sent in _sentences(text):
        low = sent.lower()
        if any(k in low for k in [
            "follow",
            "programmato",
            "ricontatto",
            "nuovo controllo",
            "prossimo controllo",
            "controllo tra",
            "rivalutazione tra",
            "da rivalutare",
            "monitorare",
            "previsto",
            "entro tre giorni",
            "prossima settimana",
            "settimana prossima",
        ]):
            relevant_lines.append(sent)

    relevant_text = " ".join(relevant_lines) if relevant_lines else text
    low = relevant_text.lower()
    days = _extract_days(relevant_text)

    if "ricontatto" in low and "telefon" in low:
        target = "caregiver" if "caregiver" in low else None
        return {"type": "ricontatto_telefonico", "target": target, "timing_days": days}

    if _contains_any(low, [r"\bferita\b", r"\blesione\b", r"\bmedicazione\b", r"\bpiaga\b", r"\bulcera\b"]):
        if any(k in low for k in ["controllo", "rivalutazione", "programmato", "previsto", "follow"]):
            return {"type": "controllo_ferita", "timing_days": days}

    if any(k in low for k in [
        "controllo",
        "rivalutazione",
        "follow",
        "programmato",
        "previsto",
        "nuovo controllo",
        "prossima settimana",
        "settimana prossima",
        "entro tre giorni",
    ]):
        return {"type": "controllo", "timing_days": days}

    return None


def extract_interventions(text: str, vitals: Optional[dict] = None, reason: Optional[str] = None) -> list[str]:
    t = (text or "").lower()
    r = (reason or "").lower()
    interventions: list[str] = []

    has_any_vital = False
    if vitals:
        has_any_vital = any(
            vitals.get(k) is not None
            for k in [
                "blood_pressure_systolic",
                "blood_pressure_diastolic",
                "heart_rate",
                "temperature",
                "spo2",
            ]
        )

    if has_any_vital or any(k in t for k in [
        "parametri rilevati",
        "rilevati parametri",
        "monitoraggio parametri",
        "controllo parametri",
        "parametri vitali",
        "segni vitali",
        "controlli dei parametri vitali",
    ]):
        interventions.append("monitoraggio_parametri_vitali")

    if any(k in t for k in ["medicazione", "cambio medicazione", "medicata", "lesione detersa", "ferita detersa"]):
        interventions.append("medicazione")

    if any(k in t for k in ["farmaco", "somministrato", "somministrazione", "terapia eseguita", "terapia praticata"]):
        interventions.append("somministrazione_farmaco")

    if any(k in t for k in ["terapia", "aderenza terapeutica", "istruita", "educazione terapeutica", "assunzione farmaci"]):
        interventions.append("educazione_terapeutica")

    if any(k in t for k in ["catetere", "vescicale", "sacca urine", "lavaggio catetere"]):
        interventions.append("gestione_catetere")

    if any(k in t for k in ["stomia", "presidio stomale", "sacca stomia", "placca stomia"]):
        interventions.append("gestione_stomia")

    if any(k in t for k in ["ossigenoterapia", "o2 terapia", "ossigeno terapia"]):
        interventions.append("gestione_ossigenoterapia")

    if any(k in t for k in ["glicemia", "glucosio capillare"]):
        interventions.append("monitoraggio_glicemia")

    if any(k in t for k in ["educato caregiver", "istruito caregiver", "caregiver istruito", "fornite indicazioni", "forniti consigli"]):
        interventions.append("educazione_terapeutica")

    if any(k in t for k in ["valutazione generale", "rivalutazione", "obiettività", "esame obiettivo", "controllo generale"]) or not interventions:
        interventions.append("valutazione_generale")

    if "lesione" in r or "ferita" in r:
        interventions.append("medicazione")
    if "catetere" in r:
        interventions.append("gestione_catetere")
    if "stomia" in r:
        interventions.append("gestione_stomia")
    if "terapia" in r or "farmaco" in r:
        interventions.append("somministrazione_farmaco")
    if "parametri" in r or "segni vitali" in r:
        interventions.append("monitoraggio_parametri_vitali")
    if "ossigenoterapia" in r or "respiratorio" in r:
        interventions.append("gestione_ossigenoterapia")

    return list(dict.fromkeys(interventions))