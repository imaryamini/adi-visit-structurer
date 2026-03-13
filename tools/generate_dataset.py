import json
import os
import random
import subprocess
from datetime import datetime, timedelta

# -------------------------
# Config
# -------------------------
MODEL = "llama3.1:8b"  # change if you use a different model in Ollama

OUT_RAW = "data/synthetic/raw"
OUT_GOLD = "data/synthetic/gold"

random.seed(7)

PROBLEM_POOL = [
    "astenia", "inappetenza", "dolore", "dispnea", "edema",
    "capogiro", "insonnia", "nausea", "febbre", "caduta"
]

INTERVENTION_POOL = [
    "monitoraggio_parametri_vitali",
    "valutazione_generale",
    "educazione_alimentare",
    "medicazione",
    "somministrazione_farmaco",
    "gestione_catetere",
    "gestione_stomia",
    "monitoraggio_glicemia",
    "educazione_terapeutica"
]

def ensure_dirs():
    os.makedirs(OUT_RAW, exist_ok=True)
    os.makedirs(OUT_GOLD, exist_ok=True)

def iso_dt(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S")

def build_gold(record_id: str, dt: datetime, scenario: str) -> dict:
    # Minimal schema (expand to match YOUR schema if needed)
    patient_age = random.choice([None, random.randint(65, 90)])
    patient_sex = random.choice([None, "F", "M"])

    vitals_present = scenario not in {"symptoms_no_vitals"}  # scenario control
    spo2_present = vitals_present and random.random() < 0.6

    vitals = {
        "blood_pressure_systolic": random.choice([110, 120, 130, 140, 150]) if vitals_present else None,
        "blood_pressure_diastolic": random.choice([70, 80, 85, 90]) if vitals_present else None,
        "heart_rate": random.choice([60, 65, 72, 78, 90, 98]) if vitals_present else None,
        "temperature": random.choice([36.2, 36.5, 36.7, 37.1, 37.8]) if vitals_present else None,
        "spo2": random.choice([96, 97, 98, 99]) if spo2_present else None
    }

    problems = []
    interventions = []
    follow_up = None
    warnings = []

    # Scenario logic
    if scenario == "routine":
        interventions = ["monitoraggio_parametri_vitali"]
        problems = []
        follow_up = {"type": "controllo", "timing_days": random.choice([3, 7, 14])}
    elif scenario == "symptoms_no_vitals":
        problems = random.sample(["astenia", "inappetenza", "capogiro", "insonnia", "nausea"], k=2)
        interventions = ["valutazione_generale", random.choice(["educazione_alimentare", "educazione_terapeutica"])]
        follow_up = {"type": "ricontatto_telefonico", "target": "caregiver", "timing_days": None}
        warnings += ["Nessun parametro vitale riportato", "Follow-up senza tempistica"]
    elif scenario == "fall":
        problems = ["caduta"]
        interventions = ["monitoraggio_parametri_vitali", "valutazione_generale"]
        follow_up = {"type": "controllo", "timing_days": None}
        warnings += ["Follow-up senza tempistica"]
    elif scenario == "wound":
        problems = ["dolore"] if random.random() < 0.4 else []
        interventions = ["medicazione", "valutazione_generale"]
        follow_up = {"type": "controllo_ferita", "timing_days": random.choice([2, 3, 7])}
    elif scenario == "medication":
        problems = random.sample(["dolore", "febbre", "dispnea"], k=1)
        interventions = ["somministrazione_farmaco", "educazione_terapeutica"]
        follow_up = {"type": "controllo", "timing_days": random.choice([2, 7])}
        # optionally trigger a warning sometimes
        if random.random() < 0.3:
            warnings += ["Farmaco menzionato senza dose"]
    else:
        # fallback
        interventions = ["valutazione_generale"]
        follow_up = {"type": "controllo", "timing_days": 7}

    # Additional warnings
    if vitals_present and vitals["spo2"] is None:
        warnings += ["SpO2 non menzionata"]

    gold = {
        "meta": {
            "record_id": record_id,
            "template_type": ["diario_clinico"],
            "visit_datetime": iso_dt(dt),
            "operator_role": "infermiere"
        },
        "patient": {
            "patient_id": f"SYNTH-{record_id}",
            "age": patient_age,
            "sex": patient_sex
        },
        "clinical": {
            "reason_for_visit": scenario_to_reason(scenario),
            "anamnesis_brief": [],
            "vitals": vitals,
            "interventions": interventions,
            "critical_issues": ["caduta_recente"] if scenario == "fall" else [],
            "follow_up": follow_up
        },
        "coding": {
            "problems_normalized": problems,
            "risk_flags": ["rischio_malnutrizione"] if ("inappetenza" in problems) else []
        },
        "quality": {
            "missing_mandatory_fields": [],
            "warnings": sorted(list(set(warnings)))
        }
    }
    return gold

def scenario_to_reason(scenario: str) -> str:
    return {
        "routine": "controllo parametri",
        "symptoms_no_vitals": "riferiti sintomi generali",
        "fall": "rivalutazione caduta recente",
        "wound": "medicazione/controllo lesione",
        "medication": "controllo terapia/somministrazione farmaco"
    }.get(scenario, "controllo generale")

def ollama_generate_dictation(gold: dict) -> str:
    # Prompt: ask for Italian clinical dictation, realistic but SHORT, no extra facts beyond gold.
    prompt = f"""
Sei un infermiere ADI. Genera una breve "dettatura" (stile appunti clinici) in italiano basata ESCLUSIVAMENTE sul record JSON qui sotto.
Regole:
- Non inventare dati non presenti.
- Se un valore è null, non menzionarlo.
- Mantieni stile realistico (PA/FC/SpO2, virgole, punti).
- Includi data e ora nel formato gg/mm/aaaa ore hh:mm.
- Se follow_up.timing_days è null, scrivi un follow-up generico senza giorni.

JSON:
{json.dumps(gold, ensure_ascii=False)}
""".strip()

    result = subprocess.run(
        ["ollama", "run", MODEL],
        input=prompt.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return result.stdout.decode("utf-8").strip()

def save_pair(record_id: str, dictation: str, gold: dict):
    raw_path = os.path.join(OUT_RAW, f"{record_id}.txt")
    gold_path = os.path.join(OUT_GOLD, f"{record_id}.json")
    with open(raw_path, "w", encoding="utf-8") as f:
        f.write(dictation + "\n")
    with open(gold_path, "w", encoding="utf-8") as f:
        json.dump(gold, f, ensure_ascii=False, indent=2)

def main(n_new: int = 60, start_id: int = 14):
    ensure_dirs()

    scenarios = (
        ["routine"] * 15
        + ["symptoms_no_vitals"] * 10
        + ["fall"] * 8
        + ["wound"] * 15
        + ["medication"] * 12
    )
    random.shuffle(scenarios)

    # Create a date window around Feb–Mar 2026
    base = datetime(2026, 2, 10, 8, 0)

    for i in range(n_new):
        rec_num = start_id + i
        record_id = f"ADI-{rec_num:04d}"
        dt = base + timedelta(days=random.randint(0, 35), hours=random.randint(0, 10), minutes=random.choice([0, 10, 20, 30, 40, 50]))
        scenario = scenarios[i % len(scenarios)]
        gold = build_gold(record_id, dt, scenario)
        dictation = ollama_generate_dictation(gold)
        save_pair(record_id, dictation, gold)
        print(f"Generated {record_id} ({scenario})")

if __name__ == "__main__":
    main()