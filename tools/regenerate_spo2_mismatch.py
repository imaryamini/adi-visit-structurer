#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

# ---- CONFIG ----
MODEL = "llama3.1:8b"  # change if your Ollama model name differs
CSV_REPORT = Path("data/synthetic/validation_report.csv")
RAW_DIR = Path("data/synthetic/raw")
GOLD_DIR = Path("data/synthetic/gold")

TARGET_WARNING = "Raw mentions SpO2 but gold spo2 is null"
MAX_RETRIES = 3


def ollama_generate_strict(gold: dict) -> str:
    """
    Generate dictation text strictly from gold JSON.
    Critical constraint: If spo2 is null, NEVER mention SpO2/saturazione.
    """
    prompt = f"""
Sei un infermiere ADI. Genera una breve "dettatura" (stile appunti clinici) in italiano basata ESCLUSIVAMENTE sul record JSON qui sotto.

REGOLE OBBLIGATORIE:
- Non inventare dati non presenti nel JSON.
- Se un valore è null, NON menzionarlo.
- È vietato inventare SpO2: se clinical.vitals.spo2 è null, NON scrivere "SpO2", "saturazione", "Sat", "%" riferita a SpO2.
- Mantieni stile realistico (PA/FC/temperatura), breve e chiaro.
- Includi data e ora nel formato "gg/mm/aaaa ore hh:mm".
- Non aggiungere diagnosi non presenti.

JSON:
{json.dumps(gold, ensure_ascii=False)}
""".strip()

    res = subprocess.run(
        ["ollama", "run", MODEL],
        input=prompt.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return res.stdout.decode("utf-8").strip()


def contains_spo2(text: str) -> bool:
    t = text.lower()
    # catch common mentions
    return ("spo2" in t) or ("satur" in t) or ("sat" in t and "%" in t)


def load_gold(record_id: str) -> dict:
    p = GOLD_DIR / f"{record_id}.json"
    return json.loads(p.read_text(encoding="utf-8"))


def find_mismatch_ids() -> list[str]:
    if not CSV_REPORT.exists():
        raise FileNotFoundError(f"CSV report not found: {CSV_REPORT}")

    ids: list[str] = []
    with CSV_REPORT.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            hw = row.get("heuristic_warnings", "") or ""
            record_id = row.get("record_id", "") or ""
            if TARGET_WARNING in hw and record_id:
                ids.append(record_id)
    return ids


def main():
    ids = find_mismatch_ids()
    if not ids:
        print("No SpO2 mismatch records found in the CSV report. Nothing to regenerate.")
        return

    print(f"Found {len(ids)} records with SpO2 mismatch. Regenerating raw dictations...")

    fixed = 0
    failed = 0

    for rid in ids:
        gold = load_gold(rid)

        # Only makes sense for cases where spo2 is actually null in gold
        spo2_val = gold.get("clinical", {}).get("vitals", {}).get("spo2", None)
        if spo2_val is not None:
            print(f"- {rid}: gold spo2 is not null ({spo2_val}) -> skipping (CSV might be stale).")
            continue

        out_path = RAW_DIR / f"{rid}.txt"

        ok = False
        last_text = ""
        for attempt in range(1, MAX_RETRIES + 1):
            text = ollama_generate_strict(gold)
            last_text = text

            if not contains_spo2(text):
                out_path.write_text(text + "\n", encoding="utf-8")
                ok = True
                break
            else:
                print(f"  {rid}: attempt {attempt} still mentions SpO2 -> retrying...")

        if ok:
            print(f"- {rid}: regenerated OK")
            fixed += 1
        else:
            # Save the last attempt to help debugging (optional)
            out_path.write_text(last_text + "\n", encoding="utf-8")
            print(f"- {rid}: FAILED (LLM kept mentioning SpO2). Saved last attempt anyway.")
            failed += 1

    print(f"\nDone. Fixed: {fixed}, Failed: {failed}")
    print("Now re-run: python tools/validate_dataset.py")


if __name__ == "__main__":
    main()