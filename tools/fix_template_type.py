import json
import re
from pathlib import Path

RAW_DIR = Path("data/synthetic/raw")
GOLD_DIR = Path("data/synthetic/gold")

INTAKE_RE = re.compile(r"\b(presa in carico|prima visita|valutazione iniziale)\b", re.IGNORECASE)

def main():
    changed = 0
    for gold_path in GOLD_DIR.glob("ADI-*.json"):
        rid = gold_path.stem
        raw_path = RAW_DIR / f"{rid}.txt"
        if not raw_path.exists():
            continue

        raw = raw_path.read_text(encoding="utf-8", errors="ignore")
        data = json.loads(gold_path.read_text(encoding="utf-8"))

        tmpl = data.get("meta", {}).get("template_type", [])
        if not isinstance(tmpl, list):
            continue

        has_intake = bool(INTAKE_RE.search(raw))
        if ("presa_in_carico" in tmpl) and (not has_intake):
            data["meta"]["template_type"] = [t for t in tmpl if t != "presa_in_carico"]
            if not data["meta"]["template_type"]:
                data["meta"]["template_type"] = ["diario_clinico"]
            gold_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            changed += 1

    print(f"Removed presa_in_carico from {changed} files (no intake signal found).")

if __name__ == "__main__":
    main()