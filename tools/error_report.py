import json
from pathlib import Path

METRICS = Path("reports/metrics.json")
GOLD_DIR = Path("data/synthetic/gold")
PRED_DIR = Path("data/synthetic/pred")

def load(p): 
    return json.loads(p.read_text(encoding="utf-8"))

def main():
    m = load(METRICS)
    per = m.get("per_record", {})
    rows = []

    for rid, r in per.items():
        reason_ok = r["text_match"]["clinical.reason_for_visit"]
        fu_ok = r["text_match"]["clinical.follow_up"]
        vit_ok = r["vitals_exact_match"]
        int_f1 = r["f1_macro"]["clinical.interventions"]["f1"]
        prob_f1 = r["f1_macro"]["coding.problems_normalized"]["f1"]

        # Score: lower = worse
        score = (1 if reason_ok else 0) + (1 if fu_ok else 0) + (1 if vit_ok else 0) + int_f1 + prob_f1
        rows.append((score, rid, reason_ok, fu_ok, vit_ok, int_f1, prob_f1))

    rows.sort(key=lambda x: x[0])
    print("WORST 15 RECORDS:")
    for s, rid, reason_ok, fu_ok, vit_ok, int_f1, prob_f1 in rows[:15]:
        print(f"- {rid} score={s:.2f} reason={reason_ok} follow={fu_ok} vitals={vit_ok} int_f1={int_f1:.2f} prob_f1={prob_f1:.2f}")

    print("\nBEST 10 RECORDS:")
    for s, rid, reason_ok, fu_ok, vit_ok, int_f1, prob_f1 in rows[-10:][::-1]:
        print(f"- {rid} score={s:.2f} reason={reason_ok} follow={fu_ok} vitals={vit_ok} int_f1={int_f1:.2f} prob_f1={prob_f1:.2f}")

if __name__ == "__main__":
    main()