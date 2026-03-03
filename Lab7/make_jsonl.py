import csv
import json
from pathlib import Path

CSV_FILE = Path(__file__).parent / "preferences_f46514d8.csv"
OUT_FILE = Path(__file__).parent / "dpo_pairs.jsonl"

rows = []
with open(CSV_FILE, encoding="utf-8") as f:
    for r in csv.DictReader(f):
        if r["is_tie"].strip().lower() not in ("true", "1", "yes"):
            rows.append({
                "prompt": r["prompt"].strip(),
                "chosen": r["chosen"].strip(),
                "rejected": r["rejected"].strip(),
            })

with open(OUT_FILE, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")

print(f"Wrote {len(rows)} pairs to {OUT_FILE}")
