#!/usr/bin/env python3
"""
Classify paper titles as 'user-prompted' (YES/NO) with no CLI args.

- Input CSV:  /mnt/data/csv-MSCormesen-set.csv   (must contain column "Title")
- Output CSV: /mnt/data/csv-MSCormesen-set_userprompt.csv
- Adds only ONE new column: "user_prompt" (YES/NO)
- LLM endpoint/model and the classification prompt are embedded below.
"""

import csv
import json
import os
import re
from pathlib import Path
from typing import Optional

import requests

# ------------ Fixed I/O (script-relative csv-data/) ------------
BASE_DIR  = Path(__file__).resolve().parent
DATA_DIR  = BASE_DIR / "csv-data"
INPUT_CSV = DATA_DIR / "csv-MSCormesen-set.csv"
OUTPUT_CSV = DATA_DIR / "csv-MSCormesen-set_results.csv"
TITLE_COL  = "Title"

# Ensure output directory exists
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

# ------------ LLM config (edit if needed) ------------
LLM_URL     = "http://192.168.0.205:80/v1/chat/completions"   # your local endpoint
LLM_MODEL   = "openai/gpt-oss-120b"
LLM_API_KEY = os.environ.get("LLM_API_KEY")  # optional; leave unset if not needed
TIMEOUT     = 600

# ------------ Embedded instructions ("user_prompt" logic) ------------
SYSTEM_PROMPT = (
    "You are a precise classifier. From a PAPER TITLE alone, decide if the study is about a " +
    "'user-prompted' system — i.e., a user's natural-language prompt directly drives the system " +
    "(LLMs, instruction-following, prompt engineering, chatbots, text-to-*, etc.). " +
    "If the title is clearly unrelated (e.g., biology/medicine without LLMs, networking, materials), answer NO. " +
    "Return ONLY one token: YES or NO."
)

# This is the user message prefix given to the model; the title is appended.
USER_PROMPT_PREFIX = ("We are conducting a review to determine whether the research paper title in any way addresses (MSC* or “mesenchymal stem cell*” or “mesenchymal stromal cell*“ or ADSC or ASCs or “adipose stem cell*”) and (aging or aged). Please Answer YES or NO.\nTITLE: "
)

YESNO_RE = re.compile(r'\b(YES|NO)\b', re.IGNORECASE)

def call_llm(title: str) -> str:
    """Send title to the LLM and expect a simple YES/NO response."""
    headers = {"Content-Type": "application/json"}
    if LLM_API_KEY:
        headers["Authorization"] = f"Bearer {LLM_API_KEY}"

    user_msg = f"{USER_PROMPT_PREFIX}{json.dumps(title)}"

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "max_tokens": 16,
        "temperature": 0.0,
        "stream": False,
    }

    r = requests.post(LLM_URL, headers=headers, json=payload, timeout=TIMEOUT)
    if r.status_code != 200:
        raise RuntimeError(f"LLM error {r.status_code}: {r.text}")
    data = r.json()
    try:
        text = data["choices"][0]["message"]["content"].strip()
    except Exception:
        text = ""

    m = YESNO_RE.search(text)
    if m:
        return m.group(1).upper()
    if text.strip().upper().startswith("Y"):
        return "YES"
    if text.strip().upper().startswith("N"):
        return "NO"
    return "NO"  # conservative default

def main():
    in_path = Path(INPUT_CSV)
    out_path = Path(OUTPUT_CSV)

    if not in_path.exists():
        raise SystemExit(f"Input CSV not found at {in_path}")

    with in_path.open("r", newline="", encoding="utf-8-sig") as f_in, \
         out_path.open("w", newline="", encoding="utf-8") as f_out:
        reader = csv.DictReader(f_in)
        headers = reader.fieldnames or []
        if TITLE_COL not in headers:
            raise SystemExit(f'Expected title column "{TITLE_COL}" not found. Headers: {headers}')

        out_fields = headers + (["user_prompt"] if "user_prompt" not in headers else [])
        writer = csv.DictWriter(f_out, fieldnames=out_fields)
        writer.writeheader()

        for row in reader:
            title = (row.get(TITLE_COL) or "").strip()
            if not title:
                row["user_prompt"] = "NO"
            else:
                try:
                    row["user_prompt"] = call_llm(title)
                except Exception:
                    row["user_prompt"] = "NO"
            writer.writerow(row)

    print(f"Done. Wrote: {out_path}")

if __name__ == "__main__":
    main()
