#!/usr/bin/env python3
# Reads:  <script_dir>/csv-data/csv-MSCormesen-set.csv  (must have "Title")
# Writes: <script_dir>/csv-data/csv-MSCormesen-set_results.csv
# Adds one column: llm_output

import csv
import json
import os
import re
from pathlib import Path
from typing import Optional, Any, Dict

import requests

# ---------- Fixed I/O (script-relative) ----------
BASE_DIR   = Path(__file__).resolve().parent
DATA_DIR   = BASE_DIR / "csv-data"
INPUT_CSV  = DATA_DIR / "csv-MSCormesen-set.csv"
OUTPUT_CSV = DATA_DIR / "csv-MSCormesen-set_results.csv"
TITLE_COL  = "Title"
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

# ---------- LLM config ----------
LLM_URL     = "http://192.168.0.205:80/v1/completions"   # change if your server prefers /v1/completions
LLM_MODEL   = "openai/gpt-oss-120b"
LLM_API_KEY = os.environ.get("LLM_API_KEY")  # optional
TIMEOUT     = 600

SYSTEM_PROMPT = (
    "You are a precise classifier. From a PAPER TITLE alone, decide if the paper addresses what user_prompt questions"
)
USER_PROMPT_PREFIX = (
    "We are conducting a review to determine whether the research paper title in any way addresses any of the keywords: "
    "(MSC* or “mesenchymal stem cell*” or “mesenchymal stromal cell*“ or ADSC or ASCs or “adipose stem cell*”) and (aging or aged). "
    "Please Answer YES or NO.\nTITLE: "
)

YESNO_RE = re.compile(r'\b(YES|NO)\b', re.IGNORECASE)

def _extract_text(data: Dict[str, Any]) -> str:
    """
    Be robust to different OpenAI-compatible response shapes.
    Try chat -> completions -> other known fields.
    """
    # Chat completions shape
    try:
        val = data["choices"][0]["message"]["content"]
        if val is not None:
            return str(val).strip()
    except Exception:
        pass
    # Text completions shape
    try:
        val = data["choices"][0]["text"]
        if val is not None:
            return str(val).strip()
    except Exception:
        pass
    # Some servers include these:
    for k in ("output_text", "response"):
        if k in data and data[k]:
            return str(data[k]).strip()
    return ""

def call_llm(title: str) -> str:
    headers = {"Content-Type": "application/json"}
    if LLM_API_KEY:
        headers["Authorization"] = f"Bearer {LLM_API_KEY}"

    # If your server actually expects /v1/completions, switch payload accordingly:
    is_completions = LLM_URL.rstrip("/").endswith("/v1/completions")

    if is_completions:
        payload = {
            "model": LLM_MODEL,
            "prompt": f"System: {SYSTEM_PROMPT}\n\nUser:\n{USER_PROMPT_PREFIX}{json.dumps(title)}",
            "max_tokens": 32,
            "temperature": 0.0,
            "stream": False,
        }
    else:
        payload = {
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"{USER_PROMPT_PREFIX}{json.dumps(title)}"},
            ],
            "max_tokens": 32,
            "temperature": 0.0,
            "stream": False,
        }

    r = requests.post(LLM_URL, headers=headers, json=payload, timeout=TIMEOUT)
    if r.status_code != 200:
        raise RuntimeError(f"LLM error {r.status_code}: {r.text}")

    data = r.json()
    text = _extract_text(data)

    # If your model returns long prose, you can keep it;
    # or collapse to YES/NO by uncommenting below.
    # m = YESNO_RE.search(text)
    # if m:
    #     return m.group(1).upper()
    return text

def main():
    if not INPUT_CSV.exists():
        raise SystemExit(f"Input CSV not found at {INPUT_CSV}")

    with INPUT_CSV.open("r", newline="", encoding="utf-8-sig") as f_in, \
         OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f_out:
        reader = csv.DictReader(f_in)
        headers = reader.fieldnames or []
        if TITLE_COL not in headers:
            raise SystemExit(f'Expected title column "{TITLE_COL}" not found. Headers: {headers}')

        # Always include llm_output in fieldnames
        out_fields = headers + ([] if "llm_output" in headers else ["llm_output"])
        writer = csv.DictWriter(f_out, fieldnames=out_fields)
        writer.writeheader()

        for row in reader:
            title = (row.get(TITLE_COL) or "").strip()
            try:
                row["llm_output"] = call_llm(title) if title else ""
            except Exception as e:
                row["llm_output"] = f"[LLM_ERROR] {e!r}"
            writer.writerow(row)

    print(f"Done. Wrote: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
