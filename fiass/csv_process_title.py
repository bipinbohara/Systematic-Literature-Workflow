#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reads:  <script_dir>/csv-data/csv-MSCormesen-set.csv  (must have "Title")
Writes: <script_dir>/csv-data/csv-MSCormesen-set_results.csv
Adds one column: llm_output (raw text returned by the LLM)
"""

import csv
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

import requests

# ---------- Fixed I/O (script-relative) ----------
BASE_DIR   = Path(__file__).resolve().parent
DATA_DIR   = BASE_DIR / "csv-data"
INPUT_CSV  = DATA_DIR / "csv-MSCormesen-set.csv"
OUTPUT_CSV = DATA_DIR / "csv-MSCormesen-set_results.csv"
TITLE_COL  = "Title"
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

# ---------- LLM config ----------
# If your server only supports /v1/completions, keep the URL as that.
# If it supports chat, use /v1/chat/completions. This script will try both when needed.
LLM_URL     = os.environ.get("LLM_URL", "http://192.168.0.205:80/v1/completions")
LLM_MODEL   = os.environ.get("LLM_MODEL", "openai/gpt-oss-120b")
LLM_API_KEY = os.environ.get("LLM_API_KEY")  # optional
TIMEOUT     = int(os.environ.get("LLM_TIMEOUT", "600"))

SYSTEM_PROMPT = (
    "You are a precise classifier. From a PAPER TITLE alone, decide if the paper addresses what user_prompt questions"
)
USER_PROMPT_PREFIX = (
    "We are conducting a review to determine whether the research paper title in any way addresses any of the keywords: "
    "(MSC* or \"mesenchymal stem cell*\" or \"mesenchymal stromal cell*\" or ADSC or ASCs or \"adipose stem cell*\") "
    "and (aging or aged). Please Answer YES or NO.\nTITLE: "
)

YESNO_RE = re.compile(r'\b(YES|NO)\b', re.IGNORECASE)

def _extract_text(data: Dict[str, Any]) -> str:
    """Extract assistant text from multiple OpenAI-compatible shapes."""
    # Chat shape
    try:
        val = data["choices"][0]["message"]["content"]
        if val is not None:
            return str(val).strip()
    except Exception:
        pass
    # Completions shape
    try:
        val = data["choices"][0]["text"]
        if val is not None:
            return str(val).strip()
    except Exception:
        pass
    # Some servers put custom fields
    for k in ("output_text", "response"):
        if k in data and data[k]:
            return str(data[k]).strip()
    # Error in body?
    if "error" in data:
        err = data["error"]
        if isinstance(err, dict):
            return f"[LLM_ERROR_BODY] {err.get('message') or err}"
        return f"[LLM_ERROR_BODY] {err}"
    return ""

def _post_json(url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if LLM_API_KEY:
        headers["Authorization"] = f"Bearer {LLM_API_KEY}"
    r = requests.post(url, headers=headers, json=payload, timeout=TIMEOUT)
    if r.status_code != 200:
        # include server body for debugging
        raise RuntimeError(f"LLM HTTP {r.status_code}: {r.text[:500]}")
    return r.json()

def call_llm_chat(title: str) -> str:
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
    data = _post_json(LLM_URL, payload)
    return _extract_text(data)

def call_llm_completions(title: str) -> str:
    # derive a completions URL from current LLM_URL if needed
    if LLM_URL.rstrip("/").endswith("/v1/chat/completions"):
        comp_url = LLM_URL.rsplit("/chat", 1)[0]  # -> .../v1/completions
    elif LLM_URL.rstrip("/").endswith("/v1/completions"):
        comp_url = LLM_URL
    else:
        # default fallback
        comp_url = "http://192.168.0.205:80/v1/completions"

    prompt = f"System: {SYSTEM_PROMPT}\n\nUser:\n{USER_PROMPT_PREFIX}{json.dumps(title)}"
    payload = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "max_tokens": 32,
        "temperature": 0.0,
        "stream": False,
    }
    data = _post_json(comp_url, payload)
    return _extract_text(data)

def call_llm(title: str, debug: bool = False, idx: int = 0) -> str:
    """
    Try chat first. If empty, try completions.
    Print short debug for the first 3 rows.
    """
    text = ""
    try:
        text = call_llm_chat(title)
    except Exception as e:
        text = f"[LLM_ERROR_CHAT] {e!r}"

    if not text or text.strip() == "":
        try:
            alt = call_llm_completions(title)
            if alt:
                text = alt
        except Exception as e:
            # only replace if we had nothing
            if not text:
                text = f"[LLM_ERROR_COMP] {e!r}"

    if debug and idx < 3:
        print(f"[DEBUG] Row {idx} title: {title[:120]}")
        print(f"[DEBUG] Output: {text[:200]}\n")

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
        out_fields = list(headers)
        if "llm_output" not in out_fields:
            out_fields.append("llm_output")

        writer = csv.DictWriter(f_out, fieldnames=out_fields)
        writer.writeheader()

        for i, row in enumerate(reader):
            title = (row.get(TITLE_COL) or "").strip()
            if not title:
                row["llm_output"] = ""
            else:
                try:
                    row["llm_output"] = call_llm(title, debug=True, idx=i)
                except Exception as e:
                    row["llm_output"] = f"[LLM_ERROR] {e!r}"
            writer.writerow(row)

    print(f"Done. Wrote: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
