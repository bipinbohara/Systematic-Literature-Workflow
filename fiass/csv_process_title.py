#!/usr/bin/env python3
# Reads:  <script_dir>/csv-data/csv-MSCormesen-set.csv  (must have "Title")
# Writes: <script_dir>/csv-data/csv-MSCormesen-set_results.csv
# Adds one column: llm_output

import csv
import json
import os
from pathlib import Path
from typing import Any, Dict

import requests

# ---------- Fixed I/O (script-relative) ----------
BASE_DIR   = Path(__file__).resolve().parent
DATA_DIR   = BASE_DIR / "csv-data"
INPUT_CSV  = DATA_DIR / "csv-MSCormesen-set.csv"
OUTPUT_CSV = DATA_DIR / "csv-MSCormesen-set_results.csv"
TITLE_COL  = "Title"
OUT_COL    = "llm_output"
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

# ---------- LLM config ----------
# Prefer setting these via env if you change endpoints/models later.
LLM_URL     = os.environ.get("LLM_URL", "http://192.168.0.205:80/v1/completions")
LLM_MODEL   = os.environ.get("LLM_MODEL", "openai/gpt-oss-120b")
LLM_API_KEY = os.environ.get("LLM_API_KEY")  # optional
TIMEOUT     = int(os.environ.get("LLM_TIMEOUT", "600"))

SYSTEM_PROMPT = (
    "You are a precise classifier. From a PAPER TITLE alone, decide if the paper addresses what user_prompt questions"
)
USER_PROMPT_PREFIX = (
    "We are conducting a review to determine whether the research paper title in any way addresses any of the keywords: "
    '(MSC* or "mesenchymal stem cell*" or "mesenchymal stromal cell*" or ADSC or ASCs or "adipose stem cell*") '
    'and (aging or aged). Please Answer YES or NO.\nTITLE: '
)

# ---------- HTTP session (faster than per-call requests) ----------
SESSION = requests.Session()
HEADERS = {"Content-Type": "application/json"}
if LLM_API_KEY:
    HEADERS["Authorization"] = f"Bearer {LLM_API_KEY}"

def _extract_text(data: Dict[str, Any]) -> str:
    """Extract assistant text from OpenAI-compatible responses (chat or completions)."""
    # Chat shape
    try:
        msg = data["choices"][0]["message"]["content"]
        if isinstance(msg, str):
            return msg.strip()
        # Some servers return list blocks for content
        if isinstance(msg, list):
            return "".join(
                (b.get("text") or b.get("content") or b if isinstance(b, str) else "")
                for b in msg
            ).strip()
    except Exception:
        pass
    # Completions shape
    try:
        txt = data["choices"][0]["text"]
        if isinstance(txt, str):
            return txt.strip()
    except Exception:
        pass
    # Other custom shapes
    for k in ("output_text", "response"):
        if k in data and data[k]:
            return str(data[k]).strip()
    # Error message in body (helpful for debugging)
    if "error" in data:
        return f"[LLM_ERROR_BODY] {data['error']}"
    return ""

def _post_json(url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    r = SESSION.post(url, headers=HEADERS, json=payload, timeout=TIMEOUT)
    # include body for easier diagnosis
    if r.status_code != 200:
        raise RuntimeError(f"LLM HTTP {r.status_code}: {r.text[:500]}")
    return r.json()

def _call_completions(title: str) -> str:
    prompt = f"System: {SYSTEM_PROMPT}\n\nUser:\n{USER_PROMPT_PREFIX}{json.dumps(title)}"
    payload = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "max_tokens": 32,
        "temperature": 0.0,
        "stream": False,
    }
    data = _post_json(LLM_URL, payload)
    return _extract_text(data)

def _call_chat(title: str, url: str) -> str:
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
    data = _post_json(url, payload)
    return _extract_text(data)

def call_llm(title: str) -> str:
    """
    Call the configured endpoint. If LLM_URL is a completions endpoint and returns 404,
    automatically retry once against the chat endpoint at /v1/chat/completions.
    """
    url = LLM_URL.rstrip("/")
    is_chat = url.endswith("/v1/chat/completions")
    try:
        if is_chat:
            return _call_chat(title, url)
        else:
            return _call_completions(title)
    except RuntimeError as e:
        # If completions 404s, try chat once
        msg = str(e)
        if (not is_chat) and "LLM HTTP 404" in msg:
            chat_url = url.rsplit("/v1/completions", 1)[0] + "/v1/chat/completions"
            try:
                return _call_chat(title, chat_url)
            except Exception:
                return f"[LLM_ERROR_CHAT] {e}"
        return f"[LLM_ERROR] {e}"
    except Exception as e:
        return f"[LLM_ERROR] {e}"

def main() -> None:
    if not INPUT_CSV.exists():
        raise SystemExit(f"Input CSV not found at {INPUT_CSV}")

    with INPUT_CSV.open("r", newline="", encoding="utf-8-sig") as f_in, \
         OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f_out:

        reader = csv.DictReader(f_in)
        headers = reader.fieldnames or []
        if TITLE_COL not in headers:
            raise SystemExit(f'Expected title column "{TITLE_COL}" not found. Headers: {headers}')

        out_fields = list(headers)
        if OUT_COL not in out_fields:
            out_fields.append(OUT_COL)
        writer = csv.DictWriter(f_out, fieldnames=out_fields)
        writer.writeheader()

        count = 0
        for row in reader:
            title = (row.get(TITLE_COL) or "").strip()
            row[OUT_COL] = call_llm(title) if title else ""
            writer.writerow(row)
            count += 1

    print(f"Processed {count} rows.")
    print(f"Done. Wrote: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
