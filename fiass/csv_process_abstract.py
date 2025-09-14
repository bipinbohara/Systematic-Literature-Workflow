#!/usr/bin/env python3
# Reads:  <script_dir>/csv-data/csv-MSCormesen-set.csv  (must have "Title"; "Abstract" optional)
# Writes: <script_dir>/csv-data/csv-MSCormesen-set_results.csv
# Adds one column: llm_output (raw model reply)

import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import requests

# ---------- Fixed I/O (script-relative) ----------
BASE_DIR   = Path(__file__).resolve().parent
DATA_DIR   = BASE_DIR / "csv-data"
INPUT_CSV  = DATA_DIR / "scopus_file.csv"
OUTPUT_CSV = DATA_DIR / "scopus_file_results.csv"
TITLE_COL  = "Title"
ABSTRACT_COL = "Abstract"
# We’ll auto-detect abstract column from these candidates:
ABSTRACT_CANDIDATES: List[str] = [
    "Abstract", "abstract", "ABSTRACT", "AbstractText", "Abstract_Text", "Abstract Text"
]
OUT_COL    = "llm_output"
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

# ---------- LLM config ----------
LLM_URL       = os.environ.get("LLM_URL",   "http://192.168.0.205:80/v1/completions")  # prefer completions; will fallback to chat if 404
LLM_MODEL     = os.environ.get("LLM_MODEL", "openai/gpt-oss-120b")
LLM_API_KEY   = os.environ.get("LLM_API_KEY")  # optional
TIMEOUT       = int(os.environ.get("LLM_TIMEOUT", "600"))
DEBUG_FIRST_N = int(os.environ.get("DEBUG_FIRST_N", "0"))  # set to e.g. 2 for quick prints

SYSTEM_PROMPT = (
    "You are a precise classifier. Using only the TITLE and ABSTRACT of a paper, determine whether the paper "
    "addresses the following query (YES or NO): "
    '(MSC* or "mesenchymal stem cell*" or "mesenchymal stromal cell*" or ADSC or ASCs or "adipose stem cell*") '
    'AND (aging OR aged). Respond with a concise YES or NO (optionally followed by a short rationale).'
)

USER_PROMPT_PREFIX = "TITLE: {title}\nABSTRACT: {abstract}"

# ---------- HTTP session ----------
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
    if "error" in data:
        return f"[LLM_ERROR_BODY] {data['error']}"
    return ""

def _post_json(url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    r = SESSION.post(url, headers=HEADERS, json=payload, timeout=TIMEOUT)
    if r.status_code != 200:
        raise RuntimeError(f"LLM HTTP {r.status_code}: {r.text[:500]}")
    return r.json()

def _build_user_content(title: str, abstract: str) -> str:
    # Safely JSON-escape Title/Abstract once; also truncate oversized inputs
    t = (title or "").strip()
    a = (abstract or "").strip()
    # if MAX_INPUT_CH > 0 and len(a) > MAX_INPUT_CH:
    #     a = a[:MAX_INPUT_CH] + " …[truncated]"
    return USER_PROMPT_PREFIX.format(title=json.dumps(t), abstract=json.dumps(a))

def _call_completions(user_content: str) -> str:
    prompt = f"System: {SYSTEM_PROMPT}\n\nUser:\n{user_content}"
    payload = {
        "model": LLM_MODEL,
        "prompt": prompt,
        #"max_tokens": MAX_TOKENS,
        "temperature": 0.0,
        "stream": False,
    }
    data = _post_json(LLM_URL, payload)
    return _extract_text(data)

def _call_chat(user_content: str, url: str) -> str:
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        #"max_tokens": MAX_TOKENS,
        "temperature": 0.0,
        "stream": False,
    }
    data = _post_json(url, payload)
    return _extract_text(data)

def call_llm(title: str, abstract: str, row_idx: int) -> str:
    """Call configured endpoint; fallback to chat endpoint if completions 404s."""
    user_content = _build_user_content(title, abstract)
    url = LLM_URL.rstrip("/")
    is_chat = url.endswith("/v1/chat/completions")

    try:
        text = _call_chat(user_content, url) if is_chat else _call_completions(user_content)
    except RuntimeError as e:
        # If completions 404s, retry at chat once
        msg = str(e)
        if (not is_chat) and "LLM HTTP 404" in msg:
            chat_url = url.rsplit("/v1/completions", 1)[0] + "/v1/chat/completions"
            try:
                text = _call_chat(user_content, chat_url)
            except Exception:
                text = f"[LLM_ERROR_CHAT] {e}"
        else:
            text = f"[LLM_ERROR] {e}"
    except Exception as e:
        text = f"[LLM_ERROR] {e}"

    if DEBUG_FIRST_N and row_idx < DEBUG_FIRST_N:
        print(f"[DEBUG] TITLE: {title[:200]}")
        if abstract:
            print(f"[DEBUG] ABSTRACT: {abstract[:200]}")
        print(f"[DEBUG] REPLY: {text[:200]}\n")

    return text

def main() -> None:
    if not INPUT_CSV.exists():
        raise SystemExit(f"Input CSV not found at {INPUT_CSV}")

    with INPUT_CSV.open("r", newline="", encoding="utf-8-sig") as f_in, \
         OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f_out:

        reader = csv.DictReader(f_in)
        headers = reader.fieldnames or []
        if TITLE_COL not in headers:
            raise SystemExit(f'Expected title column "{TITLE_COL}" not found. Headers: {headers}')
        # abstract_col = _detect_abstract_col(headers)

        out_fields = list(headers)
        if OUT_COL not in out_fields:
            out_fields.append(OUT_COL)
        writer = csv.DictWriter(f_out, fieldnames=out_fields)
        writer.writeheader()

        count = 0
        for i, row in enumerate(reader):
            title = (row.get(TITLE_COL) or "").strip()
            abstract = (row.get(ABSTRACT_COL) or "").strip()
            row[OUT_COL] = call_llm(title, abstract, i) if title else ""
            writer.writerow(row)
            count += 1

    print(f"Processed {count} rows.")
    print(f"Done. Wrote: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
