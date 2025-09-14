#!/usr/bin/env python3
"""
No-args CSV classifier:
- Reads INPUT_CSV (expects a column named "Title")
- Calls your LLM to classify whether each title is about a user-prompted system
- Writes all original columns + ["user_prompt", "user_prompt_reason"] to OUTPUT_CSV
- Can resume if OUTPUT_CSV already exists (by matching on a key built from Title+DOI when DOI exists, else Title)
"""

import csv
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import requests

# ============== Configuration (edit these if needed) ==============
INPUT_CSV   = "/mnt/data/csv-MSCormesen-set.csv"
OUTPUT_CSV  = None  # if None, will use "<input>_userprompt.csv"
TITLE_COL   = "Title"     # fixed per your request
RESUME      = True        # if True, reuse results already present in OUTPUT_CSV
DELAY_SECS  = 0.0         # sleep between LLM calls to be gentle on the server

# Environment-based LLM settings (same as your setup)
LLM_URL     = os.environ.get("LLM_URL", "http://192.168.0.205:80/v1/chat/completions")
LLM_MODEL   = os.environ.get("LLM_MODEL", "openai/gpt-oss-120b")
LLM_API_KEY = os.environ.get("LLM_API_KEY")  # optional
TIMEOUT     = int(os.environ.get("LLM_TIMEOUT", "600"))

# System prompt for "user-prompted" classification
SYSTEM_PROMPT = (
    "You are a careful research assistant.\n"
    "Task: From the PAPER TITLE alone, decide if the paper is about a 'user-prompted system': "
    "e.g., work where a user's natural-language prompt directly drives the system "
    "(LLMs, instruction-following, prompt engineering, chatbots, text-to-X, etc.).\n"
    "If the title is clearly unrelated (e.g., biology, medicine, networking), answer NO.\n"
    "Return STRICT JSON only:\n"
    '{"is_user_prompt":"YES|NO","reason":"short explanation"}\n'
)

# ============== LLM helpers ==============

def call_llm(url: str, model: str, system_prompt: str, user_prompt: str,
             api_key: Optional[str] = None) -> str:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    if url.rstrip("/").endswith("/v1/completions"):
        payload = {
            "model": model,
            "prompt": f"System: {system_prompt}\n\nUser:\n{user_prompt}",
            "max_tokens": 512,
            "temperature": 0.0,
            "stream": False,
        }
    else:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": 512,
            "temperature": 0.0,
            "stream": False,
        }

    r = requests.post(url, headers=headers, json=payload, timeout=TIMEOUT)
    if r.status_code != 200:
        raise RuntimeError(f"LLM error {r.status_code}: {r.text}")
    data = r.json()
    if url.rstrip("/").endswith("/v1/completions"):
        return data["choices"][0]["text"]
    return data["choices"][0]["message"]["content"]

JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)

def parse_llm_json(raw: str) -> Tuple[str, str]:
    """
    Extract {"is_user_prompt": "YES|NO", "reason": "..."} from model output.
    Falls back to heuristic if needed.
    """
    txt = (raw or "").strip()
    m = JSON_RE.search(txt)
    if m:
        snippet = m.group(0)
        try:
            obj = json.loads(snippet)
            val = str(obj.get("is_user_prompt", "")).strip().upper()
            reason = str(obj.get("reason", "")).strip()
            if val in {"YES", "NO"}:
                return val, reason
        except Exception:
            pass

    # heuristic fallback
    up = "YES" if "YES" in txt.upper() and "NO" not in txt.upper() else "NO"
    reason = txt[:280].replace("\n", " ")
    return up, reason

# ============== CSV helpers ==============

def make_key(row: Dict[str, str], title_col: str) -> str:
    """
    Prefer Title + DOI (if DOI exists) to avoid collisions; else Title only.
    """
    title = (row.get(title_col) or "").strip()
    doi = (row.get("DOI") or row.get("doi") or "").strip()
    if doi:
        return f"{title} || {doi}"
    return title

def load_existing_results(path: Path, title_col: str) -> Dict[str, Tuple[str, str]]:
    done: Dict[str, Tuple[str, str]] = {}
    if not path.exists():
        return done
    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            # Ensure we have our expected columns
            if not r.fieldnames or "user_prompt" not in r.fieldnames:
                return done
            for row in r:
                key = make_key(row, title_col)
                up  = (row.get("user_prompt") or "").strip().upper()
                if up in {"YES", "NO"}:
                    done[key] = (up, row.get("user_prompt_reason") or "")
    except Exception:
        # If anything goes wrong, just skip resume
        return {}
    return done

# ============== Main ==============

def main():
    in_path = Path(INPUT_CSV)
    if not in_path.exists():
        raise SystemExit(f"Input not found: {in_path}")

    out_path = Path(OUTPUT_CSV) if OUTPUT_CSV else in_path.with_name(in_path.stem + "_userprompt.csv")

    # Prepare resume cache
    existing: Dict[str, Tuple[str, str]] = {}
    if RESUME:
        existing = load_existing_results(out_path, TITLE_COL)

    with in_path.open("r", newline="", encoding="utf-8-sig") as f_in, \
         out_path.open("w", newline="", encoding="utf-8") as f_out:
        reader = csv.DictReader(f_in)
        headers = reader.fieldnames or []

        if TITLE_COL not in headers:
            raise SystemExit(f'Expected title column "{TITLE_COL}" not found. Headers: {headers}')

        out_fields = headers + [c for c in ["user_prompt", "user_prompt_reason"] if c not in headers]
        writer = csv.DictWriter(f_out, fieldnames=out_fields)
        writer.writeheader()

        processed = 0
        for row in reader:
            title = (row.get(TITLE_COL) or "").strip()
            key   = make_key(row, TITLE_COL)

            if not title:
                row["user_prompt"] = "NO"
                row["user_prompt_reason"] = "No title provided."
                writer.writerow(row)
                continue

            # Resume: reuse existing classification if present
            if RESUME and key in existing:
                up, reason = existing[key]
                row["user_prompt"] = up
                row["user_prompt_reason"] = reason
                writer.writerow(row)
                processed += 1
                continue

            try:
                raw = call_llm(
                    url=LLM_URL,
                    model=LLM_MODEL,
                    system_prompt=SYSTEM_PROMPT,
                    user_content=f'TITLE: "{title}"',
                    api_key=LLM_API_KEY,
                )
                up, reason = parse_llm_json(raw)
            except Exception as e:
                up, reason = "NO", f"LLM_ERROR: {e!r}"

            row["user_prompt"] = up
            row["user_prompt_reason"] = reason
            writer.writerow(row)
            processed += 1

            if DELAY_SECS > 0:
                time.sleep(DELAY_SECS)

    print(f"Done. Wrote: {out_path}")

if __name__ == "__main__":
    main()
