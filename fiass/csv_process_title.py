#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import requests

# ---------------- Env / Defaults ----------------
LLM_URL   = os.environ.get("LLM_URL", "http://192.168.0.205:80/v1/chat/completions")
LLM_MODEL = os.environ.get("LLM_MODEL", "openai/gpt-oss-120b")
LLM_API_KEY = os.environ.get("LLM_API_KEY")  # optional
TIMEOUT = int(os.environ.get("LLM_TIMEOUT", "600"))

# You can override via --system-prompt-file if you want
DEFAULT_SYSTEM_PROMPT = (
    "You are a careful research assistant.\n"
    "Task: From the PAPER TITLE alone, decide if the paper is about a 'user-prompted system': "
    "e.g., work where a user's natural-language prompt directly drives the system "
    "(LLMs, instruction-following, prompt engineering, chatbots, text-to-X, etc.).\n"
    "If the title is clearly unrelated (e.g., biology, medicine, networking), answer NO.\n"
    "Return STRICT JSON only:\n"
    '{"is_user_prompt":"YES|NO","reason":"short explanation"}\n'
)

# Common title header guesses (you can pass --title-col to force)
TITLE_GUESSES = [
    "Title", "title", "Paper Title", "paper_title", "TI", "Document Title",
    "dc:title", "ArticleTitle", "article_title"
]

# --------------- LLM -----------------
def call_llm(url: str, model: str, system_prompt: str, user_content: str,
             api_key: Optional[str] = None) -> str:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    if url.rstrip("/").endswith("/v1/completions"):
        payload = {
            "model": model,
            "prompt": f"System: {system_prompt}\n\nUser:\n{user_content}",
            "max_tokens": 512,
            "temperature": 0.0,
            "stream": False,
        }
    else:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
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
    Try to extract {"is_user_prompt": "YES|NO", "reason": "..."} from model output.
    Falls back to heuristic if needed.
    """
    txt = raw.strip()
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
    # keep a short reason
    reason = txt[:280].replace("\n", " ")
    return up, reason

# --------------- CSV Helpers -----------------
def detect_title_col(headers, forced: Optional[str]) -> str:
    if forced:
        if forced in headers:
            return forced
        else:
            raise SystemExit(f"--title-col '{forced}' not found in CSV headers: {list(headers)}")
    for cand in TITLE_GUESSES:
        if cand in headers:
            return cand
    raise SystemExit(
        f"Could not auto-detect title column. Headers were: {list(headers)}.\n"
        f"Pass --title-col <NAME>."
    )

def load_existing_results(path: Path, key_col: str) -> Dict[str, Tuple[str, str]]:
    """
    For resume: read already-written rows and remember user_prompt & reason by a key.
    Key default is the title column to keep it simple (collisions are rare in practice for your use).
    """
    done: Dict[str, Tuple[str, str]] = {}
    if not path.exists():
        return done
    with path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if key_col not in r.fieldnames:
            return done
        for row in r:
            up = (row.get("user_prompt") or "").strip().upper()
            if up in {"YES", "NO"}:
                done[row[key_col]] = (up, row.get("user_prompt_reason") or "")
    return done

# --------------- Main -----------------
def main():
    ap = argparse.ArgumentParser(description="Classify CSV titles for 'user-prompted' YES/NO.")
    ap.add_argument("-i", "--input", required=True, help="Input CSV path")
    ap.add_argument("-o", "--output", help="Output CSV path (default: <input>_userprompt.csv)")
    ap.add_argument("--title-col", help="Exact column name for titles (auto-detect if omitted)")
    ap.add_argument("--delay", type=float, default=0.0, help="Seconds to sleep between LLM calls")
    ap.add_argument("--resume", action="store_true", help="Skip rows already in output")
    ap.add_argument("--system-prompt-file", help="Path to a txt file with custom system prompt")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise SystemExit(f"Input not found: {in_path}")

    out_path = Path(args.output) if args.output else in_path.with_name(in_path.stem + "_userprompt.csv")

    # Load / choose system prompt
    system_prompt = DEFAULT_SYSTEM_PROMPT
    if args.system_prompt_file:
        with open(args.system_prompt_file, "r", encoding="utf-8") as spf:
            system_prompt = spf.read()

    # For resume, load existing results keyed by title
    existing: Dict[str, Tuple[str, str]] = {}
    with in_path.open("r", newline="", encoding="utf-8-sig") as f_in:
        reader = csv.DictReader(f_in)
        headers = reader.fieldnames or []
        title_col = detect_title_col(headers, args.title_col)

    if args.resume:
        existing = load_existing_results(out_path, key_col=title_col)

    # Prepare writer
    with in_path.open("r", newline="", encoding="utf-8-sig") as f_in, \
         out_path.open("w", newline="", encoding="utf-8") as f_out:
        reader = csv.DictReader(f_in)
        headers = reader.fieldnames or []
        title_col = detect_title_col(headers, args.title_col)

        # Preserve all original columns + our 2
        out_fields = headers + [c for c in ["user_prompt", "user_prompt_reason"] if c not in headers]
        writer = csv.DictWriter(f_out, fieldnames=out_fields)
        writer.writeheader()

        processed = 0
        for row in reader:
            title = (row.get(title_col) or "").strip()
            if not title:
                row["user_prompt"] = "NO"
                row["user_prompt_reason"] = "No title provided."
                writer.writerow(row)
                continue

            if args.resume and title in existing:
                # Carry over from previously computed output
                up, reason = existing[title]
                row["user_prompt"] = up
                row["user_prompt_reason"] = reason
                writer.writerow(row)
                processed += 1
                continue

            try:
                raw = call_llm(
                    url=LLM_URL,
                    model=LLM_MODEL,
                    system_prompt=system_prompt,
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

            if args.delay > 0:
                time.sleep(args.delay)

    print(f"Done. Wrote: {out_path}")

if __name__ == "__main__":
    main()
