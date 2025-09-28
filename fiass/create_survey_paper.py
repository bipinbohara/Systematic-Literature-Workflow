#!/usr/bin/env python3
"""
create_survey_paper.py
- Builds ONE integrated survey (Markdown) from all papers in a FAISS vector_db.
- No per-paper summaries in the output; only a unified synthesis.
- Overwrites a single file: output/literature_survey.md

Env vars you can set:
  LLM_URL="http://<host>:<port>/v1/chat/completions"
  LLM_MODEL="openai/gpt-oss-120b"
  LLM_API_KEY="<optional>"
  EMBED_MODEL="NeuML/pubmedbert-base-embeddings"
  VECTOR_DB_DIR="/abs/path/to/vector_db"       # optional override
  AUTO_BUILD_VECTOR_DB="1"                     # call preprocess_pdf.vectorize_pdf() if index missing
  MAX_OUTPUT_TOKENS_NOTES="900"
  MAX_OUTPUT_TOKENS_SURVEY="4096"
  CHUNK_CHAR_LIMIT="4800"
  WRITE_MARKDOWN="1"                           # always 1 for this script
  WRITE_LATEX="0"                              # keep 0 to avoid extra files
  SURVEY_BASENAME="literature_survey"          # final filename (no timestamp)
"""

import os
import json
import time
import textwrap
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import requests
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from preprocess_pdf import vectorize_pdf 

# ------------------------- Paths & IO -------------------------
BASE_DIR   = Path(__file__).resolve().parent
DEFAULT_INDEX_DIR = BASE_DIR / "vector_db"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Optional overrides
VECTOR_DB_DIR_ENV = os.environ.get("VECTOR_DB_DIR")
AUTO_BUILD = os.environ.get("AUTO_BUILD_VECTOR_DB", "0") == "1"

# Embedding model must match how the index was built
EMBED_MODEL = os.environ.get("EMBED_MODEL", "NeuML/pubmedbert-base-embeddings")

# LLM endpoint defaults (chat endpoint strongly recommended)
LLM_URL     = os.environ.get("LLM_URL",   "http://192.168.0.213:8080/v1/completions")
LLM_MODEL   = os.environ.get("LLM_MODEL", "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8")
LLM_API_KEY = os.environ.get("LLM_API_KEY")  # often not needed for local

TIMEOUT_SECONDS = int(os.environ.get("TIMEOUT_SECONDS", "1000"))

# Budgets (sane defaults to avoid server 500s)
MAX_OUTPUT_TOKENS_NOTES  = int(os.environ.get("MAX_OUTPUT_TOKENS_NOTES", "200000"))
MAX_OUTPUT_TOKENS_SURVEY = int(os.environ.get("MAX_OUTPUT_TOKENS_SURVEY", "200000"))
CHUNK_CHAR_LIMIT         = int(os.environ.get("CHUNK_CHAR_LIMIT", "200000"))

# Output toggles — single file (no timestamp), LaTeX off by default
WRITE_MARKDOWN  = True
WRITE_LATEX     = os.environ.get("WRITE_LATEX", "0") == "1"  # keep false to avoid extra files
SURVEY_BASENAME = os.environ.get("SURVEY_BASENAME", "literature_survey")

# ------------------------- Domain-agnostic defaults -------------------------
DEFAULT_SYSTEM_PROMPT_PREFIX = (
    "You are a meticulous scholarly assistant. Enforce the inclusion/exclusion criteria exactly. "
    "Produce one cohesive synthesis (no per-paper summaries). Use neutral academic tone."
)

DEFAULT_DOMAIN_DESCRIPTION = (
    "Produce a single integrated, scholarly survey synthesizing the included research papers. "
    "Focus on cross-study patterns, methodological rigor, and open questions."
)

INCLUSION_EXCLUSION = {
    "include_if": [
        "Peer-reviewed research articles",
        "Written in English",
    ],
    "exclude_if": [
        "Non-scholarly sources",
        "Duplicate or retracted works",
    ],
    "domain_rules": [
        # Put domain-specific rules here (optional), e.g.,
        # "Human-only original research",
        # "Exclude conference abstracts and reviews",
    ],
}

EXTRACTION_SCHEMA = [
    "Study type and setting (e.g., human/animal/simulated/system evaluation)",
    "Problem/task definition and scope",
    "Datasets or benchmarks (or system workloads)",
    "Methods/architectures/algorithms/protocols",
    "Metrics/evaluation criteria",
    "Key findings (directionality, effect sizes if available)",
    "Limitations/confounders/threats to validity"
]

SURVEY_OUTLINE = [
    "Title",
    "Abstract",
    "1. Introduction",
    "2. Background and Motivation",
    "3. Materials and Methods",
    "4. Experimental Procedure",
    "5. Results Discussion",
    "6. Open Problems and Future Directions",
    "7. Conclusion",
    "References"
]

CITATION_STYLE = "filename"  # "filename" | "author_year" | "title_year"
CITATION_FALLBACK = "filename"
TITLE_OVERRIDE = None  # or set a fixed title string

# ------------------------- Helpers -------------------------
def build_system_prompt() -> str:
    parts = [DEFAULT_SYSTEM_PROMPT_PREFIX, DEFAULT_DOMAIN_DESCRIPTION]
    inc = INCLUSION_EXCLUSION
    include_if = inc.get("include_if", [])
    exclude_if = inc.get("exclude_if", [])
    domain_rules = inc.get("domain_rules", [])
    parts.append("Screening policy:")
    if include_if:
        parts.append("- Include if: " + "; ".join(include_if))
    if exclude_if:
        parts.append("- Exclude if: " + "; ".join(exclude_if))
    if domain_rules:
        parts.append("- Domain rules: " + "; ".join(domain_rules))
    return " ".join(p for p in parts if p).strip()

def find_existing_index(start_dir: Path) -> Optional[Path]:
    try:
        for p in start_dir.rglob("index.faiss"):
            rel_depth = len(p.relative_to(start_dir).parts)
            if rel_depth <= 4:
                return p.parent
    except Exception:
        pass
    return None

def load_vector_store(default_index_dir: Path) -> FAISS:
    index_dir: Optional[Path] = None

    if VECTOR_DB_DIR_ENV:
        candidate = Path(VECTOR_DB_DIR_ENV).expanduser().resolve()
        if not (candidate / "index.faiss").exists():
            vectorize_pdf()
            #raise FileNotFoundError(f"VECTOR_DB_DIR points to '{candidate}', but index.faiss not found there.")
        index_dir = candidate
    elif (default_index_dir / "index.faiss").exists():
        index_dir = default_index_dir
    else:
        found = find_existing_index(BASE_DIR)
        if found:
            index_dir = found
            print(f"[info] Found FAISS index at: {index_dir}")
        elif AUTO_BUILD:
            try:
                from preprocess_pdf import vectorize_pdf
            except Exception as e:
                raise FileNotFoundError(
                    f"No FAISS index and AUTO_BUILD is on, but cannot import preprocess_pdf.vectorize_pdf(): {e}"
                )
            print("[info] No FAISS index found; building via preprocess_pdf.vectorize_pdf() …")
            vectorize_pdf()
            if (default_index_dir / "index.faiss").exists():
                index_dir = default_index_dir
            else:
                located = find_existing_index(BASE_DIR)
                if not located:
                    raise FileNotFoundError(
                        "Attempted auto-build, but index.faiss still not found. "
                        "Ensure vectorize_pdf() writes to vector_db or set VECTOR_DB_DIR."
                    )
                index_dir = located
        else:
            raise FileNotFoundError(
                f"FAISS index not found at {default_index_dir}. "
                f"Either set VECTOR_DB_DIR or enable AUTO_BUILD_VECTOR_DB=1."
            )

    print(f"[info] Using FAISS index at: {index_dir}")
    print(f"[info] Loading embeddings model for retrieval: {EMBED_MODEL}")
    embeddings = HuggingFaceEmbeddings(model=EMBED_MODEL, show_progress=True)
    return FAISS.load_local(
        index_dir,
        embeddings,
        normalize_L2=True,
        allow_dangerous_deserialization=True,
    )

def extract_all_docs(vs: FAISS) -> List[Tuple[str, Dict, str]]:
    pos_to_id = vs.index_to_docstore_id
    docdict   = vs.docstore._dict
    rows = []
    for pos in sorted(pos_to_id.keys()):
        did = pos_to_id[pos]
        doc = docdict.get(did)
        if not doc:
            continue
        text = (doc.page_content or "").strip()
        meta = dict(doc.metadata or {})
        rows.append((did, meta, text))
    return rows

def group_by_source(docs: List[Tuple[str, Dict, str]]) -> Dict[str, List[Tuple[str, Dict, str]]]:
    grouped = {}
    for did, meta, txt in docs:
        src = meta.get("source") or "UNKNOWN_SOURCE"
        grouped.setdefault(src, []).append((did, meta, txt))
    for src, lst in grouped.items():
        lst.sort(key=lambda x: (x[1].get("page", 10**8), x[1].get("chunk", 10**8), x[0]))
    return grouped

def call_llm(url: str, model: str, system_prompt: str, user_content: str,
             api_key: Optional[str] = None, max_tokens: int = 1024) -> str:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    is_completions = url.rstrip("/").endswith("/v1/completions")
    if is_completions:
        # Strongly recommend chat endpoint; but support text if needed
        payload = {
            "model": model,
            "prompt": f"System: {system_prompt}\n\nUser:\n{user_content}",
            "max_tokens": max_tokens,
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
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": False,
        }
    r = requests.post(url, headers=headers, json=payload, timeout=TIMEOUT_SECONDS)
    if r.status_code != 200:
        raise RuntimeError(f"LLM error {r.status_code}: {r.text}")
    data = r.json()
    if is_completions:
        return data["choices"][0]["text"]
    return data["choices"][0]["message"]["content"]

def chunk_text(s: str, limit: int) -> List[str]:
    s = s.strip()
    if len(s) <= limit:
        return [s]
    parts, start = [], 0
    while start < len(s):
        end = min(start + limit, len(s))
        if end < len(s):
            ws = s.rfind(" ", start, end)
            if ws > start + int(limit * 0.6):
                end = ws
        parts.append(s[start:end].strip())
        start = end
    return parts

def build_citation_key(meta: Dict, src_path: str, style: str, fallback: str) -> str:
    filename = Path(src_path).name
    if style == "filename":
        return filename
    first_author = (meta.get("author") or meta.get("authors") or "")
    if isinstance(first_author, list) and first_author:
        first_author = first_author[0]
    if isinstance(first_author, str):
        first_author = first_author.split(",")[0].split()[-1] if first_author else ""
    year = str(meta.get("year") or meta.get("publication_year") or "")
    title = (meta.get("title") or "").strip()
    if style == "author_year" and (first_author or year):
        key = f"{first_author}{year}".strip()
        return key if key else filename
    if style == "title_year" and (title or year):
        short_title = "_".join(title.split()[:4]) if title else ""
        key = (short_title + (year and f"_{year}" or "")).strip("_")
        return key if key else filename
    return filename if fallback == "filename" else filename

# ---- Internal notes (hidden) ----
def compress_single_paper_notes(source_name: str,
                                chunks: List[Tuple[str, Dict, str]],
                                system_prompt: str) -> Tuple[str, Dict]:
    """
    Returns (notes_text, meta_agg). notes_text are hidden bullet notes for one paper.
    meta_agg is merged metadata across chunks (best-effort).
    """
    meta_agg: Dict = {}
    for _, m, _ in chunks:
        meta_agg.update(m or {})

    notes_list = []
    for i, (_did, meta, txt) in enumerate(chunks, start=1):
        if not txt.strip():
            continue
        for piece in chunk_text(txt, CHUNK_CHAR_LIMIT):
            schema_bullets = os.linesep.join([f"- {item}" for item in EXTRACTION_SCHEMA])
            prompt = textwrap.dedent(f"""
            Prepare INTERNAL bullet NOTES (not public summaries) for a literature survey.

            PAPER: {Path(source_name).name}
            CHUNK: {i}
            TEXT:
            ---
            {piece}
            ---

            Follow this schema. Only bullets, terse factual items:
            {schema_bullets}

            Also capture screening tags: INCLUDED/EXCLUDED with reason based on the policy.
            Do not write prose. Avoid redundancy across chunks; prefer compact bullets.
            """).strip()

            out = call_llm(
                url=LLM_URL,
                model=LLM_MODEL,
                system_prompt=system_prompt,
                user_content=prompt,
                api_key=LLM_API_KEY,
                max_tokens=MAX_OUTPUT_TOKENS_NOTES
            )
            notes_list.append(out.strip())

    merged = "\n\n".join(notes_list) if notes_list else ""
    if not merged:
        return "", meta_agg

    dedupe_prompt = textwrap.dedent(f"""
    Merge and deduplicate the INTERNAL bullet NOTES for the paper {Path(source_name).name}.

    NOTES:
    ---
    {merged}
    ---

    Output compact non-redundant NOTES that:
    - Start with "SCREENING: INCLUDED" or "SCREENING: EXCLUDED (reason ...)"
    - Preserve key facts per the schema
    - Keep bullets only; no prose
    - Keep it short
    """).strip()

    final_notes = call_llm(
        url=LLM_URL,
        model=LLM_MODEL,
        system_prompt=system_prompt,
        user_content=dedupe_prompt,
        api_key=LLM_API_KEY,
        max_tokens=min(MAX_OUTPUT_TOKENS_NOTES, 900)
    )
    return final_notes.strip(), meta_agg

# ---- Synthesis into a single survey ----
def synthesize_survey(all_notes: Dict[str, str],
                      all_meta: Dict[str, Dict],
                      system_prompt: str) -> str:
    outline = SURVEY_OUTLINE
    cit_style = CITATION_STYLE
    cit_fallback = CITATION_FALLBACK

    packet_parts = []
    for src, notes in all_notes.items():
        key = build_citation_key(all_meta.get(src, {}), src, cit_style, cit_fallback)
        packet_parts.append(f"[{key}]\n{notes}\n")
    notes_packet = "\n".join(packet_parts)

    structure_text = "\n".join(f"- {sec}" for sec in outline)

    survey_prompt = textwrap.dedent(f"""
    Write ONE integrated scholarly survey (no per-paper summaries) using the INTERNAL NOTES below.
    Requirements:
    - Enforce the screening policy from the system prompt. Ignore EXCLUDED papers except to justify exclusion in the screening section.
    - Synthesize across included papers; discuss agreements, disagreements, and reasons.
    - Use a neutral academic tone.
    - When referring to evidence, cite inline with the provided square-bracket keys, e.g., [Key123].
    - Follow the outline EXACTLY.
    - "References" should list cited keys (one per line). Do not fabricate metadata.

    {"Use this title: " + TITLE_OVERRIDE if TITLE_OVERRIDE else "Begin with an apt, succinct title."}

    OUTLINE:
    {structure_text}

    INTERNAL NOTES (by citation key):
    ---
    {notes_packet}
    ---
    """).strip()

    survey_md = call_llm(
        url=LLM_URL,
        model=LLM_MODEL,
        system_prompt=system_prompt,
        user_content=survey_prompt,
        api_key=LLM_API_KEY,
        max_tokens=MAX_OUTPUT_TOKENS_SURVEY
    )
    return survey_md.strip()

# ---- Minimal Markdown -> LaTeX (optional; off by default) ----
def md_to_basic_latex(md_text: str, default_title: str = "Integrated Literature Survey") -> str:
    lines = md_text.splitlines()
    out = []
    for ln in lines:
        if ln.startswith("# "):
            out.append("\\section*{" + ln[2:].strip().replace("{","\\{").replace("}","\\}") + "}")
        elif ln.startswith("## "):
            out.append("\\subsection*{" + ln[3:].strip().replace("{","\\{").replace("}","\\}") + "}")
        elif ln.startswith("### "):
            out.append("\\subsubsection*{" + ln[4:].strip().replace("{","\\{").replace("}","\\}") + "}")
        elif ln.startswith("- "):
            out.append("\\begin{itemize}\\item " + ln[2:].strip() + "\\end{itemize}")
        else:
            out.append(ln.replace("%", "\\%"))
    body = "\n\n".join(out)

    latex = fr"""
\documentclass[10pt,conference]{IEEEtran}
\usepackage[utf8]{inputenc}
\usepackage{url}
\usepackage{hyperref}
\usepackage{graphicx}
\usepackage{amsmath,amssymb}
\begin{document}
\title{{{default_title}}}
\author{{\IEEEauthorblockN{{Anonymous}}}}
\maketitle
\begin{abstract}
(This abstract is autogenerated. Consider revising.)
\end{abstract}

{body}

\end{document}
"""
    return latex

# ------------------------- Main -------------------------
def main():
    load_dotenv()
    print(str(BASE_DIR))
    print(str(DEFAULT_INDEX_DIR))
    print(str(BASE_DIR / "data"))

    system_prompt = build_system_prompt()

    print("Loading FAISS…")
    vs = load_vector_store(DEFAULT_INDEX_DIR)

    print("Reading documents…")
    docs = extract_all_docs(vs)
    grouped = group_by_source(docs)
    print(f"Found {len(grouped)} sources.")

    all_notes: Dict[str, str] = {}
    all_meta: Dict[str, Dict]  = {}

    for i, (src, chunks) in enumerate(grouped.items(), start=1):
        print(f"[{i}/{len(grouped)}] Compressing -> {Path(src).name}")
        try:
            notes, meta = compress_single_paper_notes(src, chunks, system_prompt)
            if notes:
                all_notes[src] = notes
                all_meta[src]  = meta
            else:
                print(f"   (no notes produced for {Path(src).name})")
        except Exception as e:
            print(f"!! Error on {Path(src).name}: {e}")

    if not all_notes:
        raise RuntimeError("No notes produced—check index content, budgets, and LLM endpoint.")

    print("Synthesizing single survey…")
    survey_md = synthesize_survey(all_notes, all_meta, system_prompt)

    # Overwrite stable filename (no timestamp). Single output by design.
    md_path  = OUTPUT_DIR / f"{SURVEY_BASENAME}.md"
    tex_path = OUTPUT_DIR / f"{SURVEY_BASENAME}.tex"

    if WRITE_MARKDOWN:
        md_path.write_text(survey_md, encoding="utf-8")
        print(f"[OK] Markdown survey -> {md_path}")

    if WRITE_LATEX:
        tex = md_to_basic_latex(survey_md, default_title=TITLE_OVERRIDE or "Integrated Literature Survey")
        tex_path.write_text(tex, encoding="utf-8")
        print(f"[OK] LaTeX survey    -> {tex_path}")

    print("Done.")

if __name__ == "__main__":
    main()
