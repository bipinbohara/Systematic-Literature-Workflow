import os
import json
import time
import textwrap
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import requests
import yaml
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ------------------------- Paths & IO -------------------------
BASE_DIR   = Path(__file__).resolve().parent
INDEX_DIR  = BASE_DIR / "vector_db"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Config file (YAML). You can override via env: SURVEY_CONFIG=...
CONFIG_PATH = Path(os.environ.get("SURVEY_CONFIG", BASE_DIR / "survey_config.yaml"))

# Embedding model must match your FAISS index
EMBED_MODEL = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# LLM endpoint defaults (override with env if needed)
LLM_URL     = os.environ.get("LLM_URL",   "http://192.168.0.213:8080/v1/completions")
LLM_MODEL   = os.environ.get("LLM_MODEL", "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8")
LLM_API_KEY = os.environ.get("LLM_API_KEY")  # often not needed for local

TIMEOUT_SECONDS = int(os.environ.get("TIMEOUT_SECONDS", "600"))

# Budgets
MAX_OUTPUT_TOKENS_NOTES  = int(os.environ.get("MAX_OUTPUT_TOKENS_NOTES", "200000"))
MAX_OUTPUT_TOKENS_SURVEY = int(os.environ.get("MAX_OUTPUT_TOKENS_SURVEY", "200000"))
CHUNK_CHAR_LIMIT         = int(os.environ.get("CHUNK_CHAR_LIMIT", "200000"))

# Output toggles
WRITE_MARKDOWN  = os.environ.get("WRITE_MARKDOWN", "1") == "1"
WRITE_LATEX     = os.environ.get("WRITE_LATEX", "1") == "1"
SURVEY_BASENAME = os.environ.get("SURVEY_BASENAME", "literature_survey")

# ------------------------- Helpers -------------------------
def load_yaml_config(path: Path) -> Dict:
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = {}
    return apply_default_config(cfg)

def apply_default_config(cfg: Dict) -> Dict:
    # Defaults for a generic academic survey
    defaults = {
        "domain_description": (
            "Produce a single scholarly survey that synthesizes the current state of knowledge for the target field."
        ),
        "inclusion_exclusion": {
            "include_if": [
                "Peer-reviewed research articles",
                "Written in English",
            ],
            "exclude_if": [
                "Non-scholarly sources",
                "Duplicate/retracted papers",
            ],
            # Optional domain-specific guardrails (leave empty for generic)
            "domain_rules": []
        },
        # What facts to extract into hidden notes per paper
        "extraction_schema": [
            "Study type and setting (e.g., human/animal/simulated/system evaluation)",
            "Problem/task definition and scope",
            "Datasets or benchmarks (or system workloads)",
            "Methods/architectures/algorithms/protocols",
            "Metrics/evaluation criteria",
            "Key findings (directionality, effect sizes if available)",
            "Limitations/confounders/threats to validity"
        ],
        # Outline for the integrated survey
        "survey_outline": [
            "Title",
            "Abstract",
            "1. Introduction",
            "2. Background and Definitions",
            "3. Corpus and Screening Criteria",
            "4. Taxonomy of Approaches / Study Designs",
            "5. Methods and Evaluation Protocols",
            "6. Results and Cross-Study Synthesis",
            "7. Confounders and Threats to Validity",
            "8. Methodological Pitfalls and Bias",
            "9. Open Problems and Future Directions",
            "10. Limitations",
            "11. Conclusion",
            "References"
        ],
        # How to render in-text citations and references
        "citation": {
            "style": "filename",  # "filename" | "author_year" | "title_year"
            "fallback": "filename"
        },
        # System prompt prefix to steer tone/rigor (domain-agnostic by default)
        "system_prompt_prefix": (
            "You are a meticulous scholarly assistant. Enforce the inclusion/exclusion criteria. "
            "Create only an integrated survey—no per-paper summaries."
        ),
        # Optional: title override (else model writes it)
        "title_override": None
    }
    # Merge shallowly
    merged = {**defaults, **cfg}
    # Deep-merge a few keys
    for k in ("inclusion_exclusion", "citation"):
        merged[k] = {**defaults[k], **cfg.get(k, {})}
    if "survey_outline" not in cfg:
        merged["survey_outline"] = defaults["survey_outline"]
    if "extraction_schema" not in cfg:
        merged["extraction_schema"] = defaults["extraction_schema"]
    return merged

def build_system_prompt(cfg: Dict) -> str:
    parts = [cfg.get("system_prompt_prefix", "")]
    desc  = cfg.get("domain_description", "")
    if desc:
        parts.append(desc)

    inc = cfg.get("inclusion_exclusion", {})
    include_if = inc.get("include_if", [])
    exclude_if = inc.get("exclude_if", [])
    domain_rules = inc.get("domain_rules", [])

    if include_if or exclude_if or domain_rules:
        parts.append("Screening policy:")
        if include_if:
            parts.append("- Include if: " + "; ".join(include_if))
        if exclude_if:
            parts.append("- Exclude if: " + "; ".join(exclude_if))
        if domain_rules:
            parts.append("- Domain rules: " + "; ".join(domain_rules))
    return " ".join(p for p in parts if p).strip()

def load_vector_store(index_dir: Path) -> FAISS:
    if not (index_dir / "index.faiss").exists():
        raise FileNotFoundError(f"FAISS index not found at {index_dir}. Build it first.")
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

# ---- Citation helpers ----
def build_citation_key(meta: Dict, src_path: str, style: str, fallback: str) -> str:
    filename = Path(src_path).name
    if style == "filename":
        return filename
    # Try to use metadata if available
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
    # fallback
    if fallback == "filename":
        return filename
    return filename

# ---- Internal notes (hidden) ----
def compress_single_paper_notes(source_name: str, chunks: List[Tuple[str, Dict, str]],
                                cfg: Dict) -> Tuple[str, Dict]:
    """
    Returns (notes_text, meta_agg). notes_text are hidden bullet notes for one paper.
    meta_agg is a merged metadata dict used for citations (best-effort).
    """
    schema = cfg["extraction_schema"]
    sys_prompt = build_system_prompt(cfg)

    # merge metadata across chunks (last one wins)
    meta_agg: Dict = {}
    for _, m, _ in chunks:
        meta_agg.update(m or {})

    notes_list = []
    for i, (_did, meta, txt) in enumerate(chunks, start=1):
        if not txt.strip():
            continue
        for piece in chunk_text(txt, CHUNK_CHAR_LIMIT):
            prompt = textwrap.dedent(f"""
            Prepare INTERNAL bullet NOTES (not public summaries) for a literature survey.

            PAPER: {Path(source_name).name}
            CHUNK: {i}
            TEXT:
            ---
            {piece}
            ---

            Follow this schema. Only bullets, terse factual items:
            {os.linesep.join([f"- {item}" for item in schema])}

            Also capture screening tags: INCLUDED/EXCLUDED with reason based on the policy.
            Do not write prose. Avoid redundancy across chunks; prefer compact bullets.
            """).strip()

            out = call_llm(
                url=LLM_URL,
                model=LLM_MODEL,
                system_prompt=sys_prompt,
                user_content=prompt,
                api_key=LLM_API_KEY,
                max_tokens=MAX_OUTPUT_TOKENS_NOTES
            )
            notes_list.append(out.strip())

    merged = "\n\n".join(notes_list)
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
        system_prompt=sys_prompt,
        user_content=dedupe_prompt,
        api_key=LLM_API_KEY,
        max_tokens=min(MAX_OUTPUT_TOKENS_NOTES, 900)
    )
    return final_notes.strip(), meta_agg

# ---- Synthesis into a single survey ----
def synthesize_survey(all_notes: Dict[str, str],
                      all_meta: Dict[str, Dict],
                      cfg: Dict) -> str:
    outline = cfg["survey_outline"]
    sys_prompt = build_system_prompt(cfg)
    cit_style = cfg["citation"]["style"]
    cit_fallback = cfg["citation"].get("fallback", "filename")

    # Convert sources -> citation keys and pack notes
    packet_parts = []
    key_map = {}
    for src, notes in all_notes.items():
        key = build_citation_key(all_meta.get(src, {}), src, cit_style, cit_fallback)
        key_map[src] = key
        packet_parts.append(f"[{key}]\n{notes}\n")
    notes_packet = "\n".join(packet_parts)

    structure_text = "\n".join(f"- {sec}" for sec in outline)
    title_override = cfg.get("title_override")

    survey_prompt = textwrap.dedent(f"""
    Write ONE integrated scholarly survey (no per-paper summaries) using the INTERNAL NOTES below.
    Requirements:
    - Enforce the screening policy from the system prompt. Ignore EXCLUDED papers except to justify exclusion in the screening section.
    - Synthesize across included papers; discuss agreements, disagreements, and reasons.
    - Use a neutral academic tone.
    - When referring to evidence, cite inline with the provided square-bracket keys, e.g., [Key123].
    - Follow the outline EXACTLY.
    - "References" should list cited keys (one per line). Do not fabricate metadata.

    {"Use this title: " + title_override if title_override else "Begin with an apt, succinct title."}

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
        system_prompt=sys_prompt,
        user_content=survey_prompt,
        api_key=LLM_API_KEY,
        max_tokens=MAX_OUTPUT_TOKENS_SURVEY
    )
    return survey_md.strip()

# ---- Minimal Markdown -> LaTeX (optional) ----
def md_to_basic_latex(md_text: str, default_title: str = "Survey") -> str:
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
    cfg = load_yaml_config(CONFIG_PATH)

    print("Loading FAISS…")
    vs = load_vector_store(INDEX_DIR)

    print("Reading documents…")
    docs = extract_all_docs(vs)
    grouped = group_by_source(docs)
    print(f"Found {len(grouped)} sources.")

    # Build internal notes per source
    all_notes: Dict[str, str] = {}
    all_meta: Dict[str, Dict]  = {}

    for i, (src, chunks) in enumerate(grouped.items(), start=1):
        print(f"[{i}/{len(grouped)}] Compressing -> {Path(src).name}")
        try:
            notes, meta = compress_single_paper_notes(src, chunks, cfg)
            all_notes[src] = notes
            all_meta[src]  = meta
        except Exception as e:
            print(f"!! Error on {src}: {e}")

    if not all_notes:
        raise RuntimeError("No notes produced—check index content or config.")

    print("Synthesizing single survey…")
    survey_md = synthesize_survey(all_notes, all_meta, cfg)

    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    md_path  = OUTPUT_DIR / f"{SURVEY_BASENAME}_{ts}.md"
    tex_path = OUTPUT_DIR / f"{SURVEY_BASENAME}_{ts}.tex"

    if WRITE_MARKDOWN:
        md_path.write_text(survey_md, encoding="utf-8")
        print(f"[OK] Markdown survey -> {md_path}")

    if WRITE_LATEX:
        title_guess = cfg.get("title_override") or "Integrated Literature Survey"
        tex = md_to_basic_latex(survey_md, default_title=title_guess)
        tex_path.write_text(tex, encoding="utf-8")
        print(f"[OK] LaTeX survey    -> {tex_path}")

    print("Done.")

if __name__ == "__main__":
    main()
