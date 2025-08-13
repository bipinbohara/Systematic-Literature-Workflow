import asyncio
import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple, Optional

import aiohttp
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ------------------ Paths & Config ------------------
BASE_DIR     = Path(__file__).resolve().parent
INDEX_PATH   = BASE_DIR / "vector_db"
OUTPUT_DIR   = BASE_DIR / "output"
EMBED_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"

# LLM config (can override via environment)
LLM_URL      = os.environ.get("LLM_URL",   "http://192.168.0.203:80/v1/chat/completions")
LLM_MODEL    = os.environ.get("LLM_MODEL", "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")
#LLM_API_KEY  = os.environ.get("LLM_API_KEY")
SYSTEM_PROMPT = (
    "You are a research assistant. Summarize the key findings, methods, and "
    "limitations in 3–6 concise bullets."
)
TIMEOUT_SEC = 600

# ------------------ FAISS Helpers ------------------
def load_vector_store(index_dir: Path, embed_model: str) -> FAISS:
    if not (index_dir / "index.faiss").exists():
        raise FileNotFoundError(f"Missing FAISS index at: {index_dir/'index.faiss'}")
    embeddings = HuggingFaceEmbeddings(model=embed_model, show_progress=True)
    vs = FAISS.load_local(
        index_dir,
        embeddings,
        normalize_L2=True,
        allow_dangerous_deserialization=True,
    )
    return vs

def iter_faiss_entries(vs: FAISS) -> Iterable[Tuple[int, str, Dict[str, Any]]]:
    """
    Yield (faiss_pos, doc_id, payload) for each stored vector entry.
    payload = { "source": str, "content": str, "metadata": dict }
    """
    pos_to_id = vs.index_to_docstore_id  # Dict[int, str]
    docdict   = vs.docstore._dict        # Dict[str, Document]

    for pos in sorted(pos_to_id.keys()):
        doc_id = pos_to_id[pos]
        doc = docdict.get(doc_id)
        if doc is None:
            continue
        content = (doc.page_content or "").strip()
        meta = dict(doc.metadata or {})
        source = meta.get("source", "N/A")
        yield pos, doc_id, {"source": source, "content": content, "metadata": meta}

# ------------------ LLM Caller ------------------
async def call_llm(
    session: aiohttp.ClientSession,
    url: str,
    api_key: Optional[str],
    model: str,
    system_prompt: str,
    user_content: str,
) -> str:
    """
    Supports both /v1/chat/completions and /v1/completions (OpenAI-compatible).
    """
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    if url.rstrip("/").endswith("/v1/completions"):
        # Text completions style: send a single prompt
        prompt = f"System: {system_prompt}\n\nUser:\n{user_content}"
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": 512,
            "temperature": 0.0,
            "stream": False,
        }
    else:
        # Chat completions style
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

    async with session.post(url, json=payload, headers=headers, timeout=TIMEOUT_SEC) as resp:
        text = await resp.text()
        if resp.status != 200:
            raise RuntimeError(f"LLM error {resp.status}: {text}")
        data = await resp.json()

        if url.rstrip("/").endswith("/v1/completions"):
            # OpenAI text completions
            return data["choices"][0]["text"]
        else:
            # OpenAI chat completions
            return data["choices"][0]["message"]["content"]

# ------------------ Runner ------------------
async def main():
    load_dotenv()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"faiss_infer_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"

    vs = load_vector_store(INDEX_PATH, EMBED_MODEL)

    new_file = not out_path.exists()
    async with aiohttp.ClientSession() as session, out_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "faiss_pos",
                "doc_id",
                "source",
                "text_len",
                "metadata_json",
                "llm_output",
            ],
        )
        if new_file:
            writer.writeheader()

        # Sequential for simplicity; easy to parallelize later if needed
        async for row in _iter_rows(session, vs):
            writer.writerow(row)
            f.flush()

    print(f"Done. Results -> {out_path}")

async def _iter_rows(session: aiohttp.ClientSession, vs: FAISS):
    for faiss_pos, doc_id, payload in iter_faiss_entries(vs):
        text = payload["content"]
        meta_json = json.dumps(payload["metadata"], ensure_ascii=False)

        if not text:
            yield {
                "faiss_pos": faiss_pos,
                "doc_id": doc_id,
                "source": payload["source"],
                "text_len": 0,
                "metadata_json": meta_json,
                "llm_output": "[EMPTY_CONTENT]",
            }
            continue

        try:
            llm_out = await call_llm(
                session=session,
                url=LLM_URL,
                #api_key=LLM_API_KEY,
                model=LLM_MODEL,
                system_prompt=SYSTEM_PROMPT,
                user_content=text,
            )
        except Exception as e:
            llm_out = f"[LLM_ERROR] {repr(e)}"

        yield {
            "faiss_pos": faiss_pos,
            "doc_id": doc_id,
            "source": payload["source"],
            "text_len": len(text),
            "metadata_json": meta_json,
            "llm_output": llm_out,
        }

if __name__ == "__main__":
    asyncio.run(main())
