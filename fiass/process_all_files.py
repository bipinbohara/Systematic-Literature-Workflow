#!/usr/bin/env python3
"""
batch_faiss_vector_infer.py

Iterate every vector entry in a LangChain FAISS store, send the associated
Document to an OpenAI-compatible LLM endpoint, and append the outputs to a
single CSV file named by date/time.

Usage:
  python batch_faiss_vector_infer.py \
    --index-dir /path/to/vector_db \
    --llm-url http://localhost:8000/v1/chat/completions \
    --model deepseek-r1:latest \
    --concurrency 8

Notes:
- Assumes the FAISS index was saved with LangChain's FAISS wrapper and that
  the docstore is present alongside `index.faiss` in the same folder.
- Each FAISS entry corresponds to one Document chunk; we do NOT re-chunk here.
- Endpoint must be OpenAI-compatible (vLLM, OpenAI, etc.).
"""

import argparse
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


DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_vector_store(index_dir: Path, embed_model: str) -> FAISS:
    """Load FAISS store (and docstore) using the same embedding class."""
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

    payload = {
        "source": str,
        "content": str,
        "metadata": dict
    }
    """
    # Numeric FAISS positions -> docstore IDs
    pos_to_id = vs.index_to_docstore_id  # Dict[int, str]
    docdict = vs.docstore._dict          # Dict[str, Document]

    for pos in sorted(pos_to_id.keys()):
        doc_id = pos_to_id[pos]
        doc = docdict.get(doc_id)
        if doc is None:
            continue
        content = (doc.page_content or "").strip()
        meta = dict(doc.metadata or {})
        source = meta.get("source", "N/A")
        yield pos, doc_id, {"source": source, "content": content, "metadata": meta}


async def call_openai_compatible_chat(
    session: aiohttp.ClientSession,
    url: str,
    api_key: Optional[str],
    model: str,
    system_prompt: str,
    user_content: str,
    temperature: float = 0.0,
    max_tokens: int = 512,
    extra_headers: Optional[Dict[str, str]] = None,
) -> str:
    """POST to /v1/chat/completions and return assistant content."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if extra_headers:
        headers.update(extra_headers)

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }

    async with session.post(url, json=payload, headers=headers, timeout=600) as resp:
        text = await resp.text()
        if resp.status != 200:
            raise RuntimeError(f"LLM error {resp.status}: {text}")
        data = await resp.json()
        return data["choices"][0]["message"]["content"]


async def worker(
    name: str,
    queue: "asyncio.Queue[Tuple[int, str, Dict[str, Any]]]",
    session: aiohttp.ClientSession,
    out_path: Path,
    url: str,
    api_key: Optional[str],
    model: str,
    system_prompt: str,
    semaphore: asyncio.Semaphore,
) -> None:
    """Consume items and append results to CSV."""
    new_file = not out_path.exists()
    with out_path.open("a", newline="", encoding="utf-8") as f:
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

        while True:
            item = await queue.get()
            if item is None:  # sentinel
                queue.task_done()
                break

            faiss_pos, doc_id, payload = item
            text = payload["content"]
            meta_json = json.dumps(payload["metadata"], ensure_ascii=False)

            # Skip truly empty content rows (optional)
            if not text:
                writer.writerow(
                    {
                        "faiss_pos": faiss_pos,
                        "doc_id": doc_id,
                        "source": payload["source"],
                        "text_len": 0,
                        "metadata_json": meta_json,
                        "llm_output": "[EMPTY_CONTENT]",
                    }
                )
                f.flush()
                queue.task_done()
                continue

            async with semaphore:
                try:
                    llm_out = await call_openai_compatible_chat(
                        session=session,
                        url=url,
                        api_key=api_key,
                        model=model,
                        system_prompt=system_prompt,
                        user_content=text,
                    )
                except Exception as e:
                    llm_out = f"[LLM_ERROR] {repr(e)}"

            writer.writerow(
                {
                    "faiss_pos": faiss_pos,
                    "doc_id": doc_id,
                    "source": payload["source"],
                    "text_len": len(text),
                    "metadata_json": meta_json,
                    "llm_output": llm_out,
                }
            )
            f.flush()
            queue.task_done()


async def main_async(args):
    load_dotenv()

    index_dir = Path(args.index_dir).resolve()

    # Create dated output filename
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"faiss_infer_{stamp}.csv"

    vs = load_vector_store(index_dir, args.embed_model)

    # Prepare queue and HTTP session
    queue: "asyncio.Queue[Tuple[int, str, Dict[str, Any]]]" = asyncio.Queue(maxsize=args.queue_size)
    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=connector) as session:
        # Enqueue all entries
        for tup in iter_faiss_entries(vs):
            await queue.put(tup)

        # Add N sentinels (one per worker)
        for _ in range(args.workers):
            await queue.put(None)

        # Launch workers
        sem = asyncio.Semaphore(args.concurrency)
        tasks = [
            asyncio.create_task(
                worker(
                    name=f"w{idx}",
                    queue=queue,
                    session=session,
                    out_path=out_path,
                    url=args.llm_url,
                    api_key=args.api_key,
                    model=args.model,
                    system_prompt=args.system_prompt,
                    semaphore=sem,
                )
            )
            for idx in range(args.workers)
        ]

        await asyncio.gather(*tasks)

    print(f"Done. Results -> {out_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Infer over every FAISS vector entry and append results to a dated CSV.")
    p.add_argument("--index-dir", type=str, required=True, help="Path to FAISS dir (contains index.faiss).")
    p.add_argument("--out-dir", type=str, default=".", help="Directory to save the dated CSV.")
    p.add_argument("--embed-model", type=str, default=DEFAULT_EMBED_MODEL)

    # LLM endpoint
    p.add_argument("--llm-url", type=str, default=os.environ.get("LLM_URL", "http://127.0.0.1:8000/v1/chat/completions"))
    p.add_argument("--model", type=str, default=os.environ.get("LLM_MODEL", "gpt-4o-mini"))
    p.add_argument("--api-key", type=str, default=os.environ.get("LLM_API_KEY"))

    # Concurrency
    p.add_argument("--concurrency", type=int, default=8, help="Max concurrent LLM calls.")
    p.add_argument("--workers", type=int, default=4, help="Number of CSV writer workers.")
    p.add_argument("--queue-size", type=int, default=200, help="Backpressure buffer.")
    # Prompting
    p.add_argument(
        "--system-prompt",
        type=str,
        default="You are a research assistant. Summarize key findings, methods, and limitations in 3–6 bullets.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
