import os
import json
from pathlib import Path

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Document,
)
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.settings import Settings
from llama_index.readers.file.pymu_pdf import PyMuPDFReader

from huggingface_hub import login


class ConcatPyMuPDFReader(PyMuPDFReader):
    """Concatenate all pages of a PDF into a single Document."""

    def load(self, file_path, metadata=True, extra_info=None):
        import fitz

        pdf = fitz.open(file_path)
        text = "".join(page.get_text() for page in pdf)
        info = extra_info or {}
        if metadata:
            info.update(
                {"total_pages": len(pdf), "file_path": str(Path(file_path).resolve())}
            )
        return [Document(text=text, extra_info=info)]

def get_rag_data(prompt_keyword: str, *, rebuild=False, refresh=False, cutoff=0.5):
    """
    Run a RAG query against the local PDF corpus.

    Parameters
    ----------
    prompt_keyword : str
        The search string.
    rebuild : bool, optional
        Force a full rebuild of the index (default: False).
    refresh : bool, optional
        Pick up any new or changed PDFs without full rebuild (default: False).
    cutoff : float, optional
        Minimum similarity score to keep a node (default: 0.5).

    Returns
    -------
    list[dict]
        A list of JSON-serialisable dicts with title, source, score, and content.
    """
    # ---- auth & models -------------------------------------------------------

    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-large-en-v1.5", embed_batch_size=32
    )
    Settings.llm = Ollama(model="mistral")  # or your preferred local model

    # ---- (re)build or load index --------------------------------------------
    PERSIST_DIR = "./index"
    if rebuild or not os.path.isdir(PERSIST_DIR):
        reader = SimpleDirectoryReader(
            input_dir="data",
            recursive=True,
            file_extractor={".pdf": ConcatPyMuPDFReader()},
        )
        all_docs = [doc for batch in reader.iter_data() for doc in batch]
        index = VectorStoreIndex.from_documents(all_docs, show_progress=True)
        index.storage_context.persist(PERSIST_DIR)
    else:
        ctx = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(ctx)
        if refresh:
            # automatically embed any unseen / modified files
            index.as_retriever().refresh("./data")

    # ---- retrieve -----------------------------------------------------------
    retriever = VectorIndexRetriever(index=index, similarity_top_k=100)
    nodes = retriever.retrieve(prompt_keyword)

    # optional similarity filter
    #nodes = [n for n in nodes if (n.score or 0) >= cutoff]

    results = []

    for node in nodes:
        if node.score >= 0.74:
            metadata = node.metadata or {}
            file_path = metadata.get("file_path", "Unknown source")
            file_name = os.path.basename(file_path)

            # Fallback logic for title/abstract parsing
            #content = node.get_content().strip()
            #lines = [line.strip() for line in content.split("\n") if line.strip()]
            #title = metadata.get("title", lines[0] if lines else "Untitled")
            #abstract = "\n".join(lines[1:]) if len(lines) > 1 else ""

            result = {
                # "title": title,
                # "authors": metadata.get("authors", "Unknown"),
                # "journal": metadata.get("journal", "Unknown"),
                # "abstract": abstract[:2048],  # Optional truncation
                "source": file_name,
                "score": round(node.score or 0, 3),
                "content": node.get_content().strip()
            }
            results.append(result)
    return results


    # grouped_by_pdf = {}
    # for node in nodes:
    #     path = node.metadata.get("file_path", "Unknown")
    #     if path not in grouped_by_pdf:
    #         grouped_by_pdf[path] = {
    #             "source": os.path.basename(path),
    #             "score": round(node.score or 0, 3),
    #             "content": ""
    #         }
    #     grouped_by_pdf[path]["content"] += node.get_content().strip() + "\n\n"

    # # Step 7: Final JSON output
    # json_response = list(grouped_by_pdf.values())

    # # Optional: Print or return
    # import json
    # print(json.dumps(json_response, indent=2))

    # # ---- jsonify -------------------------------------------------------------
    # json_docs = []
    # for node in nodes:
    #   if node.score < 0.74:
    #     continue
    #     #print(f"DROPPED: {node.score:.3f} | {node.get_content()[:60]}")
    #   else:
    #     meta = node.metadata or {}
    #     file_path = meta.get("file_path", "unknown")
    #     title = meta.get(
    #         "title", next((l for l in node.get_content().splitlines() if l), "Untitled")
    #     )
    #     json_docs.append(
    #         {
    #             "title": title[:120],
    #             "source": Path(file_path).name,
    #             "score": round(node.score or 0.0, 3),
    #             "content": node.get_content().strip(),
    #         }
    #     )

    # #print(json.dumps(json_docs, indent=2))
    # return json_docs


# import os
# import json
# from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, load_index_from_storage
# from llama_index.core.retrievers import VectorIndexRetriever
# from llama_index.core.query_engine import RetrieverQueryEngine
# from llama_index.core.postprocessor import SimilarityPostprocessor
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from llama_index.llms.ollama import Ollama
# from llama_index.core.settings import Settings

# from huggingface_hub import login


# def get_rag_data(prompt_keyword):
#     # Set embedding and LLM (all open source)
#     Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5", embed_batch_size=32)

#     # Load or build index
#     PERSIST_DIR = "./index"

#     if not os.path.exists(PERSIST_DIR):
#         # Load and index documents
#         documents = SimpleDirectoryReader("data").load_data()
#         index = VectorStoreIndex.from_documents(documents, show_progress=True)
#         index.storage_context.persist(persist_dir=PERSIST_DIR)
#     else:
#         storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
#         index = load_index_from_storage(storage_context)

#     # Set up retriever and query engine
#     retriever = VectorIndexRetriever(index=index, similarity_top_k=100)
#     #postprocessor = SimilarityPostprocessor(similarity_cutoff=0.75)

#     # 4. Manual retrieval
#     #query = "maternal vaccination"
#     query = prompt_keyword
#     nodes = retriever.retrieve(query)
#     #filtered_nodes = postprocessor.postprocess_nodes(nodes)

#     # 5. Display top results
#     json_docs = []
#     for node in nodes:
#         if node.score < 0.75:
#             print(f"DROPPED: {node.score:.3f} | {node.get_content()[:60]}")
#         else:
#             metadata = node.metadata or {}

#             file_path = metadata.get('file_path', 'Unknown source')
#             file_name = os.path.basename(file_path)

#             # Try to extract a meaningful title
#             content_lines = [line.strip() for line in node.get_content().split('\n') if line.strip()]
#             title = metadata.get('title', content_lines[0] if content_lines else 'Untitled Document')

#             doc_json = {
#                 "title": title,
#                 "source": file_name,
#                 "score": round(node.score or 0, 3),
#                 "content": node.get_content().strip()
#             }

#             json_docs.append(doc_json)

#     print(json.dumps(json_docs, indent=2))
#     return json_docs
