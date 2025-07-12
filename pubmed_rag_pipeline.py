import os
import fitz  # PyMuPDF
from typing import List
from llama_index.core import SimpleDirectoryReader, StorageContext, load_index_from_storage
from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.service_context import ServiceContext
from llama_index.embeddings.openai import OpenAIEmbedding  # or use HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI  # Replace with your LLM provider

# Paths
PDF_DIR = "storage/pdfs"
TXT_DIR = "storage/docs"
INDEX_DIR = "storage/index"

os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(TXT_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)


def pdf_to_text(pdf_path: str) -> str:
    """Extracts all text from a given PDF file."""
    try:
        doc = fitz.open(pdf_path)
        text = "\n".join(page.get_text() for page in doc)
        return text
    except Exception as e:
        print(f"Failed to extract from {pdf_path}: {e}")
        return ""


def save_text(pmid: str, text: str):
    """Saves extracted text to a .txt file."""
    with open(f"{TXT_DIR}/{pmid}.txt", "w", encoding="utf-8") as f:
        f.write(text)


def convert_all_pdfs_to_text():
    """Converts all PDFs in storage/pdfs to text if not already done."""
    for fname in os.listdir(PDF_DIR):
        if fname.endswith(".pdf"):
            pmid = fname.replace("PMID_", "").replace(".pdf", "")
            txt_path = os.path.join(TXT_DIR, f"{pmid}.txt")

            if not os.path.exists(txt_path):
                pdf_path = os.path.join(PDF_DIR, fname)
                print(f"Extracting text from {fname}")
                text = pdf_to_text(pdf_path)
                if text.strip():
                    save_text(pmid, text)
                else:
                    print(f"⚠️ No text extracted from {fname}")


def build_vector_index() -> VectorStoreIndex:
    """Creates a vector index from all .txt documents."""
    documents = SimpleDirectoryReader(TXT_DIR).load_data()
    service_context = ServiceContext.from_defaults(
        embed_model=OpenAIEmbedding(),  # Replace with HuggingFaceEmbedding() if desired
    )
    index = VectorStoreIndex.from_documents(documents, service_context=service_context)
    index.storage_context.persist(persist_dir=INDEX_DIR)
    return index


def query_rag_index(query: str, top_k: int = 100):
    """Performs vector search over the indexed documents."""
    storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
    service_context = ServiceContext.from_defaults(embed_model=OpenAIEmbedding())

    index = load_index_from_storage(storage_context, service_context=service_context)
    retriever = VectorIndexRetriever(index=index, similarity_top_k=top_k)
    engine = RetrieverQueryEngine(retriever=retriever, service_context=service_context)

    response = engine.query(query)
    return response

# # Extract text from PDF
# def pdf_to_text(pdf_path: str) -> str:
#     try:
#         doc = fitz.open(pdf_path)
#         text = "\n".join(page.get_text() for page in doc)
#         return text
#     except Exception as e:
#         print(f"Failed to extract from {pdf_path}: {e}")
#         return ""

# # Save text to .txt file
# def save_text(pmid: str, text: str):
#     with open(f"{TXT_DIR}/{pmid}.txt", "w", encoding="utf-8") as f:
#         f.write(text)

# # Build the vector index
# def build_vector_index() -> VectorStoreIndex:
#     documents = SimpleDirectoryReader(TXT_DIR).load_data()
    
#     # Setup service context (embedding + LLM, if needed)
#     service_context = ServiceContext.from_defaults(
#         embed_model=OpenAIEmbedding(),  # You can swap this for HuggingFaceEmbedding() or others
#     )
    
#     index = VectorStoreIndex.from_documents(documents, service_context=service_context)
#     index.storage_context.persist(persist_dir=INDEX_DIR)
#     return index

# # Query the index
# def query_rag_index(query: str):
#     # Load storage and index
#     storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
#     service_context = ServiceContext.from_defaults(embed_model=OpenAIEmbedding())

#     index = load_index_from_storage(storage_context, service_context=service_context)
#     retriever = VectorIndexRetriever(index=index, similarity_top_k=4)
#     engine = RetrieverQueryEngine(retriever=retriever, service_context=service_context)
    
#     response = engine.query(query)
#     return response
