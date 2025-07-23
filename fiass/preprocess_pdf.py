import os
import shutil
from pathlib import Path

from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
import logging

BASE_DIR   = Path(__file__).resolve().parent
INDEX_PATH = BASE_DIR / "vector_db"
DATA_DIR   = BASE_DIR / "data"
print(BASE_DIR)
print(INDEX_PATH)
print(DATA_DIR)

def vectorize_pdf():
    
    load_dotenv()
    preprocessed_directory = os.path.join("data/")
    files = [os.path.join(preprocessed_directory, f) for f in os.listdir(preprocessed_directory) if f.endswith(".pdf")]

    docs = []

    for file in files:
        with open(file, "rb") as f:
            reader = PdfReader(f)
            text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
            
            # Create Document with metadata (optional: include filename, etc.)
            doc = Document(page_content=text, metadata={"source": os.path.basename(file)})
            docs.append(doc)

    INDEX_PATH = Path("vector_db")

    # Embeddings
    #embeddings = HuggingFaceEmbeddings(model=EMBED_MODEL, show_progress=True)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key="AIzaSyCqsDNnpIT1fXj-ksaFf90_0A1BSL8hu94")

    vector_store = FAISS.from_documents(docs, embeddings, normalize_L2=True)
    
    # ---- move underlying index to GPU ----
    res = FAISS.StandardGpuResources()
    gpu_index = FAISS.index_cpu_to_gpu(res, 0, vector_store.index)
    vector_store.index = gpu_index        # swap in the GPU handle

    vector_store.index = FAISS.index_gpu_to_cpu(vector_store.index)
    vector_store.save_local(INDEX_PATH)
    print("end")

def search_similarity(query):
    vectorize_pdf()
    #INDEX_PATH = "fiass/vector_db"

    # Recreate the embeddings object
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key="AIzaSyCqsDNnpIT1fXj-ksaFf90_0A1BSL8hu94")

    # Load the persisted vector store
    vector_store = FAISS.load_local(INDEX_PATH, embeddings, normalize_L2=True, allow_dangerous_deserialization=True)

    # move to GPU once, cache the handle
    res = FAISS.StandardGpuResources()
    vector_store.index = FAISS.index_cpu_to_gpu(res, 0, vector_store.index)

    # Now do a similarity search!
    query_vector = embeddings.embed_query(query)
    raw_results = vector_store.similarity_search_with_score_by_vector(query_vector, k=100)

    results = []

    for doc, score in raw_results:
        # if score <= 0.7:  # "good" threshold, tune as needed
        #     print(f"Score: {score:.3f} [GOOD]")
        # elif 0.7 <= score <= 1.0:
        #     print(f"Score: {score:.3f} [WEAK]")
        # else:
        #     print(f"Score: {score:.3f} [VERY WEAK]")
        # print(f"Source: {doc.metadata.get('source', 'N/A')}")
        # print(f"Snippet: {doc.page_content[:200]}")
        # print("---")
        if score < 1.0:
            results.append({
                "source": doc.metadata.get('source', 'N/A'),
                "score": float(round(score or 0, 3)),
                "content": doc.page_content.strip()
                })
    logging.info(results)
    return results
