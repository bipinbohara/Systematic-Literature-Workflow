import os
print("PWD:", os.getcwd())
print("Listing files:")
print(os.listdir("."))

from typing import Any, List, Dict, Optional, Union
import asyncio
import logging
from mcp.server.fastmcp import FastMCP
from pubmed_web_search import search_key_words, search_advanced, get_pubmed_metadata, download_full_text_pdf, deep_paper_analysis
from fiass.preprocess_pdf import vectorize_pdf, search_similarity

# RAG Pipeline
# try:
#     from pubmed_rag_pipeline import PDF_DIR, pdf_to_text, save_text, build_vector_index, query_rag_index
# except Exception as e:
#     print("❌ Failed to import RAG pipeline:", e)
#     raise e
#from pubmed_rag_pipeline import pdf_to_text
import shutil
from pubmed_rag_fetch import get_rag_data

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize FastMCP server
mcp = FastMCP("pubmed")

#@mcp.tool()
#async def search_pubmed_key_words2(key_words: str, num_results: int = 10) -> List[Dict[str, Any]]:
#    logging.info(f"Searching for articles with key words: {key_words}, num_results: {num_results}")
#    """
#    Search for articles on PubMed using key words.
#
#    Args:
#        key_words: Search query string
#        num_results: Number of results to return (default: 10)
#
#    Returns:
#        List of dictionaries containing article information
#    """
#    try:
#        results = await asyncio.to_thread(search_key_words, key_words, num_results)
#        return results
#    except Exception as e:
#        return [{"error": f"An error occurred while searching: {str(e)}"}]
    
@mcp.tool()
async def search_pubmed_key_words(key_words: str, num_results: int = 10) -> List[Dict[str, Any]]:
    logging.info(f"Searching for articles with key words: {key_words}, num_results: {num_results}")
    """
    Search for articles on PubMed using key words.

    Args:
        key_words: Search query string
        num_results: Number of results to return (default: 10)

    Returns:
        List of dictionaries containing article information
    """
    try:
        #results = await asyncio.to_thread(search_key_words, key_words, num_results)
        logging.info(f"Key word: {key_words}")
        results = await asyncio.to_thread(search_similarity, key_words)
        logging.info(f"Results: {results}")
        return results
    except Exception as e:
        return [{"error": f"An error occurred while searching: {str(e)}"}]

@mcp.tool()
async def search_pubmed_advanced(
    term: Optional[str] = None,
    title: Optional[str] = None,
    author: Optional[str] = None,
    journal: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    num_results: int = 10
) -> List[Dict[str, Any]]:
    logging.info(f"Performing advanced search with parameters: {locals()}")
    """
    Perform an advanced search for articles on PubMed.

    Args:
        term: General search term
        title: Search in title
        author: Author name
        journal: Journal name
        start_date: Start date for search range (format: YYYY/MM/DD)
        end_date: End date for search range (format: YYYY/MM/DD)
        num_results: Number of results to return (default: 10)

    Returns:
        List of dictionaries containing article information
    """
    try:
        results = await asyncio.to_thread(
            search_advanced,
            term, title, author, journal, start_date, end_date, num_results
        )
        return results
    except Exception as e:
        return [{"error": f"An error occurred while performing advanced search: {str(e)}"}]

@mcp.tool()
async def get_pubmed_article_metadata(pmid: Union[str, int]) -> Dict[str, Any]:
    logging.info(f"Fetching metadata for PMID: {pmid}")
    """
    Fetch metadata for a PubMed article using its PMID.

    Args:
        pmid: PMID of the article (can be string or integer)

    Returns:
        Dictionary containing article metadata
    """
    try:
        pmid_str = str(pmid)
        metadata = await asyncio.to_thread(get_pubmed_metadata, pmid_str)
        return metadata if metadata else {"error": f"No metadata found for PMID: {pmid_str}"}
    except Exception as e:
        return {"error": f"An error occurred while fetching metadata: {str(e)}"}

@mcp.tool()
async def download_pubmed_pdf(pmid: Union[str, int]) -> str:
    logging.info(f"Attempting to download PDF for PMID: {pmid}")
    """
    Attempt to download the full text PDF for a PubMed article.

    Args:
        pmid: PMID of the article (can be string or integer)

    Returns:
        String indicating the result of the download attempt
    """
    try:
        pmid_str = str(pmid)
        result = await asyncio.to_thread(download_full_text_pdf, pmid_str)
        #return result
        pdf_path = os.path.join(PDF_DIR, f"PMID_{pmid_str}.pdf")
        if os.path.exists(pdf_path):
            return f"PDF downloaded and saved: {pdf_path}"
        else:
            return f"PDF not found after download attempt: {pdf_path}"
    except Exception as e:
        return f"An error occurred while attempting to download the PDF: {str(e)}"

@mcp.prompt()
async def deep_paper_analysis(pmid: Union[str, int]) -> Dict[str, str]:
    logging.info(f"Performing deep paper analysis for PMID: {pmid}")
    """
    Perform a comprehensive analysis of a PubMed article.

    Args:
        pmid: PMID of the article

    Returns:
        Dictionary containing the comprehensive analysis structure
    """
    try:
        pmid_str = str(pmid)
        metadata = await asyncio.to_thread(get_pubmed_metadata, pmid_str)
        if not metadata:
            return {"error": f"No metadata found for PMID: {pmid_str}"}
            
        # 使用导入的 deep_paper_analysis 函数生成分析提示
        # 为避免递归调用，我们需要明确指定导入的函数
        from pubmed_web_search import deep_paper_analysis as web_deep_paper_analysis
        analysis_prompt = await asyncio.to_thread(web_deep_paper_analysis, metadata)
        
        # 返回包含分析提示的字典
        return {"analysis_prompt": analysis_prompt}
    except Exception as e:
        return {"error": f"An error occurred while performing the deep paper analysis: {str(e)}"}

# @mcp.tool(name="search_and_vectorize", description="Search PubMed, download papers, vectorize PDFs, and return top matching result using RAG")
# async def search_and_vectorize(query: str, num_results: int = 5) -> Dict[str, str]:
#     logging.info(f"Running full RAG pipeline for query: {query}")
    
#     # # 1. Clear old data (optional for freshness)
#     # for folder in [PDF_DIR, "storage/docs", "storage/index"]:
#     #     shutil.rmtree(folder, ignore_errors=True)
#     #     os.makedirs(folder, exist_ok=True)
    
#     # 1. (Skip clearing old data — now accumulating)
#     os.makedirs(PDF_DIR, exist_ok=True)
#     os.makedirs("storage/docs", exist_ok=True)
#     os.makedirs("storage/index", exist_ok=True)

#     # 2. Search articles
#     articles = await search_pubmed_key_words(query, num_results)
#     if not articles:
#         return {"error": "No articles found."}

#     # 3. Download + Extract
#     for article in articles:
#         pmid = article["PMID"]
#         await download_pubmed_pdf(pmid)
#         pdf_path = os.path.join(PDF_DIR, f"PMID_{pmid}.pdf")
#         if os.path.exists(pdf_path):
#             logging.info(f"PDF saved: {pdf_path}")
#             text = pdf_to_text(pdf_path)
#             if text.strip():
#                 save_text(pmid, text)
#         else:
#             logging.warning(f"PDF NOT FOUND: {pdf_path}")

#     # 4. Vectorize
#     build_vector_index()

#     # 5. Retrieve
#     response = query_rag_index(query)
#     return {"top_match": str(response)}

@mcp.tool(name="search_and_vectorize", description="Search PubMed, download papers, vectorize PDFs, and return top matching result using RAG")
async def search_and_vectorize(query: str, num_results: int = 5) -> Dict[str, str]:
    logging.info(f"Running full RAG pipeline for query: {query}")
    
    # Ensure directories exist (accumulating)
    os.makedirs(PDF_DIR, exist_ok=True)
    os.makedirs("storage/docs", exist_ok=True)
    os.makedirs("storage/index", exist_ok=True)

    # Step 1: Search articles
    articles = await search_pubmed_key_words(query, num_results)
    if not articles:
        return {"error": "No articles found."}

    # Step 2: Download PDFs and extract text
    for article in articles:
        pmid = article["PMID"]
        logging.info(f"Trying to download PDF for PMID: {pmid}")
        await download_pubmed_pdf(pmid)

        pdf_path = os.path.join(PDF_DIR, f"PMID_{pmid}.pdf")
        if os.path.exists(pdf_path):
            logging.info(f"PDF saved: {pdf_path}")
            try:
                text = pdf_to_text(pdf_path)
                if text.strip():
                    save_text(pmid, text)
                    logging.info(f"Text extracted and saved for PMID: {pmid}")
                else:
                    logging.warning(f"No text extracted from PDF: {pdf_path}")
            except Exception as e:
                logging.error(f"Failed to extract text from {pdf_path}: {e}")
        else:
            logging.warning(f"PDF NOT FOUND: {pdf_path}")

    # Step 3: Check if we have any documents to index
    doc_files = os.listdir("storage/docs")
    logging.info(f"Documents available for indexing: {doc_files}")
    if not doc_files:
        raise ValueError("No files found in storage/docs. PDF download or text extraction might have failed.")

    # Step 4: Build vector index
    build_vector_index()

    # Step 5: Query the RAG index
    try:
        response = query_rag_index(query)
        return {"top_match": str(response)}
    except Exception as e:
        logging.error(f"Error during RAG retrieval: {e}")
        return {"error": f"Failed to retrieve answer from RAG index: {e}"}

@mcp.tool(name="query_existing_pdfs", description="Process existing PDFs, vectorize, and return top-k relevant documents")
async def query_existing_pdfs(query: str, top_k: int = 100) -> Dict[str, Any]:
    logging.info(f"Querying existing PDFs with top_k={top_k}")

    # Step 1: Convert all PDFs to text
    for fname in os.listdir(PDF_DIR):
        if fname.endswith(".pdf"):
            pmid = fname.replace("PMID_", "").replace(".pdf", "")
            pdf_path = os.path.join(PDF_DIR, fname)
            txt_path = os.path.join(TXT_DIR, f"{pmid}.txt")

            if not os.path.exists(txt_path):
                logging.info(f"Extracting text from {fname}")
                text = pdf_to_text(pdf_path)
                if text.strip():
                    save_text(pmid, text)
                else:
                    logging.warning(f"No text extracted from {fname}")

    # Step 2: Check .txt files
    if not os.listdir(TXT_DIR):
        return {"error": "No text files found. Cannot build vector index."}

    # Step 3: Build vector index
    try:
        build_vector_index()
    except Exception as e:
        return {"error": f"Failed to build index: {e}"}

    # Step 4: Query index
    try:
        response = query_rag_index(query, top_k=top_k)
        return {"query": query, "top_results": str(response)}
    except Exception as e:
        return {"error": f"RAG query failed: {e}"}

if __name__ == "__main__":
    logging.info("Starting PubMed MCP server")
    # Initialize and run the server
    mcp.run(transport='stdio')

