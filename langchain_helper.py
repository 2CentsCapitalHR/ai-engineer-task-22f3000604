import os
import asyncio
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.chains import LLMChain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import UnstructuredURLLoader,UnstructuredFileLoader
import glob  
import textwrap  
from typing import List  
import streamlit as st
import re
import time
from urllib.parse import urljoin, urlparse
try:
    import requests
    from bs4 import BeautifulSoup
    HAVE_CRAWL_DEPS = True
except ImportError:
    HAVE_CRAWL_DEPS = False



#from pydantic import BaseModel
#import asyncio
#import random


load_dotenv()
GOOGLE_API_KEY = os.getenv("Google_api_key")

@st.cache_resource(show_spinner=False)
def get_embeddings():
    
    use_google = os.getenv("USE_GOOGLE_EMB") == "1"
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    if use_google:
        return GoogleGenerativeAIEmbeddings(google_api_key=GOOGLE_API_KEY, model="gemini-embedding-001")
    # Local BGE base (768-d) with cosine-friendly normalization
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

def multiple_data_sources(docx_files,urls):
    all_documents = []
    if docx_files:
        # Load .docx and .pdf files directly from knowledge_base
        for pattern in ("knowledge_base/*.docx", "knowledge_base/*.pdf"):
            for file in glob.glob(pattern):
                loader = UnstructuredFileLoader(file, mode="elements")
                documents = loader.load()
                all_documents.extend(documents)

    if urls:
        for url in urls:
            loader = UnstructuredURLLoader(urls=[url],mode="elements")
            documents = loader.load()
            all_documents.extend(documents)

    return all_documents    


def vector_db_of_docx_url(docx_files,urls) -> FAISS:
    documents = multiple_data_sources(docx_files, urls)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    if not docs:
        raise ValueError("No documents found. Add .docx files inside knowledge_base/ or provide URLs.")
    embeddings = get_embeddings()
    db = FAISS.from_documents(docs, embeddings)
    return db

def vector_db_from_documents(documents: List[Document]) -> FAISS:
    """Build a FAISS vector store from a provided list of Document objects.

    Used by the Streamlit app after a user uploads a file.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    if not docs:
        raise ValueError("Uploaded document produced no text chunks (file may be empty or unsupported).")
    embeddings = get_embeddings()
    return FAISS.from_documents(docs, embeddings)

# (No eager database build at import time; call vector_db_of_docx_url when needed from Streamlit.)
urls = [
    "https://www.adgm.com/registration-authority/registration-and-incorporation",
    "https://www.adgm.com/",
    "https://www.adgm.com/legal-framework/guidance-and-policy-statements",
    "https://www.adgm.com/operating-in-adgm/obligations-of-adgm-registered-entities/annual-filings/annual-accounts",
    "https://www.adgm.com/operating-in-adgm/post-registration-services/letters-and-permits",
]


def get_response_from_db(db, query, k=3):
    """Retrieve top-k chunks and get a concise grounded answer.

    Returns a short answer; if no relevant chunks, notifies the user.
    """
    docs = db.similarity_search(query, k=k)
    if not docs:
        return "No relevant content found for that question in the uploaded document."

    # Join with separator to preserve some structure
    docs_page_content = "\n---\n".join([d.page_content for d in docs])[:6000]  # safety truncate

    llm = ChatGoogleGenerativeAI(google_api_key=GOOGLE_API_KEY, model="gemini-2.0-flash")

    prompt_template = PromptTemplate(
        input_variables=["query", "context"],
        template=textwrap.dedent(
            """
            You are an ADGM document compliance assistant. Use ONLY the context snippets.
            If info is missing, state clearly what is missing; do not invent.
            Answer format:
            Answer: <concise answer (<=120 words)>
            Missing: <comma-separated missing items or 'None'>
            """
            "\nContext:\n{context}\n\nQuestion: {query}\n"""
        ),
    )

    chain = prompt_template | llm
    response = chain.invoke({"query": query, "context": docs_page_content})
    return response.content.strip()

# ---------------------------
# Simple ADGM site crawler
# ---------------------------

ALLOWED_DOMAIN = "adgm.com"

def crawl_adgm(seed_urls: List[str], max_pages: int = 8, max_depth: int = 1, delay: float = 0.5) -> List[str]:
    """Crawl ADGM site starting from seed URLs (shallow, polite, no JS rendering).

    Parameters:
        seed_urls: Starting URLs (must be full https URLs)
        max_pages: Maximum pages (including seeds) to fetch
        max_depth: Link depth (0 = only seeds)
        delay: sleep seconds between requests

    Returns:
        List of successfully fetched page URLs (unique, clipped to max_pages)
    """
    if not HAVE_CRAWL_DEPS:
        print("[crawl] Missing dependencies: install requests beautifulsoup4")
        return []
    visited = set()
    queue = [(u, 0) for u in seed_urls]
    results = []
    while queue and len(results) < max_pages:
        url, depth = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)
        parsed = urlparse(url)
        if parsed.netloc.endswith(ALLOWED_DOMAIN) is False:
            continue
        try:
            resp = requests.get(url, timeout=10, headers={"User-Agent": "MiniCrawler/0.1"})
            ctype = resp.headers.get("Content-Type", "")
            if resp.status_code != 200 or "text/html" not in ctype:
                continue
            html = resp.text
            results.append(url)
        except Exception as e:
            print(f"[crawl] fail {url}: {e}")
            continue
        if depth < max_depth:
            try:
                soup = BeautifulSoup(html, "html.parser")
                for a in soup.select('a[href]'):
                    href = a.get('href')
                    if not href:
                        continue
                    if href.startswith('#'):
                        continue
                    new_url = urljoin(url, href)
                    # Skip non-http(s)
                    if not new_url.startswith('http'):
                        continue
                    # Filter query-heavy or media links
                    if re.search(r'\.(pdf|jpg|png|gif|zip)$', new_url, re.IGNORECASE):
                        continue
                    if new_url not in visited and len(results) + len(queue) < max_pages + 10:  # small buffer
                        queue.append((new_url, depth + 1))
            except Exception as e:
                print(f"[crawl] parse fail {url}: {e}")
        time.sleep(delay)
    return results[:max_pages]

def vector_db_from_crawl(seed_urls: List[str], max_pages: int = 8, max_depth: int = 1) -> FAISS:
    """Crawl ADGM pages then build a vector DB from the fetched URLs."""
    urls = crawl_adgm(seed_urls, max_pages=max_pages, max_depth=max_depth)
    if not urls:
        raise ValueError("Crawler returned no URLs; check dependencies or seeds.")
    return vector_db_of_docx_url([], urls)