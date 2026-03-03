"""
Configuration for the AI Leadership Insight Agent.

Centralizes all settings: API keys, model names, chunking params,
retrieval thresholds, and storage paths. Loaded from .env at startup.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# -- API Keys --
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY", "")

# -- LLM Settings --
LLM_MODEL = "gpt-4.1"             # Or use gpt-4.1-mini for Fast model for real-time query pipeline
INGESTION_LLM_MODEL = "gpt-4.1"        # High-quality vision model for image captioning
EMBEDDING_MODEL = "text-embedding-3-small"

# -- Document Chunking --
CHUNK_SIZE_PARENT = 2000
CHUNK_SIZE_CHILD = 500
CHUNK_OVERLAP = 50

# -- Retrieval Settings --
TOP_K_RETRIEVAL = 10           # How many chunks to fetch from hybrid search
RERANK_TOP_N = 5               # How many to keep after cross-encoder reranking
BM25_WEIGHT = 0.3              # Weight for BM25 in the RRF fusion
VECTOR_WEIGHT = 0.7            # Weight for vector search in the RRF fusion
RRF_K = 60                     # RRF constant (standard default)

# LlamaParse v2 tier to use for rich documents (tables / complex layouts).
# "agentic_plus" 
# "agentic"        = good quality + faster
# "cost_effective" = fast, plain-text heavy docs only
LLAMAPARSE_TIER = "agentic"

# -- Metadata Filtering --
METADATA_FILTER_KEYS = ["year", "quarter", "department", "topic"]
METADATA_PROGRESSIVE_FALLBACK = True   # Drop filters one by one if no results

# -- CRAG (Corrective RAG) --
MAX_DOC_GRADING_RETRIES = 3    # How many rewrite-then-retrieve cycles
DOCS_RELEVANCE_THRESHOLD = 1   # Minimum relevant docs to proceed to generation

# -- Self-RAG --
MAX_HALLUCINATION_RETRIES = 3  # How many regeneration attempts on hallucination

# -- Storage Paths --
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")

# -- Multi-Company Support --
# Auto-discover company folders: any directory named *_company_docs is available
_PROJECT_ROOT = os.path.dirname(__file__)
AVAILABLE_COMPANIES = {}
for _entry in os.listdir(_PROJECT_ROOT):
    if _entry.endswith("_company_docs") and os.path.isdir(os.path.join(_PROJECT_ROOT, _entry)):
        _company_name = _entry.replace("_company_docs", "")
        AVAILABLE_COMPANIES[_company_name] = _entry

DEFAULT_COMPANY = "infosys"  # Default matches README output screenshots

# Active company (set via set_active_company())
_ACTIVE_COMPANY = DEFAULT_COMPANY
COMPANY_DOCS_DIR = os.path.join(_PROJECT_ROOT, AVAILABLE_COMPANIES.get(DEFAULT_COMPANY, "infosys_company_docs"))
CHROMA_COLLECTION = f"leadership_docs_{DEFAULT_COMPANY}"
PARENT_STORE_PATH = os.path.join(CHROMA_PERSIST_DIR, f"parent_store_{DEFAULT_COMPANY}.json")
HASH_MANIFEST_PATH = os.path.join(CHROMA_PERSIST_DIR, f"file_hashes_{DEFAULT_COMPANY}.json")


def set_active_company(company: str):
    """Switch all paths to the given company. Call before ingestion or querying."""
    global _ACTIVE_COMPANY, COMPANY_DOCS_DIR, CHROMA_COLLECTION, PARENT_STORE_PATH, HASH_MANIFEST_PATH
    company = company.lower().strip()
    if company not in AVAILABLE_COMPANIES:
        raise ValueError(f"Unknown company '{company}'. Available: {list(AVAILABLE_COMPANIES.keys())}")
    _ACTIVE_COMPANY = company
    COMPANY_DOCS_DIR = os.path.join(_PROJECT_ROOT, AVAILABLE_COMPANIES[company])
    CHROMA_COLLECTION = f"leadership_docs_{company}"
    PARENT_STORE_PATH = os.path.join(CHROMA_PERSIST_DIR, f"parent_store_{company}.json")
    HASH_MANIFEST_PATH = os.path.join(CHROMA_PERSIST_DIR, f"file_hashes_{company}.json")

# -- Cross-Encoder Model --
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# -- Metadata Fields --
METADATA_FIELDS = {
    "document_type": ["annual_report", "quarterly_report", "strategy_note",
                      "operational_update", "other"],
    "departments": ["finance", "hr", "operations", "technology", "marketing",
                    "sales", "legal", "r&d", "supply_chain"],
}
