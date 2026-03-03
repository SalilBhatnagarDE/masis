"""
Document Ingestion Pipeline — The heart of document processing.

Pipeline: Load (LlamaParse v2, with Hybrid Local Pipeline (Docling + PaddleOCR-VL-1.5 / GLM-OCR)
          alternative) -> Parse -> Chunk (Hierarchical) -> Tag Metadata -> Summarize Tables
          -> Embed -> Store

Key features:
- LlamaParse v2 document parsing: PDF/DOCX/PPTX converted to clean Markdown with preserved
  table structure (uses llama_cloud>=1.0, AsyncLlamaCloud client)
- Hybrid Local Pipeline (Docling + PaddleOCR-VL-1.5 / GLM-OCR) alternative: best reported
  document parsing results on OmniDocBench v1.5, fast/lightweight, local or API
  (https://huggingface.co/zai-org/GLM-OCR)
- GPT-4.1 vision captioning: standalone images described semantically for vector search
- Hierarchical chunking: child chunks (512 tokens) for retrieval, parent chunks (2048 tokens)
  for context
- Table-aware parsing: tables kept as single Markdown chunks with generated summaries
- Metadata tagging: document-level (LLM) + chunk-level (keyword) metadata
- ChromaDB storage with parent-child linking

Migration note (v1 → v2):
  - Package:  `pip install llama_cloud>=1.0`  (replaces `llama_parse`)
  - Client:   AsyncLlamaCloud (replaces LlamaParse)
  - Flow:     files.create() → parsing.parse()  (two-step upload-then-parse)
  - Tiers:    "agentic_plus" ≈ old premium_mode=True / "agentic" ≈ old default
  - Results:  result.markdown.pages[n].markdown  (replaces flat string)
"""

import asyncio
import os
import re
import json
import uuid
import base64
import hashlib
import logging
from typing import Optional, Iterable

import chromadb
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_cloud import AsyncLlamaCloud          # NEW: replaces llama_parse
from openai import OpenAI

from config import *
from ingestion.table_summarizer import summarize_table
from ingestion.metadata_tagger import (
    extract_document_metadata,
    detect_section_from_text,
    detect_department_from_text,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TEXT_EXTS          = [".pdf", ".docx", ".pptx", ".txt", ".md"]
IMAGE_EXTS         = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]
ALL_SUPPORTED_EXTS = TEXT_EXTS + IMAGE_EXTS

# ---------------------------------------------------------------------------
# Step 0: Hash-Based Caching
# ---------------------------------------------------------------------------

def _compute_file_hash(filepath: str) -> str:
    """Compute SHA-256 hash of a file for change detection."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            sha256.update(block)
    return sha256.hexdigest()


def _compute_folder_hashes(doc_folder: str) -> dict[str, str]:
    """Compute hashes for all supported files in a document folder."""
    hashes: dict[str, str] = {}
    for root, _, files in os.walk(doc_folder):
        for name in sorted(files):
            ext = os.path.splitext(name)[1].lower()
            if ext in ALL_SUPPORTED_EXTS:
                filepath = os.path.join(root, name)
                rel_path = os.path.relpath(filepath, doc_folder)
                hashes[rel_path] = _compute_file_hash(filepath)
    return hashes


def _load_hash_manifest() -> dict[str, str]:
    """Load saved file hashes from the manifest file."""
    if os.path.exists(HASH_MANIFEST_PATH):
        with open(HASH_MANIFEST_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_hash_manifest(manifest: dict[str, str]) -> None:
    """Save file hashes to the manifest file."""
    os.makedirs(os.path.dirname(HASH_MANIFEST_PATH), exist_ok=True)
    with open(HASH_MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def _has_documents_changed(doc_folder: str) -> tuple[bool, dict[str, str]]:
    current_hashes = _compute_folder_hashes(doc_folder)
    saved_hashes   = _load_hash_manifest()

    new_files      = set(current_hashes) - set(saved_hashes)
    removed_files  = set(saved_hashes)   - set(current_hashes)
    modified_files = {
        f for f in current_hashes
        if f in saved_hashes and current_hashes[f] != saved_hashes[f]
    }

    # Detect renames: a "removed" + "new" file with the same hash = rename, not a change
    saved_hash_values   = set(saved_hashes.values())
    current_hash_values = set(current_hashes.values())

    genuine_new      = {f for f in new_files      if current_hashes[f] not in saved_hash_values}
    genuine_removed  = {f for f in removed_files  if saved_hashes[f]   not in current_hash_values}
    renames          = new_files - genuine_new  # same hash existed before under a different name

    if renames:
        logger.info("  Renamed files (no re-ingest needed): %s", renames)
    if genuine_new:
        logger.info("  New files:      %s", genuine_new)
    if genuine_removed:
        logger.info("  Removed files:  %s", genuine_removed)
    if modified_files:
        logger.info("  Modified files: %s", modified_files)

    changed = bool(genuine_new or genuine_removed or modified_files)
    return changed, current_hashes


# ---------------------------------------------------------------------------
# Step 1: Load Documents
# ---------------------------------------------------------------------------

def _iter_image_files(doc_folder: str) -> Iterable[str]:
    """Iterate over standalone image files in the document folder."""
    for root, _, files in os.walk(doc_folder):
        for name in files:
            if os.path.splitext(name)[1].lower() in IMAGE_EXTS:
                yield os.path.join(root, name)


def _caption_image_via_vision(image_path: str) -> str:
    """
    Generate a semantic caption for an image using GPT-4.1 vision.
    Handles charts, tables-as-images, diagrams, and photos.
    """
    if not OPENAI_API_KEY:
        logger.warning("OPENAI_API_KEY not set; skipping image captioning for %s", image_path)
        return ""

    try:
        from prompts import IMAGE_CAPTION_PROMPT

        client = OpenAI(api_key=OPENAI_API_KEY)
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        data_url = "data:image/png;base64," + base64.b64encode(image_bytes).decode("utf-8")

        response = client.chat.completions.create(
            model=INGESTION_LLM_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text",      "text": IMAGE_CAPTION_PROMPT},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
        )
        return response.choices[0].message.content.strip() if response.choices[0].message.content else ""
    except Exception as exc:
        logger.warning("Vision captioning failed for %s: %s", image_path, exc)
        return ""


def _load_standalone_images(doc_folder: str) -> list[Document]:
    """
    Create LlamaIndex Documents from standalone image files using GPT-4.1 vision captioning.
    Each image is captioned semantically (chart data extracted, tables converted to markdown,
    diagrams described) so it becomes searchable via the vector store.
    """
    documents: list[Document] = []
    for image_path in _iter_image_files(doc_folder):
        logger.info("  Captioning image: %s", os.path.basename(image_path))
        text = _caption_image_via_vision(image_path)
        if not text.strip():
            continue
        metadata = {
            "file_name":  os.path.basename(image_path),
            "page_label": 0,
            "source":     "image",
            "file_path":  image_path,
        }
        documents.append(Document(text=text, metadata=metadata))
    return documents


# --- LlamaParse v2 helpers ---------------------------------------------------

async def _parse_file_with_llamaparse_v2(
    client: AsyncLlamaCloud,
    filepath: str,
) -> list[Document]:
    """
    Upload a single file to LlamaParse v2 and return per-page LlamaIndex Documents.

    v2 flow:
        1. client.files.create()   — uploads the raw file bytes
        2. client.parsing.parse()  — submits a parse job and polls until done
        3. Iterate result.markdown.pages to build one Document per page.

    Tables are kept as Markdown pipe tables so the regex in extract_tables_from_text
    can detect and route them into the dual-embedding table retrieval pipeline —
    solving the key limitation of pypdf/SimpleDirectoryReader which flattens tables.

    Returns an empty list on failure (with logging) so the rest of the pipeline continues.
    """
    filename = os.path.basename(filepath)
    logger.info("  [LlamaParse v2] Uploading: %s", filename)

    try:
        # Step 1 — upload
        with open(filepath, "rb") as fh:
            file_obj = await client.files.create(file=(filename, fh), purpose="parse")

        logger.info("  [LlamaParse v2] Parsing (tier=%s): %s", LLAMAPARSE_TIER, filename)

        # Step 2 — parse (SDK polls until job completes)
        result = await client.parsing.parse(
            file_id=file_obj.id,
            tier=LLAMAPARSE_TIER,
            version="latest",
            # Keep tables as Markdown pipe tables for our regex extractor
            output_options={
                "markdown": {
                    "tables": {
                        "output_tables_as_markdown": True,
                    },
                },
                # Saving images for later retrieval
                "images_to_save": ["screenshot"],
            },
            processing_options={
                "ocr_parameters": {
                    "languages": ["en"],
                    # "cost_optimizer": {
                    #     "enable": True
                    # }
                },
            },
            expand=["markdown"],
        )

        # Step 3 — one Document per page
        documents: list[Document] = []
        for page in result.markdown.pages:
            page_text = page.markdown or ""
            if not page_text.strip():
                continue
            metadata = {
                "file_name":  filename,
                "file_path":  filepath,
                "page_label": page.page_number,
                "source":     "llamaparse_v2",
            }
            documents.append(Document(text=page_text, metadata=metadata))

        logger.info(
            "  [LlamaParse v2] %s → %d page(s) parsed", filename, len(documents)
        )
        return documents

    except Exception as exc:
        logger.error("[LlamaParse v2] Failed to parse %s: %s", filename, exc)
        return []


async def _parse_text_file_simple(filepath: str) -> list[Document]:
    """
    Fallback for plain .txt / .md files — read directly without an API call.
    """
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    if not text.strip():
        return []
    metadata = {
        "file_name":  os.path.basename(filepath),
        "file_path":  filepath,
        "page_label": 1,
        "source":     "simple_reader",
    }
    return [Document(text=text, metadata=metadata)]


async def _load_documents_async(doc_folder: str) -> list[Document]:
    """
    Walk doc_folder and parse all text-based files concurrently:
      - .pdf / .docx / .pptx  → LlamaParse v2 (upload + parse)
      - .txt / .md             → direct read (no API call needed)

    All files are processed in parallel via asyncio.gather().
    """
    llamaparse_exts = {".pdf", ".docx", ".pptx"}
    simple_exts     = {".txt", ".md"}

    file_paths: list[str] = []
    for root, _, files in os.walk(doc_folder):
        for name in files:
            ext = os.path.splitext(name)[1].lower()
            if ext in llamaparse_exts or ext in simple_exts:
                file_paths.append(os.path.join(root, name))

    if not file_paths:
        logger.warning("No text documents found in %s", doc_folder)
        return []

    # One shared async client for all uploads / parse jobs
    client = AsyncLlamaCloud(api_key=LLAMA_CLOUD_API_KEY)

    async def _process(filepath: str) -> list[Document]:
        ext = os.path.splitext(filepath)[1].lower()
        if ext in llamaparse_exts:
            return await _parse_file_with_llamaparse_v2(client, filepath)
        return await _parse_text_file_simple(filepath)

    results = await asyncio.gather(*(_process(fp) for fp in file_paths))

    documents: list[Document] = []
    for doc_list in results:
        documents.extend(doc_list)
    return documents


def load_documents(doc_folder: str) -> list[Document]:
    """
    Load all documents from *doc_folder* and return a flat list of LlamaIndex Documents.

    Parsing strategy
    ----------------
    - PDF / DOCX / PPTX → LlamaParse v2 (AsyncLlamaCloud, agentic_plus tier).
      Each page becomes a separate Document so page-level metadata is preserved.
      Tables are output as Markdown pipe tables for the dual-embedding retrieval pipeline.
    - TXT / MD          → direct file read (no API round-trip).
    - Standalone images → GPT-4.1 vision captioning (one Document per image).

    Alternative parser
    ------------------
    Hybrid Local Pipeline (Docling + PaddleOCR-VL-1.5 / GLM-OCR) — best on OmniDocBench
    v1.5, fast/lightweight, runs locally or via API (https://huggingface.co/zai-org/GLM-OCR).
    Swap _load_documents_async() for a Docling-based equivalent to use it.

    Args:
        doc_folder: Path to directory containing company documents.

    Returns:
        List of LlamaIndex Document objects ready for chunking and embedding.
    """
    if not os.path.exists(doc_folder):
        raise FileNotFoundError(f"Document folder not found: {doc_folder}")

    # Run async document loading, guarding against already-running event loops
    # (e.g. Jupyter notebooks)
    try:
        documents = asyncio.run(_load_documents_async(doc_folder))
    except RuntimeError:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, _load_documents_async(doc_folder))
            documents = future.result()

    logger.info("Loaded %d text document(s) from %s", len(documents), doc_folder)

    # Caption standalone images via GPT-4.1 vision
    image_docs = _load_standalone_images(doc_folder)
    if image_docs:
        logger.info("Captioned %d image(s) from %s", len(image_docs), doc_folder)
        documents.extend(image_docs)

    logger.info("Total: %d document(s) from %s", len(documents), doc_folder)

    print(f"Documents markdown === {documents}")
    return documents


# ---------------------------------------------------------------------------
# Step 2: Detect Tables
# ---------------------------------------------------------------------------

def is_table_content(text: str) -> bool:
    """
    Detect if a text chunk contains a Markdown table.

    Heuristic: If the text contains multiple lines with pipe characters (|)
    and at least one separator line (---|---), it's likely a table.
    """
    lines          = text.strip().split("\n")
    pipe_lines     = [line for line in lines if "|" in line]
    separator_lines = [line for line in lines if re.match(r'^[\s|:-]+$', line)]
    return len(pipe_lines) >= 3 and len(separator_lines) >= 1


def extract_tables_from_text(text: str) -> list[dict]:
    """
    Extract table blocks from text content.

    Returns list of dicts with:
        - 'table':  the raw Markdown table text
        - 'before': text before the table
        - 'after':  text after the table
        - 'start':  character offset of table start
        - 'end':    character offset of table end
    """
    table_pattern = re.compile(
        r'((?:\|[^\n]+\|\n)+(?:\|[-:\s|]+\|\n)(?:\|[^\n]+\|\n)*)',
        re.MULTILINE
    )

    tables    = []
    last_end  = 0

    for match in table_pattern.finditer(text):
        start, end = match.span()
        tables.append({
            "table":  match.group(0).strip(),
            "before": text[last_end:start].strip(),
            "start":  start,
            "end":    end,
        })
        last_end = end

    return tables


# ---------------------------------------------------------------------------
# Step 3: Hierarchical Chunking
# ---------------------------------------------------------------------------

def create_hierarchical_chunks(text: str, source_file: str, page_num: int = 0) -> list[dict]:
    """
    Create parent and child chunks with linking.

    Child chunks (512 tokens) are embedded in the vector store.
    Parent chunks (2048 tokens) are stored separately and returned as LLM context.
    Each child carries a parent_id reference.

    Args:
        text: The text content to chunk.
        source_file: Source filename for metadata.
        page_num: Page number for metadata.

    Returns:
        List of chunk dicts with keys: id, text, parent_id, parent_text,
        content_type, source_file, page_number
    """
    if not text.strip():
        return []

    parent_splitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE_PARENT,
        chunk_overlap=CHUNK_OVERLAP * 2,
    )
    child_splitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE_CHILD,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = []

    for parent_text in parent_splitter.split_text(text):
        parent_id    = str(uuid.uuid4())
        child_texts  = child_splitter.split_text(parent_text)

        for child_text in child_texts:
            chunks.append({
                "id":           str(uuid.uuid4()),
                "text":         child_text,
                "parent_id":    parent_id,
                "parent_text":  parent_text,
                "content_type": "text",
                "source_file":  source_file,
                "page_number":  page_num,
            })

    return chunks


# ---------------------------------------------------------------------------
# Step 4: Process Tables
# ---------------------------------------------------------------------------

def process_table_chunk(table_markdown: str, source_file: str, page_num: int = 0) -> dict:
    """
    Process a single table: generate a summary for embedding, keep raw table as context.

    The SUMMARY is what gets embedded (for retrieval).
    The RAW TABLE is what gets returned to the LLM (for accurate answers).

    Args:
        table_markdown: The table in Markdown format.
        source_file: Source filename.
        page_num: Page number.

    Returns:
        Chunk dict with summary as 'text' (for embedding) and raw table as 'parent_text'
        (for context).
    """
    try:
        summary = summarize_table(table_markdown)
    except Exception as e:
        logger.warning("Failed to summarize table, using raw table: %s", e)
        summary = table_markdown

    return {
        "id":           str(uuid.uuid4()),
        "text":         summary,          # embedded
        "parent_id":    str(uuid.uuid4()),
        "parent_text":  table_markdown,   # returned as LLM context
        "content_type": "table",
        "source_file":  source_file,
        "page_number":  page_num,
    }


# ---------------------------------------------------------------------------
# Step 5: Full Ingestion Pipeline
# ---------------------------------------------------------------------------

def process_single_document(doc: Document, doc_metadata: dict) -> list[dict]:
    """
    Process a single document into chunks with metadata.

    Args:
        doc: LlamaIndex Document object.
        doc_metadata: Document-level metadata dict.

    Returns:
        List of chunk dicts ready for storage.
    """
    text        = doc.text
    source_file = doc.metadata.get("file_name", "unknown")
    page_num    = doc.metadata.get("page_label", 0)

    try:
        page_num = int(page_num)
    except (ValueError, TypeError):
        page_num = 0

    all_chunks = []
    tables     = extract_tables_from_text(text)

    if tables:
        last_end = 0

        for table_info in tables:
            # Text before this table
            text_before = text[last_end:table_info["start"]].strip()
            if text_before:
                all_chunks.extend(
                    create_hierarchical_chunks(text_before, source_file, page_num)
                )

            # The table itself
            all_chunks.append(
                process_table_chunk(table_info["table"], source_file, page_num)
            )
            last_end = table_info["end"]

        # Text after the last table
        remaining = text[last_end:].strip()
        if remaining:
            all_chunks.extend(
                create_hierarchical_chunks(remaining, source_file, page_num)
            )
    else:
        all_chunks.extend(
            create_hierarchical_chunks(text, source_file, page_num)
        )

    # Attach document-level + chunk-level metadata
    for chunk in all_chunks:
        chunk["document_type"] = doc_metadata.get("document_type", "other")
        chunk["year"]          = doc_metadata.get("year", "")
        chunk["quarter"]       = doc_metadata.get("quarter", "")
        chunk["section"]       = detect_section_from_text(chunk["text"])
        chunk["department"]    = detect_department_from_text(chunk["text"])

    return all_chunks


def store_in_chromadb(chunks: list[dict]) -> dict:
    """
    Store chunks in ChromaDB with embeddings and metadata.
    Also persist parent chunks to a JSON store on disk.

    Args:
        chunks: List of chunk dicts from the processing pipeline.

    Returns:
        Dict with stats: collection_name, num_chunks, num_parents
    """
    if not chunks:
        logger.warning("No chunks to store!")
        return {
            "collection_name": CHROMA_COLLECTION,
            "num_chunks":      0,
            "num_parents":     0,
        }

    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

    # Fresh ingest — drop existing collection
    try:
        client.delete_collection(CHROMA_COLLECTION)
        logger.info("Deleted existing collection: %s", CHROMA_COLLECTION)
    except Exception:
        pass

    from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

    embedding_fn = OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name=EMBEDDING_MODEL,
    )

    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        embedding_function=embedding_fn,
    )

    ids          = []
    documents    = []
    metadatas    = []
    parent_store = {}   # parent_id -> parent_text

    for chunk in chunks:
        ids.append(chunk["id"])
        documents.append(chunk["text"])
        parent_store[chunk["parent_id"]] = chunk["parent_text"]

        metadatas.append({
            "parent_id":     chunk["parent_id"],
            "content_type":  chunk.get("content_type", "text"),
            "source_file":   chunk.get("source_file", "unknown"),
            "page_number":   chunk.get("page_number", 0),
            "document_type": chunk.get("document_type", "other"),
            "year":          str(chunk.get("year", "")),
            "quarter":       str(chunk.get("quarter", "")),
            "section":       chunk.get("section", "General"),
            "department":    chunk.get("department", "Company-wide"),
        })

    # Add in batches to respect embedding token limits
    batch_size = 100
    for i in range(0, len(ids), batch_size):
        end = min(i + batch_size, len(ids))
        collection.add(
            ids=ids[i:end],
            documents=documents[i:end],
            metadatas=metadatas[i:end],
        )
        logger.info("Added batch %d: %d chunks", i // batch_size + 1, end - i)

    # Persist parent store to disk
    os.makedirs(os.path.dirname(PARENT_STORE_PATH), exist_ok=True)
    with open(PARENT_STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(parent_store, f, indent=2)

    logger.info("Stored %d chunks and %d parent chunks", len(ids), len(parent_store))

    return {
        "collection_name": CHROMA_COLLECTION,
        "num_chunks":      len(ids),
        "num_parents":     len(parent_store),
    }


def ingest_pipeline(doc_folder: Optional[str] = None, force: bool = False) -> dict:
    """
    Full end-to-end ingestion pipeline with hash-based caching.

    Computes SHA-256 hashes of all document files and compares with the saved manifest.
    If no files have changed, ingestion is skipped entirely — saving API calls and time.

    Steps
    -----
    0. Check file hashes (skip if unchanged, unless force=True)
    1. Load documents (LlamaParse v2 for PDF/DOCX/PPTX, direct read for TXT/MD,
       GPT-4.1 vision for standalone images)
    2. Extract document-level metadata via LLM (type, year, quarter)
    3. Chunk with hierarchical strategy + table awareness
    4. Tag chunk-level metadata (section, department)
    5. Summarize tables for dual-embedding retrieval
    6. Embed and store in ChromaDB (child chunks) + JSON (parent chunks)
    7. Save hash manifest

    Args:
        doc_folder: Path to document folder. Defaults to COMPANY_DOCS_DIR.
        force: If True, bypass hash cache and re-ingest everything.

    Returns:
        Dict with ingestion stats.
    """
    doc_folder = doc_folder or COMPANY_DOCS_DIR

    logger.info("=== Starting Ingestion Pipeline ===")
    logger.info("Document folder: %s", doc_folder)

    # Step 0: Hash-based caching check
    if not force:
        changed, current_hashes = _has_documents_changed(doc_folder)
        if not changed:
            _save_hash_manifest(current_hashes)  # Update manifest even if no changes, to capture renames
            logger.info("No document changes detected — skipping ingestion")
            return {
                "num_documents": 0,
                "num_files":     0,
                "num_chunks":    0,
                "num_parents":   0,
                "skipped":       True,
                "reason":        "No changes detected (use --force to re-ingest)",
            }
        logger.info("Document changes detected — running full ingestion")
    else:
        logger.info("Force flag set — bypassing cache")
        current_hashes = _compute_folder_hashes(doc_folder)

    # Step 1: Load documents
    documents = load_documents(doc_folder)
    if not documents:
        logger.warning("No documents found!")
        return {"num_documents": 0, "num_chunks": 0}

    # Step 2: Extract document-level metadata; group docs by source file first
    docs_by_file: dict[str, list[Document]] = {}
    for doc in documents:
        file_name = doc.metadata.get("file_name", "unknown")
        docs_by_file.setdefault(file_name, []).append(doc)

    all_chunks: list[dict] = []

    for file_name, file_docs in docs_by_file.items():
        logger.info("Processing: %s (%d page(s))", file_name, len(file_docs))

        # Use first ~3 pages for document-level metadata extraction
        first_pages_text = "\n\n".join(doc.text for doc in file_docs[:3])[:5000]

        try:
            doc_metadata = extract_document_metadata(first_pages_text)
            logger.info(
                "  Metadata: type=%s, year=%s, quarter=%s",
                doc_metadata.get("document_type"),
                doc_metadata.get("year"),
                doc_metadata.get("quarter"),
            )
        except Exception as e:
            logger.warning("  Failed to extract metadata for %s: %s", file_name, e)
            doc_metadata = {"document_type": "other", "year": "", "quarter": ""}

        # Steps 3-5: Process each page into chunks
        for doc in file_docs:
            all_chunks.extend(process_single_document(doc, doc_metadata))

    logger.info(
        "Created %d total chunks from %d document(s)", len(all_chunks), len(documents)
    )

    # Step 6: Embed and store
    stats = store_in_chromadb(all_chunks)

    # Step 7: Save hash manifest only after successful storage
    _save_hash_manifest(current_hashes)
    logger.info("Saved file hash manifest (%d file(s))", len(current_hashes))

    logger.info("=== Ingestion Complete ===")
    logger.info(
        "Documents: %d  |  Chunks: %d  |  Parents: %d",
        len(documents),
        stats["num_chunks"],
        stats["num_parents"],
    )

    return {
        "num_documents": len(documents),
        "num_files":     len(docs_by_file),
        "skipped":       False,
        **stats,
    }


if __name__ == "__main__":
    result = ingest_pipeline()
    print(f"\nIngestion result: {json.dumps(result, indent=2)}")