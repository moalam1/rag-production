"""
pipeline/ingester.py — PDF ingestion and indexing pipeline.
Parses PDFs → semantic chunks → Pinecone upsert.
"""
import os
import shutil
import logging
import tempfile
from pathlib import Path

from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader, Document
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone
from huggingface_hub import HfApi

from config import settings

log = logging.getLogger(__name__)

_pc    = Pinecone(api_key=settings.PINECONE_API_KEY)
_index = _pc.Index(settings.PINECONE_INDEX)
_hf    = HfApi(token=settings.HF_TOKEN)

LOCAL_PDF_DIR = Path(settings.LOCAL_PDF_DIR)
LOCAL_PDF_DIR.mkdir(parents=True, exist_ok=True)


def pdf_url(filename: str, page) -> str:
    base = (
        f"https://huggingface.co/spaces/{settings.HF_REPO_ID}"
        f"/resolve/main/{settings.PDF_DIR}/{filename}"
    )
    try:
        return f"{base}#page={int(page)}"
    except (ValueError, TypeError):
        return base


def _upload_to_hf(local_path: Path, filename: str) -> bool:
    try:
        _hf.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=f"{settings.PDF_DIR}/{filename}",
            repo_id=settings.HF_REPO_ID,
            repo_type=settings.HF_REPO_TYPE,
        )
        return True
    except Exception as e:
        log.warning("HF upload failed for %s: %s", filename, e)
        return False


def ingest_files(file_paths: list[str]) -> list[str]:
    """
    Full ingestion pipeline for a list of PDF file paths.
    Returns a log of what happened.
    """
    logs = []
    tmp_dir = tempfile.mkdtemp()

    try:
        # 1. Copy files to temp dir + upload to HF
        for path in file_paths:
            filename   = os.path.basename(path)
            dest       = os.path.join(tmp_dir, filename)
            local_dest = LOCAL_PDF_DIR / filename
            shutil.copy(path, dest)
            shutil.copy(path, local_dest)

            uploaded = _upload_to_hf(local_dest, filename)
            logs.append(f"{'☁️  Uploaded' if uploaded else '⚠️  HF upload skipped'}: {filename}")

        # 2. Parse with LlamaParse
        logs.append("🔍 Parsing PDFs with LlamaParse...")
        parser = LlamaParse(
            api_key=settings.LLAMA_CLOUD_API_KEY,
            result_type="markdown",
            verbose=False,
            language="en",
            parsing_instructions="Extract headings and tables accurately.",
        )
        reader  = SimpleDirectoryReader(tmp_dir, file_extractor={".pdf": parser})
        raw_docs = reader.load_data()
        logs.append(f"✅ Parsed {len(raw_docs)} document(s).")

        # 3. Build LlamaIndex Documents with rich metadata
        documents = []
        for i, doc in enumerate(raw_docs):
            filename   = os.path.basename(doc.metadata.get("file_name", f"doc_{i}.pdf"))
            clean_name = os.path.splitext(filename)[0].replace("_", " ").replace("-", " ").strip()
            clean_name = " ".join(w.capitalize() for w in clean_name.split())
            page       = doc.metadata.get("page_label", doc.metadata.get("page", str(i + 1)))

            documents.append(Document(
                text=doc.text,
                metadata={
                    "filename":   filename,
                    "clean_name": clean_name,
                    "page":       str(page),
                    "pdf_url":    pdf_url(filename, page),
                }
            ))

        # 4. Semantic chunking
        logs.append("✂️ Chunking semantically...")
        embed_model = OpenAIEmbedding(
            model=settings.EMBED_MODEL,
            api_key=settings.OPENAI_API_KEY,
            dimensions=settings.EMBED_DIMS,
        )
        splitter = SemanticSplitterNodeParser(
            buffer_size=1,
            breakpoint_percentile_threshold=80,
            embed_model=embed_model,
        )
        nodes = splitter.get_nodes_from_documents(documents)
        logs.append(f"✂️ Created {len(nodes)} semantic chunks.")

        # 5. Upsert to Pinecone
        logs.append("📤 Uploading to Pinecone...")
        vector_store    = PineconeVectorStore(pinecone_index=_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        VectorStoreIndex(nodes, storage_context=storage_context, embed_model=embed_model)
        logs.append("✅ Indexing complete!")

    except Exception as e:
        logs.append(f"❌ Error: {e}")
        log.exception("Ingestion failed")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return logs
