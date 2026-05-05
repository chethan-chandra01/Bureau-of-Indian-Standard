"""
src/vector_store.py
-------------------
Builds and persists a FAISS vector index from the parsed BIS standard chunks.

Workflow:
  1. Load data/chunks.json  (produced by ingestion.py)
  2. Embed each chunk's embed_text using sentence-transformers
  3. Build a FAISS IndexFlatIP (inner-product / cosine after L2-norm)
  4. Save index  → vector_db/faiss_index.bin
  5. Save metadata → vector_db/metadata.json

The index and metadata are loaded at inference time by retriever.py.
"""

import json
import sys
import numpy as np
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Paths & Model Config
# ---------------------------------------------------------------------------
REPO_ROOT      = Path(__file__).resolve().parent.parent
CHUNKS_PATH    = REPO_ROOT / "data"      / "chunks.json"
VECTOR_DB_DIR  = REPO_ROOT / "vector_db"
INDEX_PATH     = VECTOR_DB_DIR / "faiss_index.bin"
METADATA_PATH  = VECTOR_DB_DIR / "metadata.json"

# Change this:
# EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# To this:
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM   = 384   
BATCH_SIZE      = 64    

# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def load_chunks(chunks_path: Path) -> list[dict]:
    if not chunks_path.exists():
        raise FileNotFoundError(f"chunks.json not found at {chunks_path}.")
    with open(chunks_path, encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"[vector_store] Loaded {len(chunks)} chunks from {chunks_path.name}")
    return chunks

def build_embeddings(texts: list[str], model_name: str = EMBEDDING_MODEL, batch_size: int = BATCH_SIZE) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise RuntimeError("sentence-transformers not installed.")

    print(f"[vector_store] Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    print(f"[vector_store] Embedding {len(texts)} chunks...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    embeddings = embeddings.astype(np.float32)
    return embeddings

def build_faiss_index(embeddings: np.ndarray):
    try:
        import faiss
    except ImportError:
        raise RuntimeError("faiss-cpu not installed.")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"[vector_store] FAISS index built: {index.ntotal} vectors, dim={dim}")
    return index

def save_index(index, index_path: Path) -> None:
    import faiss
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    print(f"[vector_store] Index saved → {index_path}")

def save_metadata(chunks: list[dict], metadata_path: Path) -> None:
    """Save metadata in index order, including the full text for generator context."""
    metadata = [
        {
            "is_number": c["is_number"],
            "title":     c["title"],
            "section":   c["section"],
            "scope":     c["scope"],
            "full_text": c.get("full_text", "") # <-- FIX: Added full_text
        }
        for c in chunks
    ]
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"[vector_store] Metadata saved → {metadata_path}")

def load_index_and_metadata(index_path: Optional[Path] = None, metadata_path: Optional[Path] = None):
    import faiss
    index_path    = index_path    or INDEX_PATH
    metadata_path = metadata_path or METADATA_PATH

    if not index_path.exists() or not metadata_path.exists():
        raise FileNotFoundError("FAISS artifacts not found. Run src/vector_store.py first.")

    index = faiss.read_index(str(index_path))
    with open(metadata_path, encoding="utf-8") as f:
        metadata = json.load(f)

    return index, metadata

def build_vector_store(chunks_path: Optional[Path] = None, index_path: Optional[Path] = None, metadata_path: Optional[Path] = None) -> None:
    chunks_path   = chunks_path   or CHUNKS_PATH
    index_path    = index_path    or INDEX_PATH
    metadata_path = metadata_path or METADATA_PATH

    chunks = load_chunks(chunks_path)
    texts = [c["embed_text"] for c in chunks]
    embeddings = build_embeddings(texts)
    index = build_faiss_index(embeddings)
    
    save_index(index, index_path)
    save_metadata(chunks, metadata_path)

if __name__ == "__main__":
    try:
        build_vector_store()
    except Exception as e:
        print(f"[vector_store] FATAL: {e}", file=sys.stderr)
        sys.exit(1)