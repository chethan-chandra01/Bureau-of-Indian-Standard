"""
src/retriever.py
----------------
Core retrieval logic for the BIS RAG pipeline.

Given a natural language query, returns the top-K most relevant
BIS standards using FAISS cosine similarity search.
"""

import json
import time
import sys
import numpy as np
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------------
REPO_ROOT      = Path(__file__).resolve().parent.parent
VECTOR_DB_DIR  = REPO_ROOT / "vector_db"
INDEX_PATH     = VECTOR_DB_DIR / "faiss_index.bin"
METADATA_PATH  = VECTOR_DB_DIR / "metadata.json"
# Change this:
# EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# To this:
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
TOP_K           = 5

# ---------------------------------------------------------------------------
# Singleton model + index loader
# ---------------------------------------------------------------------------
_model    = None
_index    = None
_metadata = None

def _get_model():
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise RuntimeError("sentence-transformers not installed.")
        print(f"[retriever] Loading embedding model: {EMBEDDING_MODEL}")
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model

def _get_index_and_metadata(index_path: Optional[Path] = None, metadata_path: Optional[Path] = None):
    global _index, _metadata
    if _index is None or _metadata is None:
        try:
            import faiss
        except ImportError:
            raise RuntimeError("faiss-cpu not installed.")
            
        index_path    = index_path    or INDEX_PATH
        metadata_path = metadata_path or METADATA_PATH

        if not index_path.exists() or not metadata_path.exists():
            raise FileNotFoundError("FAISS artifacts not found. Run src/vector_store.py first.")

        _index = faiss.read_index(str(index_path))
        with open(metadata_path, encoding="utf-8") as f:
            _metadata = json.load(f)
            
    return _index, _metadata

# ---------------------------------------------------------------------------
# Core retrieval function
# ---------------------------------------------------------------------------

def retrieve(query: str, top_k: int = TOP_K, index_path: Optional[Path] = None, metadata_path: Optional[Path] = None) -> dict:
    t_start = time.perf_counter()

  # --- Load model and index (cached after first call) ---
    model = _get_model()
    index, metadata = _get_index_and_metadata(index_path, metadata_path)

    # --- Embed the query ---
    # FIX: BGE models require this exact string prepended to the query for retrieval
    bge_query = f"Represent this sentence for searching relevant passages: {query}"
    
    query_vec = model.encode(
        [bge_query], # <--- Pass the prefixed query here
        convert_to_numpy=True,
        normalize_embeddings=True,   
    ).astype(np.float32)

    # --- FAISS search ---
    scores, indices = index.search(query_vec, top_k)

    results = []
    retrieved_standards = []

    for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
        if idx < 0 or idx >= len(metadata):
            continue

        entry = metadata[idx]
        retrieved_standards.append(entry["is_number"])
        results.append({
            "rank":        rank + 1,
            "is_number":   entry["is_number"],
            "title":       entry["title"],
            "section":     entry["section"],
            "scope":       entry["scope"],
            "full_text":   entry.get("full_text", ""), # <-- FIX: Expose full text to the Generator
            "score":       float(round(score, 4)),
        })

    latency = round(time.perf_counter() - t_start, 4)

    return {
        "query":               query,
        "retrieved_standards": retrieved_standards,
        "results":             results,
        "latency_seconds":     latency,
    }

def warmup() -> None:
    print("[retriever] Warming up...")
    _get_model()
    _get_index_and_metadata()
    retrieve("cement building material standard")
    print("[retriever] Warm-up complete.")

if __name__ == "__main__":
    public_test_path = REPO_ROOT / "data" / "public_test_set.json"
    if not public_test_path.exists():
        sys.exit(1)

    with open(public_test_path, encoding="utf-8") as f:
        test_cases = json.load(f)

    warmup()

    hits_at_3  = 0
    mrr_sum    = 0.0
    total_lat  = 0.0

    for item in test_cases:
        result   = retrieve(item["query"])
        expected = set(s.replace(" ", "").lower() for s in item["expected_standards"])
        retrieved_norm = [s.replace(" ", "").lower() for s in result["retrieved_standards"]]

        hit = any(s in expected for s in retrieved_norm[:3])
        if hit:
            hits_at_3 += 1

        mrr = 0.0
        for rank, s in enumerate(retrieved_norm[:5], start=1):
            if s in expected:
                mrr = 1.0 / rank
                break
        mrr_sum   += mrr
        total_lat += result["latency_seconds"]

    n = len(test_cases)
    print(f"\nHit Rate @3 : {hits_at_3/n*100:.1f}%\nMRR @5      : {mrr_sum/n:.4f}\nAvg Latency : {total_lat/n:.3f}s")