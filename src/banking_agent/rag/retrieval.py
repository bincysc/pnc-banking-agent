"""
Hybrid retrieval: vector + BM25, fused with Reciprocal Rank Fusion.

Vector search (dense retrieval) captures semantic match — paraphrased
queries, synonyms, conceptual similarity. BM25 (sparse retrieval) captures
exact-term match — proper nouns, numbers, specific terminology. Running
both and fusing their rankings handles both query types and produces
more robust retrieval than either alone.

Reciprocal Rank Fusion (RRF) is the standard fusion algorithm: each
result's score becomes sum(1 / (k + rank_in_each_list)). The constant k=60
is the value used in the original RRF paper and works well in practice.
RRF sidesteps the score-normalization problem because it uses ranks, not
raw scores, which means it works regardless of how different the two
retrievers' score distributions are.
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

from banking_agent.rag.chunker import Chunk, chunk_knowledge_base
from banking_agent.rag.vector_store import query_vectors

logger = logging.getLogger(__name__)

# RRF constant from the original paper. Higher k = more weight to lower
# ranks; lower k = more weight to top ranks. k=60 is the proven default.
RRF_K = 60

KB_DIR = Path(__file__).parent.parent.parent.parent / "data" / "knowledge_base"


# --- Result container ----------------------------------------------------

@dataclass
class RetrievalResult:
    """One ranked chunk in a retrieval result set."""
    chunk_id: str
    document_name: str
    section_title: str
    content: str
    score: float           # The fused RRF score
    vector_rank: int | None  # Rank in the vector search, None if not in top-k
    bm25_rank: int | None    # Rank in BM25, None if not in top-k


# --- Simple tokenizer for BM25 -------------------------------------------

def _tokenize(text: str) -> list[str]:
    """
    Tokenize text for BM25.

    BM25 operates on bag-of-words. We lowercase and split on non-word
    characters, then drop very short tokens. Production systems would use
    a proper tokenizer (e.g., spaCy) and possibly a domain stemmer; this
    simple version is adequate for English policy text.
    """
    text = text.lower()
    tokens = re.findall(r"\w+", text)
    return [t for t in tokens if len(t) > 1]


# --- BM25 index (built lazily, cached in module) -------------------------

_bm25_index: BM25Okapi | None = None
_bm25_chunks: list[Chunk] | None = None


def _ensure_bm25_index() -> tuple[BM25Okapi, list[Chunk]]:
    """Build the BM25 index on first use; cache for subsequent queries."""
    global _bm25_index, _bm25_chunks
    if _bm25_index is None or _bm25_chunks is None:
        chunks = chunk_knowledge_base(KB_DIR)
        tokenized_corpus = [_tokenize(c.content) for c in chunks]
        _bm25_index = BM25Okapi(tokenized_corpus)
        _bm25_chunks = chunks
        logger.info("bm25_index_built corpus_size=%d", len(chunks))
    return _bm25_index, _bm25_chunks


def query_bm25(query_text: str, k: int = 5) -> list[dict]:
    """BM25 keyword search. Returns top k chunks with their BM25 scores."""
    bm25, chunks = _ensure_bm25_index()
    tokens = _tokenize(query_text)
    scores = bm25.get_scores(tokens)

    # Get indices of top-k by score
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]

    hits = []
    for idx, score in ranked:
        chunk = chunks[idx]
        hits.append({
            "chunk_id": chunk.chunk_id,
            "content": chunk.content,
            "document_name": chunk.document_name,
            "section_title": chunk.section_title,
            "score": float(score),
        })
    return hits


# --- Hybrid retrieval with Reciprocal Rank Fusion -----------------------

def hybrid_retrieve(query_text: str, k: int = 5, per_retriever_k: int = 10) -> list[RetrievalResult]:
    """
    Run vector and BM25 retrieval in parallel, fuse results with RRF.

    per_retriever_k controls how many candidates each retriever produces
    before fusion. Going higher than k catches chunks that one retriever
    ranks lower but the other ranks high — exactly the case where hybrid
    beats either alone.
    """
    vector_hits = query_vectors(query_text, k=per_retriever_k)
    bm25_hits = query_bm25(query_text, k=per_retriever_k)

    # Build a unified map of chunk_id → RRF score and provenance
    scores: dict[str, float] = {}
    metadata: dict[str, dict] = {}
    vector_ranks: dict[str, int] = {}
    bm25_ranks: dict[str, int] = {}

    for rank, hit in enumerate(vector_hits, start=1):
        cid = hit["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank)
        metadata[cid] = hit
        vector_ranks[cid] = rank

    for rank, hit in enumerate(bm25_hits, start=1):
        cid = hit["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank)
        if cid not in metadata:
            metadata[cid] = hit
        bm25_ranks[cid] = rank

    # Sort by fused score, take top k
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]

    results = []
    for cid, score in ranked:
        m = metadata[cid]
        results.append(RetrievalResult(
            chunk_id=cid,
            document_name=m["document_name"],
            section_title=m["section_title"],
            content=m["content"],
            score=score,
            vector_rank=vector_ranks.get(cid),
            bm25_rank=bm25_ranks.get(cid),
        ))
    return results