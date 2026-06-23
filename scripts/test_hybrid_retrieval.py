"""
Compare vector-only, BM25-only, and hybrid retrieval for the same queries.

Hybrid should consistently match or beat each individual retriever. Watch
for queries where one method ranks the right chunk #4 and another ranks
it #1 — hybrid should surface it at #1 or #2.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from banking_agent.rag.retrieval import hybrid_retrieve, query_bm25
from banking_agent.rag.vector_store import query_vectors

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def compare(query: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"Query: {query}")
    print("=" * 70)

    print("\n--- Vector only ---")
    for i, h in enumerate(query_vectors(query, k=3), 1):
        print(f"  [{i}] {h['document_name']} :: {h['section_title']} (sim={h['similarity']:.3f})")

    print("\n--- BM25 only ---")
    for i, h in enumerate(query_bm25(query, k=3), 1):
        print(f"  [{i}] {h['document_name']} :: {h['section_title']} (score={h['score']:.3f})")

    print("\n--- Hybrid (RRF) ---")
    for i, r in enumerate(hybrid_retrieve(query, k=3), 1):
        provenance = []
        if r.vector_rank:
            provenance.append(f"vec#{r.vector_rank}")
        if r.bm25_rank:
            provenance.append(f"bm25#{r.bm25_rank}")
        print(f"  [{i}] {r.document_name} :: {r.section_title} (rrf={r.score:.4f}, {' '.join(provenance)})")


def main() -> None:
    # Semantic query — vector should excel
    compare("how do I get my money back from a fraudulent charge?")

    # Exact-term query — BM25 should excel
    compare("Regulation E provisional credit timeline")

    # Mixed — hybrid should outperform both
    compare("daily limit for international wires")

    # Number-heavy — BM25 territory
    compare("$5000 mobile deposit")


if __name__ == "__main__":
    main()