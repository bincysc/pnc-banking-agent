"""
Quick test of vector retrieval — query the policy knowledge base.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from banking_agent.rag.vector_store import query_vectors

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def show(query: str) -> None:
    print(f"\n=== Query: {query} ===")
    hits = query_vectors(query, k=3)
    for i, hit in enumerate(hits, 1):
        print(f"\n[{i}] {hit['document_name']} — {hit['section_title']} (similarity={hit['similarity']:.3f})")
        print(f"    {hit['content'][:200]}...")


def main() -> None:
    show("what is the daily wire transfer limit?")
    show("how do I report fraud on my account?")
    show("what is the foreign transaction fee?")
    show("can I deposit a $50,000 check on mobile?")


if __name__ == "__main__":
    main()