"""
Verify the document chunker against the policy knowledge base.

Run this after creating policy documents and before building the vector
store. The output should show one chunk per H2 section, with section
titles preserved.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from banking_agent.rag.chunker import chunk_knowledge_base

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

KB_DIR = Path(__file__).parent.parent / "data" / "knowledge_base"


def main() -> None:
    chunks = chunk_knowledge_base(KB_DIR)

    print(f"\nTotal chunks: {len(chunks)}\n")

    # Show first chunk from each document for inspection
    seen_docs = set()
    for chunk in chunks:
        if chunk.document_name in seen_docs:
            continue
        seen_docs.add(chunk.document_name)
        print(f"--- {chunk.chunk_id} ---")
        print(f"Section: {chunk.section_title}")
        print(f"Length:  {len(chunk.content)} chars")
        print(f"Preview: {chunk.content[:150]}...")
        print()


if __name__ == "__main__":
    main()