"""
Ingest the policy knowledge base into the vector store.

Reads all markdown files under data/knowledge_base/, chunks them, embeds
each chunk via Bedrock, and writes the embeddings to ChromaDB.

Run this whenever the knowledge base changes. The script is idempotent —
re-running it replaces the existing index rather than duplicating it.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from banking_agent.rag.chunker import chunk_knowledge_base
from banking_agent.rag.vector_store import index_chunks

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

KB_DIR = Path(__file__).parent.parent / "data" / "knowledge_base"


def main() -> None:
    chunks = chunk_knowledge_base(KB_DIR)
    print(f"Chunked: {len(chunks)} chunks from {KB_DIR}")
    index_chunks(chunks)
    print("Ingestion complete.")


if __name__ == "__main__":
    main()