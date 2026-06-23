"""
Vector store backed by ChromaDB.

ChromaDB persists to local files under data/chroma/. The collection is a
named bucket of (id, vector, metadata, text) tuples. We store chunk metadata
alongside the vector so retrieval results include citation-ready info
(document name, section title) without needing a second lookup.

For production scale, swap ChromaDB for pgvector (in PostgreSQL), Pinecone
(managed vector DB), or OpenSearch (Elasticsearch with k-NN plugin). The
interface — store and query — is identical across all of them.
"""

import logging
from pathlib import Path

import chromadb
from chromadb.config import Settings

from banking_agent.rag.chunker import Chunk
from banking_agent.rag.embeddings import embed_texts

logger = logging.getLogger(__name__)

CHROMA_DIR = Path(__file__).parent.parent.parent.parent / "data" / "chroma"
COLLECTION_NAME = "banking_policies"


def _chroma_client():
    """Persistent ChromaDB client — files live under data/chroma/."""
    return chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )


def get_or_create_collection():
    """
    Get the policies collection, creating it if needed.

    We pass embedding_function=None because we manage embeddings ourselves
    via Bedrock — ChromaDB has its own embedding functions, but using them
    would couple us to ChromaDB's default models and break the version-pinning
    discipline.
    """
    client = _chroma_client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=None,
        metadata={"hnsw:space": "cosine"},  # cosine similarity for normalized vectors
    )


def index_chunks(chunks: list[Chunk]) -> None:
    """
    Embed all chunks and write them to the vector store.

    Replaces any existing chunks in the collection. The chunk_id is the
    primary key, so re-indexing the same content is idempotent.
    """
    collection = get_or_create_collection()

    # Wipe and rebuild — simplest correctness model for a small KB
    existing = collection.get()
    if existing["ids"]:
        collection.delete(ids=existing["ids"])
        logger.info("cleared_existing count=%d", len(existing["ids"]))

    texts = [c.content for c in chunks]
    embeddings = embed_texts(texts)

    collection.add(
        ids=[c.chunk_id for c in chunks],
        embeddings=embeddings,
        documents=texts,
        metadatas=[
            {
                "document_name": c.document_name,
                "section_title": c.section_title,
            }
            for c in chunks
        ],
    )
    logger.info("indexed chunks=%d collection=%s", len(chunks), COLLECTION_NAME)


def query_vectors(query_text: str, k: int = 5) -> list[dict]:
    """
    Vector-similarity search for a query.

    Returns the top k chunks by cosine similarity, each with metadata
    (document_name, section_title) and the chunk text. Distance is
    returned as similarity score (higher = more similar).
    """
    from banking_agent.rag.embeddings import embed_text

    collection = get_or_create_collection()
    query_embedding = embed_text(query_text)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
    )

    # ChromaDB returns parallel arrays — zip them into structured results
    hits = []
    for chunk_id, doc, metadata, distance in zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append({
            "chunk_id": chunk_id,
            "content": doc,
            "document_name": metadata["document_name"],
            "section_title": metadata["section_title"],
            "similarity": 1.0 - distance,  # convert cosine distance to similarity
        })
    return hits