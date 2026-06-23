"""
Generate embeddings for text using Amazon Titan Text Embeddings v2 on Bedrock.

Production embedding patterns:
- Batch embedding requests when possible (the API supports batch but the
  per-document call pattern is simpler for a small knowledge base).
- Cache embeddings — never re-embed the same text. We use a content-hash
  cache keyed by the SHA-256 of the input text.
- Pin the model version explicitly. Embedding spaces from different model
  versions are NOT compatible; mixing them breaks retrieval silently.
"""

import hashlib
import json
import logging
from pathlib import Path

import boto3

logger = logging.getLogger(__name__)

# Pinned model version — embedding spaces are version-specific.
# v2 is the current production-grade Titan embedding model on Bedrock.
EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"

# 1024-dim is the default for Titan v2. Other valid sizes: 256, 512.
EMBEDDING_DIMENSIONS = 1024

# Local cache for embeddings — avoids re-embedding identical text across runs.
CACHE_DIR = Path(__file__).parent.parent.parent.parent / "data" / "embedding_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_key(text: str) -> str:
    """Content-hash the text for stable caching."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _cache_path(text: str) -> Path:
    return CACHE_DIR / f"{_cache_key(text)}.json"


def _bedrock_client():
    """Lazy Bedrock client — avoid import-time AWS credentials probe."""
    return boto3.client("bedrock-runtime", region_name="us-east-1")


def embed_text(text: str) -> list[float]:
    """
    Embed a single text string to a 1024-dim vector.

    Cached: identical input text is embedded once per machine. Cache is
    invalidated only by deleting the cache directory.
    """
    cache = _cache_path(text)
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))

    body = json.dumps({
        "inputText": text,
        "dimensions": EMBEDDING_DIMENSIONS,
        "normalize": True,  # Pre-normalized vectors enable cheap cosine sim
    })

    response = _bedrock_client().invoke_model(
        modelId=EMBEDDING_MODEL_ID,
        body=body,
        contentType="application/json",
        accept="application/json",
    )
    response_body = json.loads(response["body"].read())
    embedding = response_body["embedding"]

    cache.write_text(json.dumps(embedding), encoding="utf-8")
    return embedding


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts. Calls embed_text per item, leveraging the cache."""
    embeddings = []
    for i, text in enumerate(texts):
        embedding = embed_text(text)
        embeddings.append(embedding)
        if (i + 1) % 5 == 0:
            logger.info("embedded %d/%d", i + 1, len(texts))
    return embeddings