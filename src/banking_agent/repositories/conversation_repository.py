"""
Repository for conversation persistence.

Conversations are stored as documents in MongoDB. Each conversation is a
single document keyed by conversation_id, containing the customer reference,
the message list, and metadata.

The document model fits the access pattern: conversations are read whole
when resuming, written whole when persisting, and the message list has
evolving structure across turn types.
"""

import logging
from datetime import datetime, timezone
from typing import Any

from pymongo.collection import Collection

from banking_agent.repositories.connection import get_mongo_client

logger = logging.getLogger(__name__)


class ConversationRepository:
    """
    Document-store access for conversation history.

    Each conversation is one document:
        {
            "conversation_id": "...",
            "customer_id": "...",
            "messages": [ ... full message list ... ],
            "created_at": ISODate,
            "updated_at": ISODate,
            "metadata": { ... }
        }
    """

    def __init__(self) -> None:
        client = get_mongo_client()
        self._collection: Collection = client["banking"]["conversations"]
        self._ensure_indexes()

    def _ensure_indexes(self) -> None:
        """
        Idempotent index creation. MongoDB's createIndex is a no-op if the
        index already exists, so this can run on every process startup.
        """
        self._collection.create_index("conversation_id", unique=True)
        self._collection.create_index([("customer_id", 1), ("updated_at", -1)])

    def get(self, conversation_id: str) -> dict[str, Any] | None:
        """Load a conversation by ID, or None if it does not exist."""
        return self._collection.find_one(
            {"conversation_id": conversation_id},
            {"_id": 0},  # Exclude MongoDB's internal _id from the result
        )

    def upsert(
        self,
        conversation_id: str,
        customer_id: str,
        messages: list[dict[str, Any]],
    ) -> None:
        """
        Insert or update a conversation document.

        Uses an atomic upsert: if the conversation exists, the messages and
        updated_at are replaced; if not, a new document is created with
        created_at set.
        """
        now = datetime.now(timezone.utc)
        self._collection.update_one(
            {"conversation_id": conversation_id},
            {
                "$set": {
                    "customer_id": customer_id,
                    "messages": messages,
                    "updated_at": now,
                },
                "$setOnInsert": {
                    "conversation_id": conversation_id,
                    "created_at": now,
                },
            },
            upsert=True,
        )
        logger.debug(
            "conversation_persisted conversation_id=%s messages=%d",
            conversation_id,
            len(messages),
        )

    def list_recent_for_customer(
        self, customer_id: str, limit: int = 10
    ) -> list[dict[str, Any]]:
        """List recent conversations for a customer, most-recent-first."""
        cursor = (
            self._collection.find(
                {"customer_id": customer_id},
                {"_id": 0, "messages": 0},  # Exclude messages for list view
            )
            .sort("updated_at", -1)
            .limit(limit)
        )
        return list(cursor)