"""
Repository for the Transaction domain entity.

Transactions are not cached at this layer — they are append-only and the
volume of distinct queries (by account, date range, category) would produce
high cache key cardinality with low hit ratio. The dominant query pattern
hits the composite index on (account_id, timestamp DESC) directly.
"""

import logging
from typing import Any

from banking_agent.repositories.connection import get_postgres_pool

logger = logging.getLogger(__name__)


class TransactionRepository:
    """
    Transaction data access.

    Queries hit PostgreSQL directly without caching, because the access
    patterns generate many distinct queries with low repetition.
    """

    def __init__(self) -> None:
        self._pool = get_postgres_pool()

    def list_recent_for_account(
        self, account_id: str, limit: int = 10
    ) -> list[dict[str, Any]]:
        """
        Return the most recent N transactions for an account.

        Uses the composite index on (account_id, timestamp DESC) — verify
        with EXPLAIN ANALYZE in production to confirm the index is selected.
        """
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT transaction_id, account_id, timestamp, amount_cents,
                           merchant, category, status
                    FROM transactions
                    WHERE account_id = %s
                    ORDER BY timestamp DESC
                    LIMIT %s
                    """,
                    (account_id, limit),
                )
                rows = cur.fetchall()

        return [
            {
                "transaction_id": r[0],
                "account_id": r[1],
                "timestamp": r[2].isoformat(),
                "amount_cents": r[3],
                "merchant": r[4],
                "category": r[5],
                "status": r[6],
            }
            for r in rows
        ]

    def list_by_category(
        self, account_id: str, category: str, limit: int = 50
    ) -> list[dict[str, Any]]:
        """
        Return recent transactions for an account filtered by category.

        Hits the composite index on (account_id, category). Useful for
        the agent answering "what did I spend on groceries this month".
        """
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT transaction_id, account_id, timestamp, amount_cents,
                           merchant, category, status
                    FROM transactions
                    WHERE account_id = %s AND category = %s
                    ORDER BY timestamp DESC
                    LIMIT %s
                    """,
                    (account_id, category, limit),
                )
                rows = cur.fetchall()

        return [
            {
                "transaction_id": r[0],
                "account_id": r[1],
                "timestamp": r[2].isoformat(),
                "amount_cents": r[3],
                "merchant": r[4],
                "category": r[5],
                "status": r[6],
            }
            for r in rows
        ]