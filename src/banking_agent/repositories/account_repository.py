"""
Repository for the Account domain entity.

Reads from PostgreSQL with a Redis cache layer following the cache-aside
pattern. Writes invalidate the cache after the durable store update.
"""

import json
import logging
from typing import Any

from banking_agent.config import get_config
from banking_agent.repositories.connection import get_postgres_pool, get_redis_client

logger = logging.getLogger(__name__)


class AccountRepository:
    """
    Account data access.

    All methods are cache-aware: reads check Redis first, writes invalidate
    Redis after the PostgreSQL update commits.
    """

    def __init__(self) -> None:
        self._pool = get_postgres_pool()
        self._cache = get_redis_client()
        self._ttl = get_config().cache_ttl_account

    @staticmethod
    def _cache_key(account_id: str) -> str:
        return f"account:{account_id}"

    def get_by_id(self, account_id: str) -> dict[str, Any] | None:
        """
        Retrieve an account by ID with cache-aside semantics.

        Returns None if the account does not exist; the absence is not cached
        (we re-query PostgreSQL each time for unknown IDs, which is the safe
        default — caching negative results requires negative-cache TTL tuning).
        """
        cache_key = self._cache_key(account_id)

        # 1. Cache lookup
        cached = self._cache.get(cache_key)
        if cached is not None:
            logger.debug("cache_hit key=%s", cache_key)
            return json.loads(cached)

        logger.debug("cache_miss key=%s", cache_key)

        # 2. Durable store query
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT account_id, customer_id, account_type, balance_cents,
                           currency, opened_date, status
                    FROM accounts
                    WHERE account_id = %s
                    """,
                    (account_id,),
                )
                row = cur.fetchone()

        if row is None:
            return None

        account = {
            "account_id": row[0],
            "customer_id": row[1],
            "account_type": row[2],
            "balance_cents": row[3],
            "currency": row[4],
            "opened_date": row[5].isoformat(),
            "status": row[6],
        }

        # 3. Populate cache with TTL
        self._cache.setex(cache_key, self._ttl, json.dumps(account))

        return account

    def list_by_customer(self, customer_id: str) -> list[dict[str, Any]]:
        """
        List all accounts for a given customer.

        Not cached at this layer — the access pattern is rare and the result
        set varies per customer, which would produce many distinct cache keys.
        Individual account lookups still benefit from the per-account cache.
        """
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT account_id, customer_id, account_type, balance_cents,
                           currency, opened_date, status
                    FROM accounts
                    WHERE customer_id = %s
                    ORDER BY account_id
                    """,
                    (customer_id,),
                )
                rows = cur.fetchall()

        return [
            {
                "account_id": r[0],
                "customer_id": r[1],
                "account_type": r[2],
                "balance_cents": r[3],
                "currency": r[4],
                "opened_date": r[5].isoformat(),
                "status": r[6],
            }
            for r in rows
        ]

    def invalidate_cache(self, account_id: str) -> None:
        """
        Remove an account from the cache. Called after writes to maintain
        consistency between cache and durable store.
        """
        self._cache.delete(self._cache_key(account_id))
        logger.debug("cache_invalidated key=%s", self._cache_key(account_id))
