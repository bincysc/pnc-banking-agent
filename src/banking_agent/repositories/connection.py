"""
Database connection management.

Holds the singleton connection pool for PostgreSQL, the Redis client, and the
MongoDB client. Connections are expensive to establish; production code reuses
them through pooling rather than opening per-request.
"""

import logging
from functools import lru_cache

import psycopg_pool
import pymongo
import redis

from banking_agent.config import get_config

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_postgres_pool() -> psycopg_pool.ConnectionPool:
    """
    Singleton PostgreSQL connection pool.

    The pool is opened lazily on first call and persists for the process lifetime.
    Connections are borrowed via the `connection()` context manager and returned
    automatically when the block exits.
    """
    config = get_config()
    pool = psycopg_pool.ConnectionPool(
        conninfo=config.postgres_dsn,
        min_size=config.postgres_pool_min,
        max_size=config.postgres_pool_max,
        timeout=10,  # Borrow timeout: max seconds to wait for an available connection
        max_lifetime=30 * 60,  # Recycle connections after 30 minutes
        kwargs={"autocommit": False},
    )
    pool.wait()  # Block until min_size connections are established
    logger.info(
        "postgres_pool_ready min=%d max=%d",
        config.postgres_pool_min,
        config.postgres_pool_max,
    )
    return pool


@lru_cache(maxsize=1)
def get_redis_client() -> redis.Redis:
    """
    Singleton Redis client.

    redis-py manages its own internal connection pool; we just need one client
    instance per process. Decoded responses returned as strings for ergonomic use.
    """
    config = get_config()
    client = redis.Redis.from_url(
        config.redis_url,
        decode_responses=True,
        socket_connect_timeout=5,
        socket_timeout=5,
    )
    # Verify connectivity at startup — fail fast if Redis is unreachable
    client.ping()
    logger.info("redis_client_ready")
    return client


@lru_cache(maxsize=1)
def get_mongo_client() -> pymongo.MongoClient:
    """
    Singleton MongoDB client.

    pymongo's MongoClient is itself thread-safe and manages an internal
    connection pool, so a single instance serves the entire application.
    """
    config = get_config()
    client = pymongo.MongoClient(
        config.mongodb_uri,
        serverSelectionTimeoutMS=5000,
        connectTimeoutMS=5000,
    )
    # Fail-fast connectivity check
    client.admin.command("ping")
    logger.info("mongo_client_ready")
    return client