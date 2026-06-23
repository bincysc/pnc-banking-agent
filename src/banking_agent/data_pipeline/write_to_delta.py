"""
Bronze-to-silver ingestion: read operational data from PostgreSQL,
clean and validate it, and write to Delta Lake.

The 'bronze' layer in this project is PostgreSQL itself — the operational
source of truth maintained by the application. The silver layer is the
cleaned, well-typed Delta tables this module produces. Downstream
analytical and ML workloads read from silver, not from PostgreSQL.
"""

import logging
from pathlib import Path

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from banking_agent.data_pipeline.load_from_postgres import (
    load_accounts,
    load_customers,
    load_transactions,
)

logger = logging.getLogger(__name__)


# --- Path conventions ----------------------------------------------------

# Resolved relative to the project root. In production this would be an
# S3, ADLS, or GCS path; the API is identical.
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
SILVER_ROOT = PROJECT_ROOT / "data" / "lakehouse" / "silver"

CUSTOMERS_PATH = str(SILVER_ROOT / "customers")
ACCOUNTS_PATH = str(SILVER_ROOT / "accounts")
TRANSACTIONS_PATH = str(SILVER_ROOT / "transactions")


# --- Transformations -----------------------------------------------------

def _clean_customers(df: DataFrame) -> DataFrame:
    """
    Silver-layer customer transformations.

    Adds an ingestion timestamp for lineage. In production, transformations
    here would also include PII handling (masking, tokenization), reference
    data joins (geocoding addresses, mapping channels), and validation
    (rejecting rows that fail business rules to a quarantine table).
    """
    return (
        df
        # Add the silver-layer ingestion timestamp — lineage metadata.
        .withColumn("ingested_at", F.current_timestamp())
        # Normalize email to lowercase — production discipline; queries
        # against email should be case-insensitive by virtue of the data.
        .withColumn("email", F.lower(F.col("email")))
    )


def _clean_accounts(df: DataFrame) -> DataFrame:
    """Silver-layer account transformations."""
    return (
        df
        .withColumn("ingested_at", F.current_timestamp())
        # Derive balance in dollars as a convenience column for downstream
        # analytical queries. The integer cents remain the source of truth.
        .withColumn("balance_dollars", F.col("balance_cents") / 100.0)
    )


def _clean_transactions(df: DataFrame) -> DataFrame:
    """
    Silver-layer transaction transformations.

    Partitioned by year-month for query efficiency. Transaction queries
    are overwhelmingly date-bounded ('show me my spending last month'),
    and partition pruning eliminates entire date ranges from the scan.
    """
    return (
        df
        .withColumn("ingested_at", F.current_timestamp())
        .withColumn("amount_dollars", F.col("amount_cents") / 100.0)
        # Derived partition columns. Spark prunes by these at query time.
        .withColumn("year", F.year(F.col("timestamp")))
        .withColumn("month", F.month(F.col("timestamp")))
    )


# --- Delta writers -------------------------------------------------------

def write_customers_to_silver() -> None:
    """
    Full-refresh write of customer data to the silver layer.

    Mode 'overwrite' replaces the entire table on each run. This is the
    correct semantic for a small dimension table that does not need
    incremental processing. Larger dimensions or fact tables would use
    'append' or merge for incremental ingestion.
    """
    df = _clean_customers(load_customers())
    (
        df.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")  # Allow schema changes on rewrite
        .save(CUSTOMERS_PATH)
    )
    logger.info("wrote_customers_silver count=%d path=%s", df.count(), CUSTOMERS_PATH)


def write_accounts_to_silver() -> None:
    """Full-refresh write of account data to the silver layer."""
    df = _clean_accounts(load_accounts())
    (
        df.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .save(ACCOUNTS_PATH)
    )
    logger.info("wrote_accounts_silver count=%d path=%s", df.count(), ACCOUNTS_PATH)


def write_transactions_to_silver() -> None:
    """
    Partitioned write of transaction data to the silver layer.

    Partitioning by (year, month) means each partition is a separate
    Parquet directory. Queries filtered by year and month read only the
    relevant partitions — partition pruning. For a transactions table
    with billions of rows over years of history, this is the difference
    between scanning the whole table and scanning a small slice.
    """
    def _clean_customers(df: DataFrame) -> DataFrame:
        return (
            df
            .withColumn("ingested_at", F.current_timestamp())
            .withColumn("email", F.lower(F.col("email")))
        )


# --- Convenience runner --------------------------------------------------

def write_all_silver() -> None:
    """Run all three writers in sequence. Idempotent — safe to re-run."""
    write_customers_to_silver()
    write_accounts_to_silver()
    write_transactions_to_silver()
