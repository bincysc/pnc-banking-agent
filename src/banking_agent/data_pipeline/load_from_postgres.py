"""
Load banking data from PostgreSQL into PySpark DataFrames.

This is the *extract* phase of the ETL pipeline. PostgreSQL is the operational
source of truth; Spark reads from it through the JDBC driver and processes the
data through the DataFrame API. The downstream transformations and writes to
Delta Lake will follow in subsequent files.

PostgreSQL JDBC driver is downloaded automatically by Spark on first run.
"""

import logging

from pyspark.sql import DataFrame
from pyspark.sql.types import (
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from banking_agent.config import get_config
from banking_agent.data_pipeline.spark_session import get_spark

logger = logging.getLogger(__name__)


# --- Explicit schemas — production discipline ----------------------------

# Defining the schema explicitly is faster than inference and serves as
# documentation. Spark JDBC will refuse to load if the source schema does
# not match — which is the failure mode we want.

CUSTOMER_SCHEMA = StructType([
    StructField("customer_id", StringType(), nullable=False),
    StructField("first_name", StringType(), nullable=False),
    StructField("last_name", StringType(), nullable=False),
    StructField("email", StringType(), nullable=False),
    StructField("enrollment_date", TimestampType(), nullable=False),
    StructField("created_at", TimestampType(), nullable=False),
    StructField("updated_at", TimestampType(), nullable=False),
])

ACCOUNT_SCHEMA = StructType([
    StructField("account_id", StringType(), nullable=False),
    StructField("customer_id", StringType(), nullable=False),
    StructField("account_type", StringType(), nullable=False),
    StructField("balance_cents", LongType(), nullable=False),
    StructField("currency", StringType(), nullable=False),
    StructField("opened_date", TimestampType(), nullable=False),
    StructField("status", StringType(), nullable=False),
    StructField("created_at", TimestampType(), nullable=False),
    StructField("updated_at", TimestampType(), nullable=False),
])

TRANSACTION_SCHEMA = StructType([
    StructField("transaction_id", StringType(), nullable=False),
    StructField("account_id", StringType(), nullable=False),
    StructField("timestamp", TimestampType(), nullable=False),
    StructField("amount_cents", LongType(), nullable=False),
    StructField("merchant", StringType(), nullable=True),
    StructField("category", StringType(), nullable=False),
    StructField("status", StringType(), nullable=False),
    StructField("created_at", TimestampType(), nullable=False),
])


# --- PostgreSQL JDBC connection helper ------------------------------------

def _jdbc_url() -> str:
    """
    Convert our libpq-style DSN to a JDBC URL.

    PostgreSQL's two URL formats — the libpq URI used by psycopg, and the
    JDBC URL used by JVM-based clients like Spark — are slightly different.
    Production code typically configures the JDBC URL directly; for this
    project we derive it from the DSN we already have.
    """
    config = get_config()
    # DSN looks like: postgresql://agent:agent_dev_password@localhost:5432/banking
    # JDBC needs:     jdbc:postgresql://localhost:5432/banking
    # User and password go as separate properties, not in the URL.
    dsn = config.postgres_dsn
    # Strip the protocol and credentials from the front
    after_at = dsn.split("@", 1)[1]  # "localhost:5432/banking"
    return f"jdbc:postgresql://{after_at}"


def _jdbc_properties() -> dict[str, str]:
    """Connection properties for the JDBC driver."""
    return {
        "user": "agent",
        "password": "agent_dev_password",
        "driver": "org.postgresql.Driver",
    }


# --- Public load functions ------------------------------------------------

def load_customers() -> DataFrame:
    """Load the customers table from PostgreSQL."""
    spark = get_spark()
    df = (
        spark.read
        .format("jdbc")
        .option("url", _jdbc_url())
        .option("dbtable", "customers")
        .options(**_jdbc_properties())
        .load()
    )
    logger.info("loaded_customers count=%d", df.count())
    return df


def load_accounts() -> DataFrame:
    """Load the accounts table from PostgreSQL."""
    spark = get_spark()
    df = (
        spark.read
        .format("jdbc")
        .option("url", _jdbc_url())
        .option("dbtable", "accounts")
        .options(**_jdbc_properties())
        .load()
    )
    logger.info("loaded_accounts count=%d", df.count())
    return df


def load_transactions() -> DataFrame:
    """Load the transactions table from PostgreSQL."""
    spark = get_spark()
    df = (
        spark.read
        .format("jdbc")
        .option("url", _jdbc_url())
        .option("dbtable", "transactions")
        .options(**_jdbc_properties())
        .load()
    )
    logger.info("loaded_transactions count=%d", df.count())
    return df
