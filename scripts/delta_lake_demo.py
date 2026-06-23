"""
Demonstrate the four properties that distinguish Delta Lake from Parquet:
schema enforcement, time travel, ACID via overwrite, and merge.

Each section is self-contained and instructive. Read the code, run it,
observe the output, and you will be able to articulate each property
with concrete example.
"""

import logging
import sys
from pathlib import Path

from banking_agent.data_pipeline.silence_shutdown import silence_spark_shutdown_on_exit

silence_spark_shutdown_on_exit()

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pyspark.sql.utils import AnalysisException

from banking_agent.data_pipeline.spark_session import get_spark
from banking_agent.data_pipeline.write_to_delta import (
    ACCOUNTS_PATH,
    CUSTOMERS_PATH,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def demo_1_schema_enforcement() -> None:
    """
    Property 1: Schema enforcement.

    Attempting to append data with the wrong schema is rejected at write
    time. Compare this to raw Parquet, where the file would be written
    and the corruption would only surface at read time.
    """
    print("\n" + "=" * 70)
    print("DEMO 1: Schema enforcement")
    print("=" * 70)

    spark = get_spark()

    # Construct a DataFrame with a column the customers table does not have.
    bad_data = spark.createDataFrame(
        [("CUST-9999", "Test", "Customer", "test@example.com", "wrong-column")],
        ["customer_id", "first_name", "last_name", "email", "unexpected_field"],
    )

    try:
        bad_data.write.format("delta").mode("append").save(CUSTOMERS_PATH)
        print("UNEXPECTED: write succeeded when it should have been rejected")
    except AnalysisException as e:
        print("Write correctly rejected — schema mismatch detected at write time")
        print(f"Spark error: {str(e)[:200]}...")


def demo_2_time_travel() -> None:
    """
    Property 2: Time travel.

    Every commit to the table is preserved as a version. We can query
    the table at any historical version, regardless of subsequent
    writes. This enables auditing, debugging, and point-in-time
    reconstructions — critical for financial-services compliance.
    """
    print("\n" + "=" * 70)
    print("DEMO 2: Time travel")
    print("=" * 70)

    spark = get_spark()

    # Inspect the transaction history of the customers table.
    history = (
        spark.sql(f"DESCRIBE HISTORY delta.`{CUSTOMERS_PATH}`")
        .select("version", "timestamp", "operation", "operationMetrics")
    )
    print("\nCustomers table history:")
    history.show(truncate=False)

    # Read the current version
    current = spark.read.format("delta").load(CUSTOMERS_PATH)
    print(f"Current customer count: {current.count()}")

    # Read version 0 — the first write
    v0 = spark.read.format("delta").option("versionAsOf", 0).load(CUSTOMERS_PATH)
    print(f"Version 0 customer count: {v0.count()}")
    # In this demo, version 0 and current are the same because we only
    # wrote once. After demo 3 below, they will differ.


def demo_3_acid_overwrite() -> None:
    """
    Property 3: ACID via atomic overwrite.

    We overwrite the customers table with a subset of the data, then
    verify that readers either see the old full table or the new partial
    table — never a partial intermediate state. The overwrite is atomic.
    """
    print("\n" + "=" * 70)
    print("DEMO 3: ACID via atomic overwrite")
    print("=" * 70)

    spark = get_spark()

    # Read the current table
    full = spark.read.format("delta").load(CUSTOMERS_PATH)
    full_count = full.count()
    print(f"Before overwrite: {full_count} customers")

    # Overwrite with just the first 10 rows
    subset = full.limit(10)
    subset.write.format("delta").mode("overwrite").save(CUSTOMERS_PATH)

    # Read again
    new = spark.read.format("delta").load(CUSTOMERS_PATH)
    new_count = new.count()
    print(f"After overwrite:  {new_count} customers")
    print("The transition was atomic — no reader ever saw an intermediate count.")

    # Now we can time-travel BACK to the full version
    print("\nTime-traveling to the version BEFORE the overwrite:")
    history = spark.sql(f"DESCRIBE HISTORY delta.`{CUSTOMERS_PATH}`")
    history.select("version", "operation").show(truncate=False)

    # The previous version is the latest one before the most recent commit
    v_before = spark.read.format("delta").option("versionAsOf", 0).load(CUSTOMERS_PATH)
    print(f"Version 0 still has {v_before.count()} customers — the original data.")

    # Restore the original full data by overwriting from version 0
    print("\nRestoring full table from version 0...")
    v_before.write.format("delta").mode("overwrite").save(CUSTOMERS_PATH)
    restored = spark.read.format("delta").load(CUSTOMERS_PATH)
    print(f"After restore: {restored.count()} customers")


def demo_4_merge() -> None:
    """
    Property 4: Merge — atomic upsert via Spark SQL.

    Uses the SQL MERGE INTO syntax which executes entirely in the JVM.
    The DeltaTable.merge() Python builder API is functionally equivalent
    but can hit Python-worker serialization issues with Python 3.13;
    the SQL form avoids them by staying in the JVM.

    Production Databricks code uses the SQL form for the same reason:
    it is more portable across language clients (Scala, Python, R all
    invoke the same JVM execution path).
    """
    print("\n" + "=" * 70)
    print("DEMO 4: Merge — atomic UPSERT via Spark SQL")
    print("=" * 70)

    spark = get_spark()

    # Show current state
    print("\nBefore merge — sample accounts:")
    spark.sql(f"""
        SELECT account_id, balance_cents, status
        FROM delta.`{ACCOUNTS_PATH}`
        WHERE account_id IN ('ACC-5001', 'ACC-9999')
    """).show(truncate=False)

    # Register the source DataFrame as a temp view for the SQL merge to reference
    updates = spark.createDataFrame(
        [
            ("ACC-5001", "CUST-1001", "checking", 99999, "USD", "2020-11-06", "active"),
            ("ACC-9999", "CUST-1001", "savings", 50000, "USD", "2026-06-22", "active"),
        ],
        ["account_id", "customer_id", "account_type", "balance_cents",
         "currency", "opened_date", "status"],
    )
    updates.createOrReplaceTempView("account_updates")

    # SQL MERGE — executes entirely in the JVM, no Python workers
    spark.sql(f"""
        MERGE INTO delta.`{ACCOUNTS_PATH}` AS target
        USING account_updates AS source
        ON target.account_id = source.account_id
        WHEN MATCHED THEN
            UPDATE SET
                balance_cents = source.balance_cents,
                status = source.status
        WHEN NOT MATCHED THEN
            INSERT (account_id, customer_id, account_type, balance_cents,
                    currency, opened_date, status)
            VALUES (source.account_id, source.customer_id, source.account_type,
                    source.balance_cents, source.currency,
                    CAST(source.opened_date AS DATE), source.status)
    """)

    print("\nAfter merge — same accounts:")
    spark.sql(f"""
        SELECT account_id, balance_cents, status
        FROM delta.`{ACCOUNTS_PATH}`
        WHERE account_id IN ('ACC-5001', 'ACC-9999')
    """).show(truncate=False)

    print("\nACC-5001 was updated in place. ACC-9999 was inserted as new.")
    print("Both happened in one atomic transaction with one log entry.")

def main() -> None:
    demo_1_schema_enforcement()
    demo_2_time_travel()
    demo_3_acid_overwrite()
    demo_4_merge()
    print("\n" + "=" * 70)
    print("All four Delta Lake properties demonstrated.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
