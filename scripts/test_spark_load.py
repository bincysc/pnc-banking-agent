"""
Quick test script for the PySpark loader.

Run from project root:
    python scripts/test_spark_load.py
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from banking_agent.data_pipeline.load_from_postgres import (
    load_accounts,
    load_customers,
    load_transactions,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def main() -> None:
    # Load — these are lazy operations, returning DataFrames
    customers = load_customers()
    accounts = load_accounts()
    transactions = load_transactions()

    # Print schemas — this does NOT trigger execution, just inspects metadata
    print("\n--- Customer schema ---")
    customers.printSchema()

    print("\n--- Account schema ---")
    accounts.printSchema()

    print("\n--- Transaction schema ---")
    transactions.printSchema()

    # Show a few rows from each — show() IS an action, it triggers execution
    print("\n--- Customer sample ---")
    customers.show(5, truncate=False)

    print("\n--- Account sample ---")
    accounts.show(5, truncate=False)

    print("\n--- Transaction sample ---")
    transactions.show(5, truncate=False)

    # An aggregation — also an action
    print("\n--- Transactions by category ---")
    from pyspark.sql.functions import count
    from pyspark.sql.functions import sum as spark_sum

    by_category = (
        transactions
        .groupBy("category")
        .agg(
            count("*").alias("count"),
            spark_sum("amount_cents").alias("total_cents"),
        )
        .orderBy("category")
    )
    by_category.show(truncate=False)


if __name__ == "__main__":
    main()
