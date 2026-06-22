"""
Generate synthetic banking data and load it into PostgreSQL.

This script is idempotent — running it multiple times replaces existing data
rather than duplicating it. Designed for local development; do not run against
production data stores.

Usage:
    python scripts/seed_data.py
"""

import logging
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import psycopg
from faker import Faker

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from banking_agent.config import get_config

# Deterministic faker seed — same data every run, useful for development.
fake = Faker()
Faker.seed(42)
random.seed(42)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# --- Configuration -----------------------------------------------------------

NUM_CUSTOMERS = 50
ACCOUNTS_PER_CUSTOMER = (1, 3)
TRANSACTIONS_PER_ACCOUNT = (15, 40)

TRANSACTION_CATEGORIES = [
    ("groceries", -10000, -2000, ["Whole Foods", "Trader Joes", "Kroger", "Safeway"]),
    ("dining", -5000, -800, ["Starbucks", "Chipotle", "Local Restaurant", "Uber Eats"]),
    ("gas", -8000, -1500, ["Shell", "Exxon", "BP", "Chevron"]),
    ("subscriptions", -3000, -500, ["Netflix", "Spotify", "NYTimes", "Adobe"]),
    ("travel", -50000, -5000, ["Delta Airlines", "Marriott", "United", "Airbnb"]),
    ("utilities", -25000, -3000, ["Electric Co", "Water Dept", "Gas Co", "Internet ISP"]),
    ("income", 200000, 800000, ["ACH-Payroll", "Direct-Deposit"]),
    ("transfer", 5000, 200000, ["Transfer-from-savings", "Transfer-from-checking"]),
]


# --- Connection helper -------------------------------------------------------

def get_pg_connection() -> psycopg.Connection:
    """Open a PostgreSQL connection using configured credentials."""
    return psycopg.connect(
        host="localhost",
        port=5432,
        dbname="banking",
        user="agent",
        password="agent_dev_password",
        autocommit=False,
    )


# --- Data generation ---------------------------------------------------------

def generate_customers(count: int) -> list[dict]:
    customers = []
    for i in range(1, count + 1):
        first = fake.first_name()
        last = fake.last_name()
        customers.append({
            "customer_id": f"CUST-{1000 + i}",
            "first_name": first,
            "last_name": last,
            "email": f"{first.lower()}.{last.lower()}{i}@example.com",
            "enrollment_date": fake.date_between(start_date="-8y", end_date="-30d"),
        })
    return customers


def generate_accounts(customers: list[dict]) -> list[dict]:
    accounts = []
    acc_counter = 5000
    for customer in customers:
        n_accounts = random.randint(*ACCOUNTS_PER_CUSTOMER)
        types = random.sample(["checking", "savings", "credit_card"], k=min(n_accounts, 3))
        for acc_type in types:
            acc_counter += 1
            # Realistic starting balance distribution
            if acc_type == "checking":
                balance = random.randint(50000, 1500000)  # $500 - $15,000
            elif acc_type == "savings":
                balance = random.randint(100000, 8000000)  # $1,000 - $80,000
            else:  # credit_card — small negative balance
                balance = random.randint(-500000, -10000)
            accounts.append({
                "account_id": f"ACC-{acc_counter}",
                "customer_id": customer["customer_id"],
                "account_type": acc_type,
                "balance_cents": balance,
                "currency": "USD",
                "opened_date": customer["enrollment_date"] + timedelta(days=random.randint(0, 60)),
                "status": "active",
            })
    return accounts


def generate_transactions(accounts: list[dict]) -> list[dict]:
    transactions = []
    now = datetime.now(timezone.utc)
    for account in accounts:
        n_txns = random.randint(*TRANSACTIONS_PER_ACCOUNT)
        for i in range(n_txns):
            category, min_amt, max_amt, merchants = random.choice(TRANSACTION_CATEGORIES)
            amount = random.randint(min_amt, max_amt)
            merchant = random.choice(merchants)
            timestamp = now - timedelta(days=random.randint(0, 60), hours=random.randint(0, 23))
            transactions.append({
                "transaction_id": f"TXN-{account['account_id']}-{i:04d}",
                "account_id": account["account_id"],
                "timestamp": timestamp,
                "amount_cents": amount,
                "merchant": merchant,
                "category": category,
                "status": "posted",
            })
    return transactions


# --- Database loading --------------------------------------------------------

def truncate_existing(conn: psycopg.Connection) -> None:
    """Wipe existing data — script is idempotent on re-run."""
    with conn.cursor() as cur:
        cur.execute("TRUNCATE transactions, accounts, customers RESTART IDENTITY CASCADE")
    conn.commit()
    logger.info("Truncated existing tables")


def insert_customers(conn: psycopg.Connection, customers: list[dict]) -> None:
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO customers (customer_id, first_name, last_name, email, enrollment_date)
            VALUES (%(customer_id)s, %(first_name)s, %(last_name)s, %(email)s, %(enrollment_date)s)
            """,
            customers,
        )
    conn.commit()
    logger.info("Inserted %d customers", len(customers))


def insert_accounts(conn: psycopg.Connection, accounts: list[dict]) -> None:
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO accounts (
                account_id, customer_id, account_type, balance_cents,
                currency, opened_date, status
            )
            VALUES (
                %(account_id)s, %(customer_id)s, %(account_type)s, %(balance_cents)s,
                %(currency)s, %(opened_date)s, %(status)s
            )
            """,
            accounts,
        )
    conn.commit()
    logger.info("Inserted %d accounts", len(accounts))


def insert_transactions(conn: psycopg.Connection, transactions: list[dict]) -> None:
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO transactions (
                transaction_id, account_id, timestamp, amount_cents,
                merchant, category, status
            )
            VALUES (
                %(transaction_id)s, %(account_id)s, %(timestamp)s, %(amount_cents)s,
                %(merchant)s, %(category)s, %(status)s
            )
            """,
            transactions,
        )
    conn.commit()
    logger.info("Inserted %d transactions", len(transactions))


# --- Main --------------------------------------------------------------------

def main() -> None:
    logger.info("Generating synthetic banking data...")
    customers = generate_customers(NUM_CUSTOMERS)
    accounts = generate_accounts(customers)
    transactions = generate_transactions(accounts)

    logger.info(
        "Generated: %d customers, %d accounts, %d transactions",
        len(customers), len(accounts), len(transactions),
    )

    with get_pg_connection() as conn:
        truncate_existing(conn)
        insert_customers(conn, customers)
        insert_accounts(conn, accounts)
        insert_transactions(conn, transactions)

    logger.info("Seed complete.")
    logger.info("Sample customer IDs: %s", [c["customer_id"] for c in customers[:5]])
    logger.info("Sample account IDs: %s", [a["account_id"] for a in accounts[:5]])


if __name__ == "__main__":
    main()