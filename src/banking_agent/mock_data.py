"""
Mock banking data for local development.

This is the in-memory representation of what the production agent will read
from PostgreSQL on Day 10. The data is synthetic — no real customer
information. The structure mirrors the production schema so swapping from
this mock to the real database on Day 10 is a one-place change.
"""

from datetime import datetime, timedelta
from decimal import Decimal

# Mock customers — keyed by customer_id.
# In production this is a SELECT from the customers table in PostgreSQL.
CUSTOMERS = {
    "CUST-1001": {
        "customer_id": "CUST-1001",
        "first_name": "Margaret",
        "last_name": "Chen",
        "email": "margaret.chen@example.com",
        "enrollment_date": "2019-03-14",
    },
    "CUST-1002": {
        "customer_id": "CUST-1002",
        "first_name": "James",
        "last_name": "Patel",
        "email": "james.patel@example.com",
        "enrollment_date": "2021-08-22",
    },
}

# Mock accounts — keyed by account_id, with customer_id as foreign key.
ACCOUNTS = {
    "ACC-5001": {
        "account_id": "ACC-5001",
        "customer_id": "CUST-1001",
        "account_type": "checking",
        "balance_cents": 487523,  # $4,875.23 — stored as integer cents
        "currency": "USD",
        "opened_date": "2019-03-14",
        "status": "active",
    },
    "ACC-5002": {
        "account_id": "ACC-5002",
        "customer_id": "CUST-1001",
        "account_type": "savings",
        "balance_cents": 1245000,  # $12,450.00
        "currency": "USD",
        "opened_date": "2019-03-14",
        "status": "active",
    },
    "ACC-5003": {
        "account_id": "ACC-5003",
        "customer_id": "CUST-1002",
        "account_type": "checking",
        "balance_cents": 92034,
        "currency": "USD",
        "opened_date": "2021-08-22",
        "status": "active",
    },
}

# Mock transactions — list of dicts ordered most-recent-first.
# In production this is a SELECT from transactions filtered by account_id,
# ordered by transaction_timestamp DESC.
_today = datetime(2026, 6, 18)


def _txn(account_id: str, days_ago: int, amount_cents: int, merchant: str, category: str) -> dict:
    """Helper to construct a transaction with consistent shape."""
    return {
        "transaction_id": f"TXN-{account_id}-{days_ago:03d}",
        "account_id": account_id,
        "timestamp": (_today - timedelta(days=days_ago)).isoformat(),
        "amount_cents": amount_cents,  # positive = credit, negative = debit
        "merchant": merchant,
        "category": category,
        "status": "posted",
    }


TRANSACTIONS = [
    _txn("ACC-5001", 0, -4523, "Whole Foods", "groceries"),
    _txn("ACC-5001", 1, -1850, "Shell", "gas"),
    _txn("ACC-5001", 2, -2200, "Netflix", "subscriptions"),
    _txn("ACC-5001", 3, 250000, "ACH-Payroll", "income"),
    _txn("ACC-5001", 5, -8945, "Delta Airlines", "travel"),
    _txn("ACC-5001", 7, -3200, "Whole Foods", "groceries"),
    _txn("ACC-5001", 10, -1200, "Starbucks", "dining"),
    _txn("ACC-5002", 14, 50000, "Transfer-from-checking", "transfer"),
    _txn("ACC-5003", 1, -3200, "Trader Joes", "groceries"),
    _txn("ACC-5003", 2, 180000, "ACH-Payroll", "income"),
]


def get_customer(customer_id: str) -> dict | None:
    return CUSTOMERS.get(customer_id)


def get_accounts_for_customer(customer_id: str) -> list[dict]:
    return [acc for acc in ACCOUNTS.values() if acc["customer_id"] == customer_id]


def get_account(account_id: str) -> dict | None:
    return ACCOUNTS.get(account_id)


def get_transactions_for_account(account_id: str, limit: int = 10) -> list[dict]:
    matching = [txn for txn in TRANSACTIONS if txn["account_id"] == account_id]
    return matching[:limit]