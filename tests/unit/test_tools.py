"""
Unit tests for tool functions.

These tests use mocked repositories — no database connection is required.
Integration tests exercising the real database live in tests/integration/.
"""

from unittest.mock import MagicMock

import pytest

from banking_agent.tools import (
    dispatch_tool,
    get_account_balance,
    search_transactions,
    search_transactions_by_category,
)


@pytest.fixture(autouse=True)
def mock_repositories(monkeypatch):
    """
    Replace real repositories with mocks for unit tests.

    The repositories in this codebase return plain dicts, so the fake
    data here is dicts as well. The repositories themselves are MagicMock
    so we can stub their method calls.
    """
    # --- Fake account (plain dict, matches what real repo returns) ---
    fake_account = {
        "account_id": "ACC-5001",
        "customer_id": "CUST-1001",
        "balance_cents": 563575,
        "account_type": "checking",
        "currency": "USD",
        "status": "active",
    }

    # --- Fake transactions ---
    def make_txn(txn_id, amount, category, merchant):
        return {
            "transaction_id": txn_id,
            "account_id": "ACC-5001",
            "amount_cents": amount,
            "merchant": merchant,
            "category": category,
            "status": "posted",
            "timestamp": "2026-06-01T12:00:00Z",
        }

    fake_txns = [
        make_txn("TXN-ACC-5001-0000", -7535, "gas", "Shell"),
        make_txn("TXN-ACC-5001-0001", -4444, "dining", "Chipotle"),
        make_txn("TXN-ACC-5001-0002", -629, "subscriptions", "Netflix"),
        make_txn("TXN-ACC-5001-0003", 792683, "income", "Direct-Deposit"),
        make_txn("TXN-ACC-5001-0004", -1714, "subscriptions", "Spotify"),
    ]
    fake_grocery_txns = [
        make_txn("TXN-ACC-5001-0010", -8523, "groceries", "Whole Foods"),
        make_txn("TXN-ACC-5001-0011", -3210, "groceries", "Trader Joes"),
    ]

    # --- Fake repositories ---
    fake_account_repo = MagicMock()
    fake_account_repo.get_by_id.side_effect = lambda aid: (
        fake_account if aid == "ACC-5001" else None
    )

    fake_txn_repo = MagicMock()

    def fake_get_by_account(account_id, limit=10):
        return fake_txns[:limit]

    def fake_get_by_category(account_id, category, limit=50):
        if category == "groceries":
            return fake_grocery_txns
        return []

    fake_txn_repo.list_recent_for_account.side_effect = fake_get_by_account
    fake_txn_repo.list_by_category.side_effect = fake_get_by_category

    from banking_agent import tools
    monkeypatch.setattr(tools, "_accounts", lambda: fake_account_repo)
    monkeypatch.setattr(tools, "_transactions", lambda: fake_txn_repo)


# --- get_account_balance tests -------------------------------------------

class TestGetAccountBalance:

    def test_returns_balance_for_valid_account(self):
        result = get_account_balance({"account_id": "ACC-5001"})
        assert "error" not in result
        assert result["account_id"] == "ACC-5001"
        assert result["balance_dollars"] == 5635.75

    def test_returns_error_for_unknown_account(self):
        result = get_account_balance({"account_id": "ACC-99999"})
        assert "error" in result

    def test_returns_error_for_missing_account_id(self):
        result = get_account_balance({})
        assert "error" in result

    def test_returns_error_for_empty_account_id(self):
        result = get_account_balance({"account_id": ""})
        assert "error" in result


# --- search_transactions tests -------------------------------------------

class TestSearchTransactions:

    def test_returns_transactions_for_valid_account(self):
        result = search_transactions({"account_id": "ACC-5001"})
        assert "error" not in result
        assert "transactions" in result
        assert len(result["transactions"]) > 0

    def test_respects_limit_parameter(self):
        result = search_transactions({"account_id": "ACC-5001", "limit": 3})
        assert "error" not in result
        assert len(result["transactions"]) == 3

    def test_returns_error_for_unknown_account(self):
        result = search_transactions({"account_id": "ACC-99999"})
        assert "error" in result

    def test_returns_error_for_limit_out_of_range(self):
        result = search_transactions({"account_id": "ACC-5001", "limit": 9999})
        assert "error" in result


# --- search_transactions_by_category tests --------------------------------

class TestSearchTransactionsByCategory:

    def test_returns_category_total_for_valid_account(self):
        result = search_transactions_by_category(
            {"account_id": "ACC-5001", "category": "groceries"}
        )
        assert "error" not in result
        assert result["category"] == "groceries"
        assert result["count"] == 2

    def test_returns_zero_count_for_category_with_no_transactions(self):
        # The tool does not validate categories — it returns zero counts
        # for unknown ones. Real validation happens at the LLM tool-calling
        # layer via the JSON schema. This is intentional: tools should be
        # tolerant of arbitrary inputs and let upstream validation enforce
        # the schema.
        result = search_transactions_by_category(
            {"account_id": "ACC-5001", "category": "not_a_real_category"}
        )
        assert "error" not in result
        assert result["count"] == 0


# --- dispatch_tool tests --------------------------------------------------

class TestDispatchTool:

    def test_dispatches_known_tool(self):
        result = dispatch_tool("get_account_balance", {"account_id": "ACC-5001"})
        assert "error" not in result
        assert result["account_id"] == "ACC-5001"

    def test_returns_error_for_unknown_tool(self):
        result = dispatch_tool("nonexistent_tool", {})
        assert "error" in result
