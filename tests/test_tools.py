"""
Tests for the banking tool implementations.

These tests exercise the full path: tool function → repository → real
PostgreSQL → Redis cache. They require the docker compose stack to be running
and seed_data.py to have been executed.

In a production codebase, these would be integration tests in a separate
directory (tests/integration/) with a marker that skips them when the
database is unavailable. For this project the simpler arrangement is fine.
"""

from banking_agent.tools import (
    dispatch_tool,
    get_account_balance,
    search_transactions,
    search_transactions_by_category,
)


class TestGetAccountBalance:
    def test_returns_balance_for_valid_account(self):
        result = get_account_balance({"account_id": "ACC-5001"})

        assert "error" not in result
        assert result["account_id"] == "ACC-5001"
        assert result["currency"] == "USD"
        assert result["status"] == "active"
        assert "balance_dollars" in result
        assert isinstance(result["balance_dollars"], (int, float))

    def test_returns_error_for_unknown_account(self):
        result = get_account_balance({"account_id": "ACC-99999"})

        assert result["error"] == "account_not_found"
        assert result["account_id"] == "ACC-99999"

    def test_returns_error_for_missing_account_id(self):
        result = get_account_balance({})

        assert result["error"] == "invalid_arguments"

    def test_returns_error_for_empty_account_id(self):
        result = get_account_balance({"account_id": ""})

        assert result["error"] == "invalid_arguments"


class TestSearchTransactions:
    def test_returns_transactions_for_valid_account(self):
        result = search_transactions({"account_id": "ACC-5001"})

        assert "error" not in result
        assert result["account_id"] == "ACC-5001"
        assert result["count"] > 0

    def test_respects_limit_parameter(self):
        result = search_transactions({"account_id": "ACC-5001", "limit": 3})

        assert result["count"] <= 3

    def test_returns_error_for_unknown_account(self):
        result = search_transactions({"account_id": "ACC-99999"})

        assert result["error"] == "account_not_found"

    def test_returns_error_for_limit_out_of_range(self):
        result = search_transactions({"account_id": "ACC-5001", "limit": 100})

        assert result["error"] == "invalid_arguments"


class TestSearchTransactionsByCategory:
    def test_returns_category_total_for_valid_account(self):
        result = search_transactions_by_category(
            {"account_id": "ACC-5001", "category": "groceries"}
        )

        assert "error" not in result
        assert result["category"] == "groceries"
        assert "total_dollars" in result
        assert "count" in result

    def test_returns_error_for_invalid_category(self):
        # Empty category fails validation
        result = search_transactions_by_category(
            {"account_id": "ACC-5001", "category": ""}
        )

        assert result["error"] == "invalid_arguments"


class TestDispatchTool:
    def test_dispatches_known_tool(self):
        result = dispatch_tool("get_account_balance", {"account_id": "ACC-5001"})

        assert "account_id" in result or "error" in result

    def test_returns_error_for_unknown_tool(self):
        result = dispatch_tool("nonexistent_tool", {})

        assert result["error"] == "unknown_tool"