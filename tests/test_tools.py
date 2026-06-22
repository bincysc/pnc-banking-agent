"""
Tests for the banking tool implementations.

Tools are the most test-worthy component of the agent because they have
deterministic logic over the data layer. The agent itself depends on the
non-deterministic LLM and is best validated through integration tests rather
than unit tests.
"""

import pytest

from banking_agent.tools import (
    dispatch_tool,
    get_account_balance,
    search_transactions,
)


class TestGetAccountBalance:
    def test_returns_balance_for_valid_account(self):
        result = get_account_balance({"account_id": "ACC-5001"})

        assert result["account_id"] == "ACC-5001"
        assert result["account_type"] == "checking"
        assert result["balance_dollars"] == 4875.23
        assert result["currency"] == "USD"
        assert result["status"] == "active"
        assert "error" not in result

    def test_returns_error_for_unknown_account(self):
        result = get_account_balance({"account_id": "ACC-9999"})

        assert result["error"] == "account_not_found"
        assert result["account_id"] == "ACC-9999"

    def test_returns_error_for_missing_account_id(self):
        result = get_account_balance({})

        assert result["error"] == "invalid_arguments"
        assert "details" in result

    def test_returns_error_for_empty_account_id(self):
        result = get_account_balance({"account_id": ""})

        assert result["error"] == "invalid_arguments"


class TestSearchTransactions:
    def test_returns_transactions_for_valid_account(self):
        result = search_transactions({"account_id": "ACC-5001"})

        assert result["account_id"] == "ACC-5001"
        assert result["count"] > 0
        assert len(result["transactions"]) == result["count"]
        assert "error" not in result

    def test_respects_limit_parameter(self):
        result = search_transactions({"account_id": "ACC-5001", "limit": 3})

        assert result["count"] <= 3

    def test_returns_error_for_unknown_account(self):
        result = search_transactions({"account_id": "ACC-9999"})

        assert result["error"] == "account_not_found"

    def test_returns_error_for_limit_out_of_range(self):
        result = search_transactions({"account_id": "ACC-5001", "limit": 100})

        assert result["error"] == "invalid_arguments"

    def test_transactions_have_required_fields(self):
        result = search_transactions({"account_id": "ACC-5001"})

        for txn in result["transactions"]:
            assert "transaction_id" in txn
            assert "timestamp" in txn
            assert "amount_dollars" in txn
            assert "merchant" in txn
            assert "category" in txn


class TestDispatchTool:
    def test_dispatches_known_tool(self):
        result = dispatch_tool("get_account_balance", {"account_id": "ACC-5001"})

        assert result["account_id"] == "ACC-5001"

    def test_returns_error_for_unknown_tool(self):
        result = dispatch_tool("nonexistent_tool", {})

        assert result["error"] == "unknown_tool"
        assert result["tool_name"] == "nonexistent_tool"