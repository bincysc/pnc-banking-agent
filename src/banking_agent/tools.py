"""
Banking domain tools the agent can invoke.

Each tool is a Pydantic-validated function with a structured return type.
The schema declared in TOOL_SCHEMAS is the Converse-format tool specification
the model sees; the implementation in TOOL_HANDLERS is what executes when the
model emits a tool-use block.
"""

import logging
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from banking_agent import mock_data

logger = logging.getLogger(__name__)


# --- Input validation schemas ---------------------------------------------

class GetAccountBalanceInput(BaseModel):
    """Arguments for get_account_balance."""
    account_id: str = Field(..., min_length=1, description="The account identifier")


class SearchTransactionsInput(BaseModel):
    """Arguments for search_transactions."""
    account_id: str = Field(..., min_length=1, description="The account identifier")
    limit: int = Field(default=10, ge=1, le=50, description="Max transactions to return")


# --- Tool implementations -------------------------------------------------

def get_account_balance(arguments: dict[str, Any]) -> dict[str, Any]:
    """
    Retrieve the current balance for an account.

    Returns a structured response dict. Errors are returned as data, not raised,
    so the agent can reason about the failure and respond gracefully.
    """
    try:
        args = GetAccountBalanceInput.model_validate(arguments)
    except ValidationError as e:
        return {"error": "invalid_arguments", "details": e.errors()}

    account = mock_data.get_account(args.account_id)
    if account is None:
        return {"error": "account_not_found", "account_id": args.account_id}

    return {
        "account_id": account["account_id"],
        "account_type": account["account_type"],
        "balance_dollars": account["balance_cents"] / 100,
        "currency": account["currency"],
        "status": account["status"],
    }


def search_transactions(arguments: dict[str, Any]) -> dict[str, Any]:
    """
    Retrieve recent transactions for an account, ordered most-recent-first.
    """
    try:
        args = SearchTransactionsInput.model_validate(arguments)
    except ValidationError as e:
        return {"error": "invalid_arguments", "details": e.errors()}

    account = mock_data.get_account(args.account_id)
    if account is None:
        return {"error": "account_not_found", "account_id": args.account_id}

    transactions = mock_data.get_transactions_for_account(args.account_id, limit=args.limit)

    return {
        "account_id": args.account_id,
        "count": len(transactions),
        "transactions": [
            {
                "transaction_id": txn["transaction_id"],
                "timestamp": txn["timestamp"],
                "amount_dollars": txn["amount_cents"] / 100,
                "merchant": txn["merchant"],
                "category": txn["category"],
                "status": txn["status"],
            }
            for txn in transactions
        ],
    }


# --- Converse-format tool schemas the model sees ---

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "toolSpec": {
            "name": "get_account_balance",
            "description": (
                "Retrieve the current balance for a specific account. Use this tool when the "
                "customer asks how much money is in an account, or asks about their current "
                "balance. Returns the balance in dollars along with account type and status."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "account_id": {
                            "type": "string",
                            "description": "The account identifier, in the format ACC-NNNN",
                        }
                    },
                    "required": ["account_id"],
                }
            },
        }
    },
    {
        "toolSpec": {
            "name": "search_transactions",
            "description": (
                "Retrieve recent transactions for an account, ordered most-recent-first. Use "
                "this tool when the customer asks about their spending, recent purchases, "
                "transaction history, or specific transactions. Returns up to 'limit' "
                "transactions with timestamp, amount, merchant, and category."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "account_id": {
                            "type": "string",
                            "description": "The account identifier, in the format ACC-NNNN",
                        },
                        "limit": {
                            "type": "integer",
                            "description": (
                                "Maximum number of transactions to return "
                                "(default 10, max 50)"
                            ),
                            "minimum": 1,
                            "maximum": 50,
                        },
                    },
                    "required": ["account_id"],
                }
            },
        }
    },
]


# --- Dispatch table mapping tool name to handler function ---

TOOL_HANDLERS = {
    "get_account_balance": get_account_balance,
    "search_transactions": search_transactions,
}


def dispatch_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """
    Look up and invoke the named tool with the supplied arguments.

    This is the indirection layer between the model's tool-use output and the
    function implementations. The model emits a tool name string and an
    arguments dict; this function resolves that to the right handler and
    returns the structured result.
    """
    handler = TOOL_HANDLERS.get(name)
    if handler is None:
        logger.warning("unknown_tool tool_name=%s", name)
        return {"error": "unknown_tool", "tool_name": name}

    logger.info("tool_dispatch tool=%s args=%s", name, arguments)
    result = handler(arguments)
    logger.info("tool_result tool=%s error=%s", name, result.get("error"))
    return result
