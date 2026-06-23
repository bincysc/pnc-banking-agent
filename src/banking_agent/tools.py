"""
Banking domain tools the agent can invoke.

Each tool is a Pydantic-validated function with a structured return type.
Tool implementations delegate to repositories for data access — they do not
contain SQL or MongoDB queries directly.
"""

import logging
from typing import Any
from banking_agent.rag.retrieval import hybrid_retrieve
from pydantic import BaseModel, Field, ValidationError

from banking_agent.repositories import (
    AccountRepository,
    TransactionRepository,
)

logger = logging.getLogger(__name__)


# --- Repository singletons (constructed lazily on first tool call) ---------

_account_repo: AccountRepository | None = None
_transaction_repo: TransactionRepository | None = None


def _accounts() -> AccountRepository:
    global _account_repo
    if _account_repo is None:
        _account_repo = AccountRepository()
    return _account_repo


def _transactions() -> TransactionRepository:
    global _transaction_repo
    if _transaction_repo is None:
        _transaction_repo = TransactionRepository()
    return _transaction_repo


# --- Input validation schemas ---------------------------------------------

class GetAccountBalanceInput(BaseModel):
    account_id: str = Field(..., min_length=1, description="The account identifier")

class LookupPolicyInput(BaseModel):
    """Input schema for the lookup_policy tool."""
    query: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Natural-language question about bank policies, fees, "
                    "or procedures (e.g. 'wire transfer limits', 'how to "
                    "report fraud').",
    )
    max_results: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Maximum number of policy excerpts to return (1-5).",
    )

class SearchTransactionsInput(BaseModel):
    account_id: str = Field(..., min_length=1, description="The account identifier")
    limit: int = Field(default=10, ge=1, le=50, description="Max transactions to return")


class SearchTransactionsByCategoryInput(BaseModel):
    account_id: str = Field(..., min_length=1)
    category: str = Field(..., min_length=1, description="Spending category")
    limit: int = Field(default=20, ge=1, le=100)


# --- Tool implementations -------------------------------------------------

def get_account_balance(arguments: dict[str, Any]) -> dict[str, Any]:
    """Retrieve the current balance for an account."""
    try:
        args = GetAccountBalanceInput.model_validate(arguments)
    except ValidationError as e:
        return {"error": "invalid_arguments", "details": e.errors()}

    account = _accounts().get_by_id(args.account_id)
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
    """Retrieve recent transactions for an account."""
    try:
        args = SearchTransactionsInput.model_validate(arguments)
    except ValidationError as e:
        return {"error": "invalid_arguments", "details": e.errors()}

    account = _accounts().get_by_id(args.account_id)
    if account is None:
        return {"error": "account_not_found", "account_id": args.account_id}

    transactions = _transactions().list_recent_for_account(args.account_id, args.limit)

    return {
        "account_id": args.account_id,
        "count": len(transactions),
        "transactions": [
            {
                "transaction_id": t["transaction_id"],
                "timestamp": t["timestamp"],
                "amount_dollars": t["amount_cents"] / 100,
                "merchant": t["merchant"],
                "category": t["category"],
                "status": t["status"],
            }
            for t in transactions
        ],
    }


def search_transactions_by_category(arguments: dict[str, Any]) -> dict[str, Any]:
    """Retrieve transactions for an account filtered by spending category."""
    try:
        args = SearchTransactionsByCategoryInput.model_validate(arguments)
    except ValidationError as e:
        return {"error": "invalid_arguments", "details": e.errors()}

    account = _accounts().get_by_id(args.account_id)
    if account is None:
        return {"error": "account_not_found", "account_id": args.account_id}

    transactions = _transactions().list_by_category(
        args.account_id, args.category, args.limit
    )

    total_cents = sum(t["amount_cents"] for t in transactions)

    return {
        "account_id": args.account_id,
        "category": args.category,
        "count": len(transactions),
        "total_dollars": total_cents / 100,
        "transactions": [
            {
                "timestamp": t["timestamp"],
                "amount_dollars": t["amount_cents"] / 100,
                "merchant": t["merchant"],
            }
            for t in transactions
        ],
    }

def lookup_policy(arguments: dict[str, Any]) -> dict[str, Any]:
    """
    Retrieve bank policy excerpts relevant to a natural-language query.

    Uses hybrid retrieval (vector + BM25 with RRF fusion) to surface the
    most relevant policy chunks. Returns the chunks with explicit citation
    metadata (document name, section title) so the LLM can ground its
    answer with specific references rather than paraphrasing freely.
    """
    try:
        args = LookupPolicyInput.model_validate(arguments)
    except ValidationError as e:
        return {"error": "invalid_arguments", "details": e.errors()}

    try:
        results = hybrid_retrieve(args.query, k=args.max_results)
    except Exception as e:
        logger.error("lookup_policy_failed query=%r error=%s", args.query, e)
        return {"error": "retrieval_failed", "message": str(e)}

    if not results:
        return {
            "query": args.query,
            "results": [],
            "message": "No relevant policy information found.",
        }

    return {
        "query": args.query,
        "results": [
            {
                "document": r.document_name,
                "section": r.section_title,
                "content": r.content,
                "relevance_score": round(r.score, 4),
            }
            for r in results
        ],
    }

# --- Converse-format tool schemas ---

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "toolSpec": {
            "name": "get_account_balance",
            "description": (
                "Retrieve the current balance for a specific account. Use when "
                "the customer asks about their balance. Returns the balance in "
                "dollars along with account type and status."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "account_id": {
                            "type": "string",
                            "description": "The account identifier (format ACC-NNNN)",
                        }
                    },
                    "required": ["account_id"],
                }
            },
        }
    },
    {
        "toolSpec": {
            "name": "lookup_policy",
            "description": (
                "Search the bank's policy knowledge base for information about "
                "wire transfer limits, account types, fees, fraud reporting, "
                "mobile deposit, and other policies. Use this when the customer "
                "asks about policies, procedures, fees, limits, or eligibility "
                "rules. Returns relevant policy excerpts with citations to the "
                "source document and section."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural-language question about a policy.",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of excerpts (default 3, max 5).",
                            "default": 3,
                            "minimum": 1,
                            "maximum": 5,
                        },
                    },
                    "required": ["query"],
                },
            },
        },
    },
    {
        "toolSpec": {
            "name": "search_transactions",
            "description": (
                "Retrieve recent transactions for an account, ordered "
                "most-recent-first. Use when the customer asks about recent "
                "spending or transaction history without a specific category."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "account_id": {
                            "type": "string",
                            "description": "The account identifier",
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
    {
        "toolSpec": {
            "name": "search_transactions_by_category",
            "description": (
                "Retrieve transactions for an account filtered by spending "
                "category. Returns transactions plus the category total in "
                "dollars. Use when the customer asks 'how much did I spend on "
                "X' where X is a category like groceries, gas, dining, travel, "
                "subscriptions, or utilities."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "account_id": {
                            "type": "string",
                            "description": "The account identifier",
                        },
                        "category": {
                            "type": "string",
                            "description": (
                                "Spending category. Valid values: groceries, "
                                "dining, gas, subscriptions, travel, utilities, "
                                "income, transfer."
                            ),
                        },
                        "limit": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 100,
                        },
                    },
                    "required": ["account_id", "category"],
                }
            },
        }
    },
]


# --- Dispatch table ---

TOOL_HANDLERS = {
    "get_account_balance": get_account_balance,
    "search_transactions": search_transactions,
    "search_transactions_by_category": search_transactions_by_category,
    "lookup_policy": lookup_policy,
}


def dispatch_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    handler = TOOL_HANDLERS.get(name)
    if handler is None:
        logger.warning("unknown_tool tool_name=%s", name)
        return {"error": "unknown_tool", "tool_name": name}

    logger.info("tool_dispatch tool=%s args=%s", name, arguments)
    result = handler(arguments)
    logger.info("tool_result tool=%s error=%s", name, result.get("error"))
    return result
