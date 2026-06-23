"""
Repository layer — the data access boundary.

Each repository encapsulates queries for one domain entity. Code outside this
package depends on the repository interfaces, not on the underlying storage.
This is the pattern that keeps SQL out of the tool layer and makes the data
sources swappable.
"""

from banking_agent.repositories.account_repository import AccountRepository
from banking_agent.repositories.conversation_repository import ConversationRepository
from banking_agent.repositories.transaction_repository import TransactionRepository

__all__ = [
    "AccountRepository",
    "ConversationRepository",
    "TransactionRepository",
]
