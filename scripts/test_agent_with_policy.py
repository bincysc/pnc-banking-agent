"""
End-to-end test of the agent with the policy lookup tool.

Drives four policy questions through the LangGraph agent and prints the
agent's responses. The agent should call lookup_policy, receive grounded
excerpts, and produce cited answers.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from banking_agent.agent import build_graph

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def print_response(message: dict) -> None:
    """Print only the text content of an assistant message."""
    for block in message.get("content", []):
        if "text" in block:
            print(block["text"])


def ask(graph, question: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"USER: {question}")
    print("=" * 70)

    # Use the same message format as run_local.py: role + content with text block.
    messages = [{"role": "user", "content": [{"text": question}]}]
    result = graph.invoke({"messages": messages, "turn_count": 0})

    # The final assistant message is the answer.
    final_message = result["messages"][-1]
    if final_message.get("role") == "assistant":
        print("\nAGENT:")
        print_response(final_message)


def main() -> None:
    graph = build_graph()

    ask(graph, "What is the daily wire transfer limit?")
    ask(graph, "How do I report a fraudulent charge on my debit card?")
    ask(graph, "What's the fee for an outgoing international wire?")
    ask(graph, "Can I deposit a $7,000 check on the mobile app?")


if __name__ == "__main__":
    main()