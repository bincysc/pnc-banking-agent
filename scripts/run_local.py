"""
Local entry point for interactive agent conversations.

Run with: python scripts/run_local.py

Type your questions at the prompt. Type 'quit' or Ctrl+C to exit.
"""

import logging
import sys
from pathlib import Path

# Make the src/ layout importable when running from the project root.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from banking_agent.agent import build_graph
from banking_agent.bedrock_client import BedrockClient
from banking_agent.config import get_config


def setup_logging() -> None:
    """Configure logging with a format that surfaces structured fields readably."""
    config = get_config()
    logging.basicConfig(
        level=config.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def print_response(message: dict) -> None:
    """Pretty-print the assistant's text response, stripping any non-text blocks."""
    for block in message.get("content", []):
        if "text" in block:
            print(f"\nAgent: {block['text']}\n")


def main() -> None:
    setup_logging()
    print("PNC Banking Agent — local development mode")
    print("Type your questions. Type 'quit' to exit.\n")

    graph = build_graph()
    messages: list = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if user_input.lower() in {"quit", "exit"}:
            print("Goodbye.")
            break

        if not user_input:
            continue

        messages.append({"role": "user", "content": [{"text": user_input}]})

        result = graph.invoke({"messages": messages, "turn_count": 0})

        # Update our local message history with everything the graph produced.
        messages = result["messages"]

        # Print the final assistant message.
        final_message = messages[-1]
        if final_message.get("role") == "assistant":
            print_response(final_message)


if __name__ == "__main__":
    main()