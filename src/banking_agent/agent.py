"""
LangGraph agent definition.

This is the production pattern: a state graph with two nodes — the agent node
(LLM reasoning) and the tools node (deterministic execution) — connected by a
conditional edge that loops until the model produces a response with no
tool calls or the turn limit is hit.
"""

import logging
from operator import add
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from banking_agent.bedrock_client import BedrockClient
from banking_agent.config import get_config
from banking_agent.tools import TOOL_SCHEMAS, dispatch_tool

logger = logging.getLogger(__name__)


# --- State schema ---------------------------------------------------------

class AgentState(TypedDict):
    """
    The shared state that flows through the graph.

    Messages are accumulated through the add_messages reducer — each node
    appends to the list rather than overwriting it. The turn counter prevents
    runaway loops.
    """
    messages: Annotated[list[dict[str, Any]], add]
    turn_count: int


# --- System prompt --------------------------------------------------------

SYSTEM_PROMPT = """You are a customer service agent for PNC Bank. You help customers with \
questions about their accounts and transactions.

Operating principles:
- Use the tools provided to look up account information rather than guessing.
- When you need an account_id and the customer has not provided one, ask for it.
- Format monetary amounts as dollars (e.g., "$4,875.23"), not cents.
- If a tool returns an error, explain it to the customer in plain language and \
ask for clarification if needed.
- You can only discuss banking topics. For non-banking questions, politely \
redirect the conversation.
- Never speculate about account information you have not verified through a tool.
"""


# --- Module-scoped resources ---

_bedrock = BedrockClient()


# --- Node implementations -------------------------------------------------

def agent_node(state: AgentState) -> dict[str, Any]:
    """
    The LLM reasoning step.

    Sends the conversation history to Bedrock and gets back either a text
    response (terminal) or one or more tool-use blocks (continue to tools node).
    """
    config = get_config()

    if state["turn_count"] >= config.agent_max_turns:
        # Hard cap reached. Return a synthetic message that ends the conversation
        # safely rather than letting the loop continue indefinitely.
        logger.warning(
            "agent_max_turns_exceeded turn_count=%d limit=%d",
            state["turn_count"],
            config.agent_max_turns,
        )
        return {
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "text": (
                                "I apologize — I have hit my reasoning limit on this question. "
                                "Please try rephrasing or breaking it into smaller parts."
                            )
                        }
                    ],
                }
            ],
            "turn_count": state["turn_count"] + 1,
        }

    response = _bedrock.converse(
        messages=state["messages"],
        system=[{"text": SYSTEM_PROMPT}],
        tools=TOOL_SCHEMAS,
    )

    assistant_message = response["output"]["message"]

    return {
        "messages": [assistant_message],
        "turn_count": state["turn_count"] + 1,
    }


def tools_node(state: AgentState) -> dict[str, Any]:
    """
    The deterministic tool execution step.

    Reads the most recent assistant message, extracts the tool-use blocks,
    dispatches each one, and constructs a user-role message containing the
    tool results that the agent node will see on its next invocation.
    """
    last_message = state["messages"][-1]
    tool_use_blocks = [
        block for block in last_message["content"]
        if "toolUse" in block
    ]

    tool_result_blocks = []
    for block in tool_use_blocks:
        tool_use = block["toolUse"]
        tool_name = tool_use["name"]
        tool_args = tool_use["input"]
        tool_use_id = tool_use["toolUseId"]

        result = dispatch_tool(tool_name, tool_args)

        tool_result_blocks.append(
            {
                "toolResult": {
                    "toolUseId": tool_use_id,
                    "content": [{"json": result}],
                    "status": "error" if "error" in result else "success",
                }
            }
        )

    return {
        "messages": [{"role": "user", "content": tool_result_blocks}],
        "turn_count": state["turn_count"],
    }


# --- Routing function -----------------------------------------------------

def should_continue(state: AgentState) -> str:
    """
    Decide whether to loop to the tools node or end the graph.

    Inspects the most recent assistant message. If it contains any tool-use
    blocks, we route to the tools node. Otherwise the model has produced a
    final text response and we terminate.
    """
    last_message = state["messages"][-1]
    if last_message.get("role") != "assistant":
        return "end"

    has_tool_use = any("toolUse" in block for block in last_message.get("content", []))
    return "tools" if has_tool_use else "end"


# --- Graph construction ---------------------------------------------------

def build_graph() -> Any:
    """
    Construct and compile the agent state graph.

    The topology:
        START -> agent -> (tools -> agent | END)

    The agent node always runs first. Its output either contains tool-use
    blocks (route to tools) or does not (terminate). The tools node always
    routes back to the agent for the next reasoning turn.
    """
    builder = StateGraph(AgentState)

    builder.add_node("agent", agent_node)
    builder.add_node("tools", tools_node)

    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "end": END},
    )
    builder.add_edge("tools", "agent")

    return builder.compile()
