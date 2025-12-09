from __future__ import annotations

import logging

from langgraph.graph import END, StateGraph

from .handlers import (
    fallback_handler,
    finalize_node,
    intent_recognition_node,
    router_node,
    search_tool_handler,
    sentiment_handler,
    summarize_handler,
)
from .models import AgentState

logger = logging.getLogger(__name__)


def route_next(state: AgentState) -> str:
    """
    Определяем следующую ноду исходя из очереди интентов.
    Когда интенты закончились — переходим в finalize.
    """
    if state.next_intent_index >= len(state.detected_intents):
        return "finalize"
    return state.detected_intents[state.next_intent_index]


def build_graph():
    """Собираем и компилируем граф агента."""
    graph = StateGraph(AgentState)
    graph.add_node("intent_recognition", intent_recognition_node)
    graph.add_node("router", router_node)
    graph.add_node("search", search_tool_handler)
    graph.add_node("summarize", summarize_handler)
    graph.add_node("sentiment", sentiment_handler)
    graph.add_node("fallback", fallback_handler)
    graph.add_node("finalize", finalize_node)

    graph.set_entry_point("intent_recognition")
    graph.add_edge("intent_recognition", "router")

    graph.add_conditional_edges(
        "router",
        route_next,
        {
            "search": "search",
            "summarize": "summarize",
            "sentiment": "sentiment",
            "none": "fallback",
            "finalize": "finalize",
        },
    )

    for node_name in ("search", "summarize", "sentiment", "fallback"):
        graph.add_edge(node_name, "router")

    graph.add_edge("finalize", END)
    compiled = graph.compile()
    logger.info("Граф агента собран.")
    return compiled
