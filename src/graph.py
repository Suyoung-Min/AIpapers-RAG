"""LangGraph 워크플로우 구성 모듈."""

from langgraph.graph import END, StateGraph

from src.nodes import (
    chat_node,
    generator_node,
    reranker_node,
    retriever_node,
    router_node,
)
from src.state import AgentState


def route_decision(state: AgentState) -> str:
    """Router 결과에 따라 다음 노드를 결정한다."""
    if state["route"] == "retrieve":
        return "retrieve"
    return "chat"


def build_graph() -> StateGraph:
    """RAG 워크플로우 그래프를 구성한다.

    Flow:
        Router -> (retrieve) -> Retriever -> Reranker -> Generator -> END
        Router -> (chat) -> Chat -> END
    """
    workflow = StateGraph(AgentState)

    # 노드 추가
    workflow.add_node("router", router_node)
    workflow.add_node("retrieve", retriever_node)
    workflow.add_node("rerank", reranker_node)
    workflow.add_node("generate", generator_node)
    workflow.add_node("chat", chat_node)

    # 엣지 구성
    workflow.set_entry_point("router")

    workflow.add_conditional_edges(
        "router",
        route_decision,
        {
            "retrieve": "retrieve",
            "chat": "chat",
        },
    )

    workflow.add_edge("retrieve", "rerank")
    workflow.add_edge("rerank", "generate")
    workflow.add_edge("generate", END)
    workflow.add_edge("chat", END)

    return workflow.compile()
