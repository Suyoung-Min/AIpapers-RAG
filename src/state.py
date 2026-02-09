"""LangGraph 상태 정의 모듈."""

from typing import Any

from langgraph.graph import MessagesState
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """RAG 에이전트의 상태를 정의한다.

    Attributes:
        question: 사용자의 질문.
        documents: 검색된 문서 리스트.
        filters: 추출된 필터 조건 (연도 등).
        generation: 최종 생성된 답변.
        steps: 워크플로우 진행 단계 기록.
        route: 라우팅 결과 ('retrieve' 또는 'chat').
    """

    question: str
    documents: list[dict[str, Any]]
    filters: dict[str, Any]
    generation: str
    steps: list[str]
    route: str
