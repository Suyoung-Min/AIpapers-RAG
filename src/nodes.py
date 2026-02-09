"""LangGraph 노드 로직 모듈.

Router, Retriever, Reranker, Generator 노드를 정의한다.
"""

import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from src.state import AgentState

load_dotenv()

CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"
COLLECTION_NAME = "arxiv_papers"

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def get_collection() -> chromadb.Collection:
    """ChromaDB 컬렉션을 가져온다."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_collection(name=COLLECTION_NAME)


# ── Router Node ──────────────────────────────────────────────────────────────

def router_node(state: AgentState) -> AgentState:
    """질문이 논문 검색이 필요한지, 일반 대화인지 판단한다."""
    question = state["question"]

    prompt = f"""다음 사용자 질문을 분석하여, 학술 논문 검색이 필요한 질문인지 판단하세요.

질문: {question}

- 논문 검색이 필요하면 "retrieve"를 출력하세요.
- 일반 대화(인사, 잡담 등)이면 "chat"을 출력하세요.

반드시 "retrieve" 또는 "chat" 중 하나만 출력하세요."""

    response = llm.invoke(prompt)
    route = response.content.strip().lower()

    if "retrieve" in route:
        route = "retrieve"
    else:
        route = "chat"

    return {
        **state,
        "route": route,
        "steps": state.get("steps", []) + [f"Router: {route}"],
    }


# ── Retriever Node ───────────────────────────────────────────────────────────

def retriever_node(state: AgentState) -> AgentState:
    """질문에서 키워드와 필터를 추출하고 벡터 검색을 수행한다."""
    question = state["question"]

    # LLM으로 검색 키워드 및 필터 추출
    extract_prompt = f"""다음 질문에서 학술 논문 검색에 사용할 정보를 추출하세요.

질문: {question}

다음 형식으로 응답하세요:
검색어: <벡터 검색에 사용할 영어 검색 쿼리>
연도필터: <특정 연도가 언급되면 해당 연도, 없으면 "없음">"""

    response = llm.invoke(extract_prompt)
    lines = response.content.strip().split("\n")

    search_query = question
    year_filter = None

    for line in lines:
        if line.startswith("검색어:"):
            search_query = line.replace("검색어:", "").strip()
        elif line.startswith("연도필터:"):
            year_val = line.replace("연도필터:", "").strip()
            if year_val != "없음" and year_val.isdigit():
                year_filter = year_val

    filters = {"year": year_filter} if year_filter else {}

    # ChromaDB 벡터 검색
    collection = get_collection()

    where_filter = None
    if year_filter:
        where_filter = {
            "published": {"$gte": f"{year_filter}-01-01"},
        }

    results = collection.query(
        query_texts=[search_query],
        n_results=20,
        where=where_filter if where_filter else None,
    )

    documents = []
    if results and results["documents"]:
        for i, doc in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i] if results["metadatas"] else {}
            distance = results["distances"][0][i] if results["distances"] else None
            documents.append({
                "content": doc,
                "metadata": meta,
                "distance": distance,
            })

    return {
        **state,
        "documents": documents,
        "filters": filters,
        "steps": state.get("steps", [])
        + [f"Retriever: '{search_query}' -> {len(documents)}개 문서 검색"],
    }


# ── Reranker Node ────────────────────────────────────────────────────────────

def reranker_node(state: AgentState) -> AgentState:
    """검색된 문서 중 질문과 가장 관련 있는 Top-5를 선정한다."""
    question = state["question"]
    documents = state["documents"]

    if not documents:
        return {
            **state,
            "steps": state.get("steps", []) + ["Reranker: 검색 결과 없음"],
        }

    # 문서 요약 리스트 생성
    doc_summaries = []
    for i, doc in enumerate(documents[:20]):
        title = doc["metadata"].get("title", "N/A")
        doc_summaries.append(f"[{i}] {title}\n{doc['content'][:300]}")

    docs_text = "\n\n".join(doc_summaries)

    rerank_prompt = f"""다음 질문과 검색된 논문 목록을 보고, 질문에 가장 관련 있는 논문 5편을 선택하세요.

질문: {question}

논문 목록:
{docs_text}

각 선택에 대해 다음 형식으로 응답하세요:
순위 1: [인덱스] - 선정 이유 (한 줄)
순위 2: [인덱스] - 선정 이유 (한 줄)
순위 3: [인덱스] - 선정 이유 (한 줄)
순위 4: [인덱스] - 선정 이유 (한 줄)
순위 5: [인덱스] - 선정 이유 (한 줄)"""

    response = llm.invoke(rerank_prompt)
    rerank_text = response.content.strip()

    # 선택된 인덱스 파싱
    selected_indices = []
    for line in rerank_text.split("\n"):
        line = line.strip()
        if not line:
            continue
        # "순위 N: [인덱스]" 형식에서 인덱스 추출
        try:
            bracket_start = line.index("[")
            bracket_end = line.index("]")
            idx = int(line[bracket_start + 1 : bracket_end])
            if 0 <= idx < len(documents):
                selected_indices.append(idx)
        except (ValueError, IndexError):
            continue

    # 상위 5개 문서 선택 (파싱 실패 시 거리 기반 상위 5개)
    if selected_indices:
        reranked = [documents[i] for i in selected_indices[:5]]
    else:
        reranked = documents[:5]

    return {
        **state,
        "documents": reranked,
        "steps": state.get("steps", [])
        + [f"Reranker: {len(reranked)}개 논문 선정\n{rerank_text}"],
    }


# ── Generator Node ───────────────────────────────────────────────────────────

def generator_node(state: AgentState) -> AgentState:
    """논문 리스트를 바탕으로 최종 답변을 생성한다."""
    question = state["question"]
    documents = state["documents"]

    if not documents:
        return {
            **state,
            "generation": "검색된 관련 논문이 없습니다. 다른 키워드로 질문해 주세요.",
            "steps": state.get("steps", []) + ["Generator: 문서 없음"],
        }

    # 문서 컨텍스트 구성
    context_parts = []
    for i, doc in enumerate(documents, 1):
        meta = doc["metadata"]
        title = meta.get("title", "N/A")
        url = meta.get("url", "N/A")
        published = meta.get("published", "N/A")
        authors = meta.get("authors", "N/A")
        context_parts.append(
            f"[{i}] 제목: {title}\n"
            f"    저자: {authors}\n"
            f"    발행일: {published}\n"
            f"    링크: {url}\n"
            f"    내용: {doc['content'][:500]}"
        )

    context = "\n\n".join(context_parts)

    gen_prompt = f"""당신은 학술 논문 전문가입니다. 아래 검색된 논문 정보를 바탕으로 사용자의 질문에 답변하세요.

질문: {question}

검색된 논문:
{context}

답변 지침:
1. 질문에 대해 논문들의 내용을 종합하여 답변하세요.
2. 각 논문을 인용할 때 [번호] 형식으로 출처를 표기하세요.
3. 답변 마지막에 참고 논문 목록을 링크와 함께 제공하세요.
4. 한국어로 답변하세요."""

    response = llm.invoke(gen_prompt)

    return {
        **state,
        "generation": response.content,
        "steps": state.get("steps", []) + ["Generator: 답변 생성 완료"],
    }


# ── Chat Node (일반 대화) ────────────────────────────────────────────────────

def chat_node(state: AgentState) -> AgentState:
    """일반 대화에 대한 응답을 생성한다."""
    question = state["question"]

    prompt = f"""당신은 학술 논문 검색을 도와주는 친절한 AI 어시스턴트입니다.
사용자의 일반적인 질문이나 인사에 적절히 응답하세요.
논문 검색이 필요한 경우 논문에 대해 질문해달라고 안내하세요.
한국어로 답변하세요.

사용자: {question}"""

    response = llm.invoke(prompt)

    return {
        **state,
        "generation": response.content,
        "steps": state.get("steps", []) + ["Chat: 일반 대화 응답"],
    }
