1. 프로젝트 개요
목표: arXiv 논문 초록 데이터를 활용하여 사용자의 질문에 답변하고 관련 논문을 추천하는 에이전틱 RAG 시스템 구축.

핵심 도구: Python, LangGraph, LangChain, ChromaDB, arxiv API.

핵심 로직: 사용자의 의도 파악(Router) -> 검색 및 필터링 -> 결과 재정렬(Rerank) -> 최종 답변 생성.

2. 기술 스택 및 환경 설정
설치된 anaconda langgraph 가상환경 기반으로

Language: Python 3.10+

Orchestration: LangGraph, LangChain

Vector DB: ChromaDB (Local)

LLM: OpenAI GPT-4o-mini 또는 Claude 3.5 Sonnet

Data Source: arxiv python library

3. 디렉토리 구조 (Proposed)
Plaintext
arxiv-rag-project/
├── data/               # 수집된 JSON 데이터 저장
├── src/
│   ├── ingestion.py    # arXiv API 수집 및 DB 저장
│   ├── state.py        # LangGraph State 정의
│   ├── nodes.py        # 각 노드(Retrieve, Rerank, Generate) 로직
│   ├── graph.py        # LangGraph 워크플로우 구성
│   └── main.py         # CLI 실행 진입점
├── .env                # API 키 관리
└── requirements.txt
4. 상세 구현 가이드 (클로드에게 지시할 내용)
Step 1: 데이터 수집 및 인덱싱 (ingestion.py)
arxiv 라이브러리를 사용하여 cs.CL 카테고리 최신 논문 1,000개를 수집한다.

수집 간격은 API 제한을 고려하여 3초의 딜레이를 둔다.

논문의 title + abstract를 합쳐 임베딩하고, 메타데이터(url, categories, published)와 함께 ChromaDB에 저장한다.

Step 2: LangGraph 상태 관리 (state.py)
AgentState를 정의한다. (필수 필드: question, documents, filters, generation, steps)

Step 3: 노드 및 엣지 구성 (nodes.py, graph.py)
Router Node: 질문이 검색이 필요한지, 일반 대화인지 판단한다.

Retriever Node: 질문에서 키워드와 필터(연도 등)를 추출해 벡터 검색을 수행한다.

Reranker Node: 검색된 결과 중 질문과 가장 관련 있는 Top-5를 선정하고 선정 이유를 짧게 기록한다.

Generator Node: 논문 리스트를 바탕으로 답변을 생성하고 출처 링크를 포함한다.

Step 4: 인터페이스 (main.py)
CLI 환경에서 사용자의 입력을 받고 LangGraph의 스트리밍 출력을 보여준다.