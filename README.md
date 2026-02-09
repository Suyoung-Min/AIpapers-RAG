# arXiv RAG 시스템

arXiv 논문 초록 데이터를 활용한 에이전틱 RAG(Retrieval-Augmented Generation) 시스템입니다.

## 기술 스택

| 구분 | 기술 |
|------|------|
| Language | Python 3.10+ |
| Orchestration | LangGraph, LangChain |
| Vector DB | ChromaDB (Local, Persistent) |
| Embedding | all-MiniLM-L6-v2 (ChromaDB 기본 임베딩) |
| LLM | OpenAI GPT-4o-mini |
| Data Source | arXiv API (arxiv 라이브러리) |
| Web Framework | FastAPI + Uvicorn |

## 데이터 수집 (Ingestion)

### 개요

`src/ingestion.py` 모듈이 arXiv 논문 데이터 수집 및 벡터 인덱싱을 담당합니다.

### 수집 설정

- **대상 카테고리**: `cs.CL` (Computation and Language)
- **수집 논문 수**: 1,000편
- **API 호출 딜레이**: 3초 (arXiv API rate limit 준수)
- **정렬 기준**: 최신 등록일 (SubmittedDate, Descending)

### 수집 데이터 필드

| 필드 | 설명 |
|------|------|
| id | arXiv 논문 고유 ID |
| title | 논문 제목 |
| abstract | 논문 초록 |
| url | arXiv 논문 링크 |
| categories | 논문 카테고리 목록 |
| published | 발행일 (ISO 형식) |
| authors | 저자 목록 |

### 임베딩 및 인덱싱

1. **문서 구성**: `title + "\n\n" + abstract` 형태로 결합
2. **임베딩 모델**: ChromaDB 기본 임베딩 (all-MiniLM-L6-v2, 384차원)
3. **유사도 측정**: Cosine Similarity (`hnsw:space = cosine`)
4. **배치 처리**: 100개 단위로 분할 삽입

### 저장 경로

- JSON 원본 데이터: `data/papers.json`
- ChromaDB 벡터 DB: `chroma_db/`

### 실행 방법

```bash
# 데이터 수집 및 인덱싱 실행
python -m src.ingestion
```

## 프로젝트 구조

```
arxiv_rag/
├── data/               # 수집된 JSON 데이터
├── chroma_db/          # ChromaDB 벡터 저장소
├── src/
│   ├── ingestion.py    # arXiv API 수집 및 DB 인덱싱
│   ├── state.py        # LangGraph State 정의
│   ├── nodes.py        # 노드 로직 (Router, Retriever, Reranker, Generator)
│   ├── graph.py        # LangGraph 워크플로우 구성
│   ├── app.py          # FastAPI 웹 애플리케이션
│   └── main.py         # CLI 실행 진입점
├── .env                # API 키 (OPENAI_API_KEY)
└── requirements.txt    # 의존성 패키지
```