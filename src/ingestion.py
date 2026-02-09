"""arXiv 논문 수집 및 ChromaDB 인덱싱 모듈."""

import json
import os
import time
from pathlib import Path

import arxiv
import chromadb
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(__file__).parent.parent / "data"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"
COLLECTION_NAME = "arxiv_papers"


def fetch_arxiv_papers(
    category: str = "cs.CL",
    max_results: int = 5000,
    delay: float = 3.0,
) -> list[dict]:
    """arXiv API로부터 논문 메타데이터를 수집한다.

    Args:
        category: arXiv 카테고리 (기본: cs.CL).
        max_results: 수집할 논문 수.
        delay: API 호출 간 딜레이(초).

    Returns:
        논문 메타데이터 딕셔너리 리스트.
    """
    search = arxiv.Search(
        query=f"cat:{category}",
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    papers = []
    client = arxiv.Client(page_size=100, delay_seconds=delay)

    for result in client.results(search):
        paper = {
            "id": result.entry_id,
            "title": result.title,
            "abstract": result.summary,
            "url": result.entry_id,
            "categories": [cat for cat in result.categories],
            "published": result.published.isoformat(),
            "authors": [author.name for author in result.authors],
        }
        papers.append(paper)

    print(f"총 {len(papers)}편의 논문을 수집했습니다.")
    return papers


def save_papers_json(papers: list[dict], filename: str = "papers.json") -> Path:
    """수집한 논문 데이터를 JSON 파일로 저장한다."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    filepath = DATA_DIR / filename
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(papers, f, ensure_ascii=False, indent=2)
    print(f"데이터를 {filepath}에 저장했습니다.")
    return filepath


def index_to_chromadb(papers: list[dict]) -> chromadb.Collection:
    """논문 데이터를 ChromaDB에 인덱싱한다.

    title + abstract를 결합하여 임베딩하고, 메타데이터와 함께 저장한다.
    """
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # 기존 컬렉션이 있으면 삭제 후 재생성
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # ChromaDB는 배치 크기 제한이 있으므로 분할 삽입
    batch_size = 100
    for i in range(0, len(papers), batch_size):
        batch = papers[i : i + batch_size]

        ids = [paper["id"] for paper in batch]
        documents = [
            f"{paper['title']}\n\n{paper['abstract']}" for paper in batch
        ]
        metadatas = [
            {
                "title": paper["title"],
                "url": paper["url"],
                "categories": ", ".join(paper["categories"]),
                "published": paper["published"],
                "authors": ", ".join(paper["authors"][:5]),
            }
            for paper in batch
        ]

        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )
        print(f"  인덱싱 진행: {min(i + batch_size, len(papers))}/{len(papers)}")

    print(f"ChromaDB 인덱싱 완료: {collection.count()}개 문서")
    return collection


def run_ingestion():
    """전체 수집-저장-인덱싱 파이프라인을 실행한다."""
    print("=== arXiv 논문 수집 시작 ===")
    papers = fetch_arxiv_papers()

    print("\n=== JSON 파일 저장 ===")
    save_papers_json(papers)

    print("\n=== ChromaDB 인덱싱 시작 ===")
    index_to_chromadb(papers)

    print("\n=== 수집 및 인덱싱 완료 ===")


if __name__ == "__main__":
    run_ingestion()
