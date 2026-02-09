"""CLI 실행 진입점."""

import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src.graph import build_graph


def main():
    """CLI 환경에서 사용자 입력을 받고 RAG 파이프라인을 실행한다."""
    print("=" * 60)
    print("  arXiv 논문 RAG 시스템")
    print("  cs.CL 카테고리 논문을 검색하고 답변합니다.")
    print("  종료하려면 'quit' 또는 'exit'를 입력하세요.")
    print("=" * 60)

    graph = build_graph()

    while True:
        print()
        try:
            question = input("질문: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n프로그램을 종료합니다.")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("프로그램을 종료합니다.")
            break

        initial_state = {
            "question": question,
            "documents": [],
            "filters": {},
            "generation": "",
            "steps": [],
            "route": "",
        }

        print("\n처리 중...")
        print("-" * 40)

        # 스트리밍 출력: 각 노드 실행 결과를 순차적으로 표시
        final_state = None
        for event in graph.stream(initial_state):
            for node_name, node_state in event.items():
                final_state = node_state
                steps = node_state.get("steps", [])
                if steps:
                    latest_step = steps[-1]
                    print(f"[{node_name}] {latest_step}")

        # 최종 답변 출력
        print("-" * 40)
        if final_state and final_state.get("generation"):
            print("\n답변:")
            print(final_state["generation"])
        else:
            print("\n답변을 생성하지 못했습니다.")


if __name__ == "__main__":
    main()
