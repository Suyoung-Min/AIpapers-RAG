"""FastAPI 웹 서버 - arXiv RAG 시스템 웹 인터페이스."""

import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

load_dotenv()

from src.graph import build_graph

app = FastAPI(title="arXiv 논문 RAG")
graph = build_graph()


class Question(BaseModel):
    question: str


@app.post("/ask")
def ask(q: Question):
    """질문을 받아 RAG 파이프라인을 실행하고 JSON으로 반환한다."""
    initial_state = {
        "question": q.question,
        "documents": [],
        "filters": {},
        "generation": "",
        "steps": [],
        "route": "",
    }

    result = graph.invoke(initial_state)

    docs = []
    for doc in result.get("documents", []):
        meta = doc.get("metadata", {})
        docs.append({
            "title": meta.get("title", ""),
            "url": meta.get("url", ""),
            "published": meta.get("published", ""),
            "authors": meta.get("authors", ""),
        })

    return {
        "generation": result.get("generation", "답변을 생성하지 못했습니다."),
        "steps": result.get("steps", []),
        "documents": docs,
    }


@app.get("/", response_class=HTMLResponse)
async def index():
    """메인 웹 페이지를 반환한다."""
    return HTML_PAGE


HTML_PAGE = """\
<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>arXiv 논문 RAG</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: #0f172a;
    color: #e2e8f0;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
  }
  header {
    background: #1e293b;
    border-bottom: 1px solid #334155;
    padding: 16px 24px;
    text-align: center;
  }
  header h1 {
    font-size: 1.3rem;
    font-weight: 600;
    color: #f8fafc;
  }
  header p {
    font-size: 0.85rem;
    color: #94a3b8;
    margin-top: 4px;
  }
  #chat-container {
    flex: 1;
    overflow-y: auto;
    padding: 24px;
    max-width: 800px;
    width: 100%;
    margin: 0 auto;
  }
  .message {
    margin-bottom: 20px;
    animation: fadeIn 0.3s ease;
  }
  @keyframes fadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
  }
  .message.user {
    display: flex;
    justify-content: flex-end;
  }
  .message.user .bubble {
    background: #3b82f6;
    color: #fff;
    border-radius: 16px 16px 4px 16px;
    padding: 10px 16px;
    max-width: 70%;
    word-break: break-word;
  }
  .message.assistant .bubble {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 16px 16px 16px 4px;
    padding: 16px;
    max-width: 85%;
    line-height: 1.7;
    word-break: break-word;
  }
  .message.assistant .bubble p { margin-bottom: 8px; }
  .message.assistant .bubble p:last-child { margin-bottom: 0; }
  .steps {
    font-size: 0.8rem;
    color: #64748b;
    margin-bottom: 8px;
  }
  .steps .step {
    padding: 3px 0;
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .steps .step::before {
    content: '';
    display: inline-block;
    width: 6px; height: 6px;
    background: #3b82f6;
    border-radius: 50%;
    flex-shrink: 0;
  }
  .loading {
    display: inline-flex;
    gap: 4px;
    align-items: center;
    padding: 8px 0;
  }
  .loading span {
    width: 6px; height: 6px;
    background: #64748b;
    border-radius: 50%;
    animation: bounce 1.4s infinite both;
  }
  .loading span:nth-child(2) { animation-delay: 0.16s; }
  .loading span:nth-child(3) { animation-delay: 0.32s; }
  @keyframes bounce {
    0%, 80%, 100% { transform: scale(0); }
    40% { transform: scale(1); }
  }
  #input-area {
    background: #1e293b;
    border-top: 1px solid #334155;
    padding: 16px 24px;
  }
  #input-wrap {
    max-width: 800px;
    margin: 0 auto;
    display: flex;
    gap: 10px;
  }
  #question {
    flex: 1;
    background: #0f172a;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 12px 16px;
    color: #e2e8f0;
    font-size: 0.95rem;
    outline: none;
    transition: border-color 0.2s;
  }
  #question:focus { border-color: #3b82f6; }
  #question::placeholder { color: #475569; }
  #send-btn {
    background: #3b82f6;
    color: #fff;
    border: none;
    border-radius: 12px;
    padding: 12px 24px;
    font-size: 0.95rem;
    font-weight: 500;
    cursor: pointer;
    transition: background 0.2s;
    white-space: nowrap;
  }
  #send-btn:hover { background: #2563eb; }
  #send-btn:disabled { background: #475569; cursor: not-allowed; }

  /* Markdown-like rendering */
  .bubble h3 { font-size: 1rem; margin: 12px 0 6px; color: #f8fafc; }
  .bubble ul, .bubble ol { padding-left: 20px; margin: 6px 0; }
  .bubble li { margin: 4px 0; }
  .bubble a { color: #60a5fa; text-decoration: none; }
  .bubble a:hover { text-decoration: underline; }
  .bubble code {
    background: #334155;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 0.85em;
  }
  .bubble strong { color: #f8fafc; }
</style>
</head>
<body>

<header>
  <h1>arXiv 논문 RAG 시스템</h1>
  <p>cs.CL 카테고리 논문을 검색하고 답변합니다</p>
</header>

<div id="chat-container">
  <div class="message assistant">
    <div class="bubble">
      안녕하세요! arXiv 논문 검색 시스템입니다.<br>
      자연어처리(cs.CL) 분야 논문에 대해 궁금한 점을 질문해 주세요.
    </div>
  </div>
</div>

<div id="input-area">
  <div id="input-wrap">
    <input type="text" id="question" placeholder="논문에 대해 질문하세요..." autocomplete="off">
    <button id="send-btn">전송</button>
  </div>
</div>

<script>
const chatContainer = document.getElementById('chat-container');
const questionInput = document.getElementById('question');
const sendBtn = document.getElementById('send-btn');

function scrollBottom() {
  chatContainer.scrollTop = chatContainer.scrollHeight;
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

function renderMarkdown(text) {
  // Simple markdown: bold, links, newlines
  let html = escapeHtml(text);
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/\[([^\]]+)\]\((https?:\/\/[^\)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
  html = html.replace(/(?:^|\\n)### (.+)/g, '<h3>$1</h3>');
  html = html.replace(/\\n/g, '<br>');
  return html;
}

async function sendQuestion() {
  const q = questionInput.value.trim();
  if (!q) return;

  // User message
  const userDiv = document.createElement('div');
  userDiv.className = 'message user';
  userDiv.innerHTML = '<div class="bubble">' + escapeHtml(q) + '</div>';
  chatContainer.appendChild(userDiv);

  questionInput.value = '';
  sendBtn.disabled = true;
  questionInput.disabled = true;
  scrollBottom();

  // Assistant message (placeholder with loading)
  const assistDiv = document.createElement('div');
  assistDiv.className = 'message assistant';

  const loadingDiv = document.createElement('div');
  loadingDiv.className = 'loading';
  loadingDiv.innerHTML = '<span></span><span></span><span></span> &nbsp; 처리 중...';

  const bubbleDiv = document.createElement('div');
  bubbleDiv.className = 'bubble';
  bubbleDiv.style.display = 'none';

  assistDiv.appendChild(loadingDiv);
  assistDiv.appendChild(bubbleDiv);
  chatContainer.appendChild(assistDiv);
  scrollBottom();

  try {
    const response = await fetch('/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: q }),
    });

    if (!response.ok) {
      throw new Error('서버 오류 (' + response.status + ')');
    }

    const data = await response.json();

    loadingDiv.style.display = 'none';
    bubbleDiv.style.display = 'block';
    bubbleDiv.innerHTML = renderMarkdown(data.generation);
    scrollBottom();
  } catch (err) {
    loadingDiv.style.display = 'none';
    bubbleDiv.style.display = 'block';
    bubbleDiv.textContent = '오류가 발생했습니다: ' + err.message;
  }

  sendBtn.disabled = false;
  questionInput.disabled = false;
  questionInput.focus();
  scrollBottom();
}

sendBtn.addEventListener('click', sendQuestion);
questionInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.isComposing) {
    e.preventDefault();
    sendQuestion();
  }
});

questionInput.focus();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.app:app", host="0.0.0.0", port=8000, reload=True)
