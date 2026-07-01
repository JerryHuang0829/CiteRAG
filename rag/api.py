"""D1/D2 對外 REST API + 自製 Web 前端（FastAPI）。

啟動（在 rag/）：uvicorn api:app --host 127.0.0.1 --port 8000
  前端：http://127.0.0.1:8000/app　　API 文件：http://127.0.0.1:8000/docs
前置：Ollama 在跑、索引已建（python ingest.py）。
端點：
  GET  /health           服務與模型資訊（不碰 Ollama）
  POST /ask    {query,k?}             → {answer, stripped_pages, sources[]}
  POST /agent  {message, history?}    → {answer, trace[]}   history＝[{role,content}] 最近數輪（多輪記憶）
  POST /vlm    {image_b64, question?} → {text}
  GET  /app                           自製前端（同源呼叫上面端點）
"""
import base64
import binascii
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from pydantic import BaseModel, Field

import agent
import core
import security

app = FastAPI(title="CiteRAG API", version="1.1")

_WEB = Path(__file__).resolve().parent / "web"
_MAX_TEXT = 2000
_MAX_IMG_B64 = 20_000_000   # ~15MB 圖


class AskReq(BaseModel):
    query: str = Field(..., min_length=1, max_length=_MAX_TEXT)
    k: int | None = Field(None, ge=1, le=20)


class Msg(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., min_length=1, max_length=_MAX_TEXT)


class AgentReq(BaseModel):
    message: str = Field(..., min_length=1, max_length=_MAX_TEXT)
    # 前端持有對話、每次帶最近幾輪（無狀態 server）；讓「它/那家」等代名詞可解析。
    history: list[Msg] | None = Field(None, max_length=12)


class VlmReq(BaseModel):
    image_b64: str = Field(..., min_length=1, max_length=_MAX_IMG_B64)
    question: str | None = Field(None, max_length=_MAX_TEXT)


@app.exception_handler(core.OllamaError)
def _ollama_error(request, exc):
    # 外部依賴（Ollama）失敗回 503（服務暫不可用），而非 500 + traceback
    return JSONResponse(status_code=503, content={"error": str(exc)})


@app.exception_handler(FileNotFoundError)
def _index_error(request, exc):
    return JSONResponse(status_code=503, content={"error": str(exc)})


@app.exception_handler(RequestValidationError)
def _validation_error(request, exc):
    # 驗證錯誤統一回 {error}（與其他 handler 對齊）；預設的 {detail:[...]} 前端只讀 data.error 會顯示成裸「HTTP 422」
    msgs = "；".join(str(e.get("msg", "")) for e in exc.errors()[:3])
    return JSONResponse(status_code=422, content={"error": f"輸入格式不符：{msgs}"})


@app.exception_handler(Exception)
def _generic_error(request, exc):
    # 兜底：未預期錯誤回乾淨 500，不洩漏內部 traceback
    return JSONResponse(status_code=500, content={"error": f"內部錯誤：{type(exc).__name__}"})


@app.get("/health")
def health():
    import llm_router
    backend = os.environ.get("CITERAG_LLM_BACKEND", "ollama").strip().lower()
    cloud = backend == "cloud"
    # 雲端時回實際模型（Gemini + Groq fallback），並標 VLM 不可用（Gemma 讀圖走本機 Ollama，雲端無 GPU/Ollama）
    gen = f"{llm_router.GEMINI_MODEL} +{llm_router.GROQ_MODEL}(fallback)" if cloud else core.GEN_MODEL
    return {"status": "ok", "llm_backend": backend, "gen_model": gen,
            "vlm_model": core.VLM_MODEL, "vlm_available": not cloud,
            "rerank_model": core.RERANK_MODEL if core.USE_RERANK else None,
            "use_rerank": core.USE_RERANK, "top_k": core.TOP_K}


@app.post("/ask")
def ask(req: AskReq):
    k = req.k or core.TOP_K
    hits = core.retrieve(req.query, k)
    raw = core.generate(core.build_prompt(req.query, hits))
    answer, stripped = core.verify_citations(raw, {h["page"] for h in hits})
    answer, pii = security.redact_pii(answer)          # 輸出護欄：遮罩任何 PII（defense-in-depth）
    return {
        "answer": answer,
        "stripped_pages": stripped,
        "pii_redacted": [t for t, _ in pii],
        "sources": [
            {"source": h["source"], "page": h["page"], "score": round(h["score"], 4)}
            for h in hits
        ],
    }


@app.post("/agent")
def run_agent(req: AgentReq):
    # 只保留最近 6 則（3 輪），控制 4B 的 context（num_ctx=4096）
    history = [m.model_dump() for m in (req.history or [])][-6:] or None
    final, trace = agent.run(req.message, history=history, verbose=False)
    final, pii = security.redact_pii(final)             # 輸出護欄：遮罩任何 PII
    return {"answer": final, "trace": trace, "pii_redacted": [t for t, _ in pii]}


@app.post("/vlm")
def vlm(req: VlmReq):
    b64 = req.image_b64.split(",", 1)[-1]   # 容忍 data:image/...;base64, 前綴
    try:
        base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError):
        return JSONResponse(status_code=400, content={"error": "image_b64 不是合法的 base64"})
    text = core.vlm_b64(b64, req.question or "請讀出圖片中的所有文字與數值。")
    return {"text": text}


# ---- 自製 Web 前端（同源服務，免 CORS）----
@app.get("/", include_in_schema=False)
def _root():
    return RedirectResponse("/app")


@app.get("/app", response_class=HTMLResponse, include_in_schema=False)
def _web_index():
    return (_WEB / "index.html").read_text(encoding="utf-8")


@app.get("/app/style.css", include_in_schema=False)
def _web_css():
    return Response((_WEB / "style.css").read_text(encoding="utf-8"),
                    media_type="text/css; charset=utf-8")


@app.get("/app/app.js", include_in_schema=False)
def _web_js():
    return Response((_WEB / "app.js").read_text(encoding="utf-8"),
                    media_type="application/javascript; charset=utf-8")
