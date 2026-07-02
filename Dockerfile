# CiteRAG — CPU-only RAG+Agent，部署於 Hugging Face Spaces (Docker SDK) 等免費主機。
# 生成走雲端免費 API（CITERAG_LLM_BACKEND=cloud；金鑰用 Space secrets 注入，勿 bake 進 image）；
# embedding/檢索/護欄一律本機（bge-small-zh ONNX），資料不外洩。
FROM python:3.12-slim

# HF Spaces 慣例：非 root user（uid 1000）、家目錄可寫
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/app/hf_cache \
    HF_HUB_DISABLE_SYMLINKS=1 \
    CITERAG_LLM_BACKEND=cloud \
    PYTHONUNBUFFERED=1

WORKDIR /home/user/app

# 先裝依賴（獨立快取層，改 code 不重裝）
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 程式 + 語料 + 預建索引（index/ 已 commit；HF 免費 build 記憶體無法 embed 9.6k chunks，故不在 build 時重建）
COPY --chown=user:user rag/ rag/
COPY --chown=user:user data/ data/
COPY --chown=user:user index/ index/

# WORKDIR 由 root 建立；切 user 前確保 app 目錄（含待建的 hf_cache）可寫
RUN mkdir -p hf_cache && chown -R user:user /home/user/app

USER user

# 索引已 commit，不重跑 embedding。build 時只預先下載並快取模型：
# bge-small-zh（查詢嵌入）+ bge-reranker-base（重排），避免 runtime 首次請求才下載（~140s + HF 限速會 500）
RUN cd rag && python -c "import core; core.get_embedder(); core.get_reranker(core.RERANK_MODEL)"

EXPOSE 7860
# FastAPI：/app 前端 + /ask /agent（cloud 生成）；/health 不碰 LLM。VLM /vlm 走本機 Ollama，雲端 demo 不含。
CMD ["sh", "-c", "cd rag && uvicorn api:app --host 0.0.0.0 --port 7860"]
