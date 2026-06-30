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

# 程式 + 語料（.dockerignore 已排除 index/hf_cache/.env/tests/w0）
COPY --chown=user:user rag/ rag/
COPY --chown=user:user data/ data/

# WORKDIR 由 root 建立；切 user 前確保 app 目錄（含待建的 hf_cache/index）可寫
RUN mkdir -p hf_cache index && chown -R user:user /home/user/app

USER user

# build 時建索引（下載 bge-small-zh + 嵌入 78 chunks → FAISS）+ 暖機 reranker（bge-reranker-base）
# 一次 retrieve 觸發 reranker 下載並快取進 image，避免 runtime 首次請求才下載（~140s + HF 限速會 500）
RUN cd rag && python ingest.py && python -c "import core; core.retrieve('暖機載入 reranker 模型', k=3)"

EXPOSE 7860
# FastAPI：/app 前端 + /ask /agent（cloud 生成）；/health 不碰 LLM。VLM /vlm 走本機 Ollama，雲端 demo 不含。
CMD ["sh", "-c", "cd rag && uvicorn api:app --host 0.0.0.0 --port 7860"]
