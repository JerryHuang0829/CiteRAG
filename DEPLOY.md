# 部署到 Hugging Face Spaces（免費 CPU，免信用卡）

把 CPU-only 本機專案升級成「線上可點的雲端產品」。生成走免費雲端 API（Gemini Flash-Lite 主 + Groq fallback），embedding/檢索/護欄一律在 Space 容器內本機跑（資料不外洩）。

## 0. 前置：拿兩把免費金鑰（免信用卡）
- Gemini：aistudio.google.com → Create API key
- Groq：console.groq.com → API Keys → Create
- 本機測試先寫進 `.env`（見 `.env.example`），跑 `python w0/03_cloud_probe.py` 應印 `auth OK`。
- ⚠️ 金鑰是密碼：勿貼進截圖/聊天/commit。

## 1. 建 Space
huggingface.co → New Space → **SDK = Docker（Blank）**、Hardware = **CPU basic（free）**。
建立後在 repo 根 `README.md` 最上方需有這段 HF metadata（HF 建立流程會幫你產生，填這些值即可）：

```yaml
---
title: CiteRAG
emoji: 📄
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---
```

## 2. 設 secrets（不要放進 repo）
Space → **Settings → Variables and secrets** 新增：
- `GEMINI_API_KEY`（secret）
- `GROQ_API_KEY`（secret）
- `CITERAG_LLM_BACKEND` = `cloud`（variable；Dockerfile 已預設 cloud，這行可省）

容器啟動時這些會以環境變數注入，`llm_router._getenv` 直接讀到——**不需 .env**。

## 3. 推上去
```bash
git remote add space https://huggingface.co/spaces/<你的帳號>/CiteRAG
git push space main
```
HF 會自動 `docker build`（跑 `ingest.py` 建索引、下載 bge-small-zh）→ 啟動 `uvicorn api:app:7860`。
完成後 demo URL：`https://<你的帳號>-citerag.hf.space`，前端在 `/app`。

## 4. 本機先驗 Docker（選配，建議）
```bash
docker build -t citerag .
docker run --rm -p 7860:7860 \
  -e GEMINI_API_KEY=$GEMINI_API_KEY -e GROQ_API_KEY=$GROQ_API_KEY \
  citerag
# 開 http://localhost:7860/app 或 GET /health
```

## 注意
- **VLM 讀圖（/vlm）走本機 Ollama，雲端 demo 不含**（免費主機無 GPU/Ollama）；/ask /agent 正常。
- 免費主機 48h 無流量會睡，下次造訪冷啟動數秒——portfolio demo 可接受。
- HF cpu-basic 2 vCPU / 16 GB / Docker，免信用卡；secrets 由 Space 管，repo 不含金鑰。
- 履歷誠實描述：「CPU-only、scale-to-zero、$0/月；embedding 本機跑、文件資料不外洩」——對受監管金融是加分。
