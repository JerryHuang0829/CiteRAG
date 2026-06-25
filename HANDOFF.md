# HANDOFF — CiteRAG

當前狀態快照（非變更日誌）。每 session 結束覆寫更新。

- 更新時間：2026-06-25
- 階段：**W0–W3 ✅ ＋ 工程強化 A1–D2 ✅ ＋ RAG 優化 ①②③ ✅ ＋ 結構化 DB 層 ✅ ＋ pgvector 後端 ✅ ＋ 多輪對話記憶 ✅ ＋ E3 打包 ✅ ＋ CI gate ✅ ＋ git/GitHub 已上線（JerryHuang0829/CiteRAG）**。
  下一步：**E2 一頁 case study**；之後選配 C1 雲端 router（待 key）/ 測試集擴充。

---

## 一、現況總結

通用 RAG+Agent 文件助手（轉職 AI 工程師作品集）。i7-1260P CPU-only / 16GB / 零預算。RAG 全鏈路（解析→切塊→嵌入→**hybrid 檢索**→重排→生成→**引用護欄**）在真實繁中金融語料跑通，有**三層量化 eval + bootstrap CI + CI gate**、對外 **REST API + 自製前端**。從 L1（有檢索評測）推到 **L2（hybrid + 生成端 groundedness + eval CI）**。剩 git init + 一頁 case study 即可投。

## 二、進度

- **W0–W3 ✅**：最小 RAG（PDF→chunk→bge-small-zh→FAISS→帶頁碼引用→qwen3:4b streaming + 拒答）；三層 eval；**程式碼編排單步 Agent**（非自由 ReAct，工具+護欄+max_turns+強制收尾）；VLM 讀圖（Gemma3:4b）；Gradio 三分頁。
- **工程強化 A1–D2**：
  - A1 例外處理（`_open_ollama`/`OllamaError`/索引守衛，CLI/API 友善錯誤）。
  - A2 Agent crash 修復（巢狀 args→`json.dumps` sig、非標量/`tool` 非字串/非物件 JSON 守衛）。
  - A3 引用護欄 `verify_citations`（剝除模式，多頁 `(p.3,4)` 逐頁；agent `allowed_pages` 取自 `search_filings.last_pages` 可靠頁，非 re-parse 字串）。接 core.answer/agent/app/api。
  - B1 eval 統計重構（`rag/stats.py`：bootstrap/paired/rule-of-three，純 stdlib、固定 seed、線性 percentile；修 M1/M4/M7）。
  - B2 難情境壓測 `eval_agent_hard.py`。
  - D1 FastAPI `api.py`（/health /ask /agent /vlm，503）。**D2 自製前端 `rag/web/`**（HTML/CSS/JS 打 API，同源免 CORS）+ api 輸入防護（422/400/500）。
- **RAG 優化 ①②③（L1→L2）**：
  - **① Hybrid 檢索**（`core` BM25 手刻 + RRF；`USE_HYBRID` 預設開）。`eval_hybrid.py` 兩層級 A/B。
  - **② Golden set + RAG Triad**：`golden.jsonl`（28 題版本化）+ `golden.py`（驗證器）+ `eval_rag_triad.py`（手刻 RAGAS-style：answer correctness / context recall / precision / faithfulness / answer relevancy）。
  - **③ eval CI gate**：`tests/`（pytest）+ `.github/workflows/ci.yml`；雲端跑確定性測試、本地跑需 index/Ollama 的 gate。
- **結構化 DB 層（FinMind，`findata.py`）**：agent 的 `lookup_metric` 由 mock 改接真實 FinMind；新增 `compare`（跨公司比較/排名/篩選）、`stock_price`（即時股價）工具，涵蓋全市場 ~2500 家（TaiwanStockInfo 名→碼）。「**數字走 DB、文字走 RAG、agent 自動分流**」；磁碟快取 `.findata_cache.json`（git 忽略）省 API 額度。實測：台積電 2023 EPS 32.34、跨公司排名、台積電股價 2390/近一年+144%。
- **向量資料庫後端（`pgstore.py`）**：FAISS 之外新增可切換的 **pgvector**（`CITERAG_PGVECTOR=1`；Postgres+pgvector docker 容器 citerag-pg），多了 **SQL metadata 過濾**（如 `retrieve(q, source=...)` 只查某來源文件，FAISS 做不到）+ 持久化；FAISS 維持預設。實測過濾生效、FAISS 路徑無回歸、雲端 pytest 21 passed。psycopg2 直接格式化向量字串(免 pgvector py 套件)。
- **多輪對話記憶（輕量）**：agent `run/_events/run_iter` 加 `history` 參數（`[{role,content}]`），messages＝`SYSTEM + history + 本輪`，SYSTEM 加代名詞解析提示。前端持有對話、每次帶最近 3 輪（無狀態 server；`api.py` `AgentReq.history`＋角色/長度驗證＋server 端再裁 6 則）。`web/` Agent 分頁改成對話串（user 泡泡 + 工具軌跡 + 清除鈕）。`agent.py` 無參數啟動＝多輪 REPL。實測：先問「台積電 2023 EPS」→ 再問「那它的營收呢」，agent 把「它」解析為台積電且連年份 2023 一起帶過去（lookup_metric 營收 2023＝2.162 兆），CLI/HTTP 兩路皆過。**邊界**：只塞前文 Q/A（非完整多輪 ReAct），4B + num_ctx=4096 下控制 context；逾 3 輪或長對話可靠度未驗。
- **E3 打包**：`README.md`（架構圖/quickstart/評測報告卡/failure cases）、`.gitignore`、`requirements.txt`。

## 三、關鍵實測數字（本機，可重現／seed 固定）

- 速度：decode ~3–4 tok/s、一題約 30–60 秒（瓶頸是慢非不可靠）。
- **檢索（n=12）**：dense baseline hit@5 0.833 / recall@5 0.651 / MRR 0.562 / nDCG@5 0.552。
- **① Hybrid（檢索層，paired bootstrap）**：dense→hybrid MRR 0.562→0.792、nDCG@5 0.552→0.714；**ΔMRR +0.229 [+0.076,+0.396] 顯著、ΔnDCG +0.163 [+0.029,+0.305] 顯著**。**端到端（再經 reranker）Δ=0**——78-chunk 小語料 + top-N 佔 26% + 強 reranker 把 dense 漏召補回；hybrid 端到端 ROI 隨語料變大上升。
- **Reranker ablation（n=12）**：bge-base ΔMRR +0.271 [+0.014,+0.507] 顯著、hit@3 1.000；jina-v2 ΔMRR 觸 0 不顯著 → 預設採 **bge-reranker-base**（n=12 結論，待擴題定論）。
- **② RAG Triad（golden 28 題）**：answer correctness **0.893 [0.786,1.000]**；**context recall 1.000**（檢索不漏）；context precision 0.490；faithfulness(4B judge) 0.667（**noisy proxy，judge 把多題答對的判不忠實→再證 4B judge 不可靠**）；relevancy 1.000。分類：factual/qualitative 1.000、exact-term 0.875、refuse 0.667、trap 0.000(n=1)。
- **Agent 難情境（n=8）**：hard success 0.875 [0.625,1.000]。
- **③ CI**：雲端確定性套件 21 passed；本地 gate（golden grounding + context recall≥0.7）2 passed。

## 四、failure-case 素材（root cause → fix → 數字/行為）

1. **頁碼引用造假**：取錯 metadata key 致 p.-1、LLM 吐似是而非頁碼 → 修正用 `metadata['page_number']` + 加 runtime 引用護欄。
2. **reranker 小樣本翻轉**：85→78 索引重建後 bge/jina 排序翻面 → n=12 不足定 reranker 終局；預設改 bge、降級「顯著」宣稱。
3. **hybrid 被 reranker 遮蔽**：檢索層顯著、端到端打平 → 拆兩層級量測才看清，效益隨語料規模上升。
4. **小模型條件式/拒答失誤**：4B 對「若 X 才 Y」易失敗、會把民國年誤算西元；golden r2「2025 營收」該拒答卻沒拒、h8「資本支出 979」已在 context 卻沒抽出 → 可定位的生成端缺口（更強抽取/拒答 prompt）。
5. **deterministic 揭穿 LLM judge**：faithfulness(4B)=0.667 < 客觀 correctness 0.893，judge 把答對的判不忠實 → 本地小模型 judge 不可靠，需獨立/更強 judge 校準。

## 五、已知問題 / 待辦

- **E2 一頁 case study 未產出**（把上面 failure cases 寫成顧問敘事；補 JD「需求→AI 方案」軸線）。
- **C1 雲端 router 未接**（JD「串接雲端 API」；免費金鑰未申請；`core` 僅本地 Ollama）。
- 測試集偏小（檢索 n=12 / golden 28）、僅 2 份 PDF → 擴題是 reranker/hybrid 終局與收窄 CI 的前提。
- `TOP_K=5`（已對齊 eval MAIN_K）；n8n/Dify、STT/TTS/影像生成 未碰（多模態 2/5）。
- 多輪只塞前文 Q/A、上限 3 輪；長對話／4B 多輪可靠度未系統性評測（無 multi-turn golden）。
- `w0_results_agent.json` 為舊 schema，待重跑刷新。

## 六、檔案

- `PLAN.md`(SSOT) / `CLAUDE.md`(守則) / `HANDOFF.md`(本檔) / `README.md`(對外) / `requirements.txt` / `.gitignore`
- `rag/`：`core.py`（引擎：解析/嵌入/FAISS/**BM25+RRF**/重排/生成/`verify_citations`/`vlm_b64`）/ `agent.py`（5 工具）/ **`findata.py`**（FinMind 結構化查詢：lookup/compare/stock_price）/ **`pgstore.py`**（pgvector 向量資料庫後端，可切換） / `app.py`(Gradio) / `api.py`(FastAPI) / **`web/`**(自製前端) / `stats.py` / **`golden.py`+`golden.jsonl`** / `ingest.py` / `ask.py`
  - eval：`eval_retrieval.py` / `eval_rerank.py` / **`eval_hybrid.py`** / `eval_generation.py` / **`eval_rag_triad.py`** / `eval_agent.py` / `eval_agent_hard.py`
- `tests/`：pytest（雲端確定性 + 本地 gate）　|　`.github/workflows/ci.yml`
- `w0/`：benchmark 腳本 + json　|　`data/`：公開 PDF + sample_nameplate.png　|　`index/`：faiss + chunks(78)
- `hf_cache/`：bge-small-zh / bge-reranker-base / jina-v2 權重（git 忽略）　|　`models/`：舊快取（git 忽略）

## 七、環境

- conda env `rag` @ `E:\conda\envs\rag`（Python 3.12）：fastembed 0.8.0 / faiss-cpu 1.14.3 / pymupdf4llm 1.27.2.3 / langchain-text-splitters 1.1.2 / numpy 2.5.0 / gradio 6.19.0 / fastapi 0.138.0 / uvicorn 0.49.0 / pydantic 2.13.4 / pytest 9.1.1。
- 執行（cwd 須在 `rag/`）：`ingest.py` 建索引、`ask.py`/`agent.py` CLI、`app.py`(7860 Gradio)、`uvicorn api:app --port 8000`(API+前端 `/app`)、`eval_*.py` 評測、`golden.py` 驗證。
- 測試：`pytest -m "not local"`（雲端確定性，秒級）／`pytest -m local`（需 index/reranker）。
- Ollama：`qwen3:4b-instruct`(LLM) + `gemma3:4b`(VLM)。預設 reranker `BAAI/bge-reranker-base`。
- ⚠️ 16GB：避免同時跑多個載模型程序（uvicorn+gradio+eval 併發會 OOM；重型序列化）。

## 八、下一步（待 user 選）

1. **E2 一頁 case study**（顧問敘事 + 4 個 failure cases；CP 值最高的對外敘事）。
2. **C1 雲端 router**（待免費 Groq/Gemini key；provider 抽象 + 本地 vs 雲端對照表）。
3. 測試集擴充（含 multi-turn golden）+ 重跑全 eval；2 分鐘 demo 影片。
