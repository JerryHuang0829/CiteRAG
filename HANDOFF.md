# HANDOFF — CiteRAG

當前狀態快照（非變更日誌）。每 session 結束覆寫更新。

- 更新時間：2026-07-01
- 中文名：**本地檢索增強問答系統（CiteRAG）**（人看的標題用；code/repo/`CiteRAG API` 維持英文識別不動）。
- 階段：**W0–W3 ✅ ＋ 工程強化 ✅ ＋ RAG 優化 ①②③ ✅ ＋ 結構化 DB 層 ✅ ＋ pgvector ✅ ＋ 多輪對話記憶 ✅ ＋ 數值幻覺護欄 ✅ ＋ E3 打包 ✅ ＋ CI gate ✅ ＋ 全專案稽核 43 確認問題全修 ✅ ＋ git/GitHub 上線（JerryHuang0829/CiteRAG）＋ 雲端 LLM router ✅（`CITERAG_LLM_BACKEND=cloud`：Gemini Flash-Lite 主 + Groq fallback；離線測試 63 passed）**。
  ＋ Dockerfile/HF 部署檔 ✅ → **HF Spaces 上線 ✅ live: https://jerry0829-citerag.hf.space/app**（雲端生成 Gemini+Groq、embedding/檢索本機；VLM 分頁雲端自動隱藏）。
  ＋ **P2.1 eval-as-CI-gate ✅**：`slo.py`（凍結 SLO）+ `eval_gate.py`（低於門檻 exit 1）。檢索 gate recall 1.000/prec 0.432 PASS、mutation 驗「會擋」；生成 gate 走雲端 backend + retry 退避/節流（n=5 corr/faith 1.00）；`ci.yml` 3 jobs（test + retrieval-gate 每 PR + generation-gate nightly/手動）。
  ＋ **P2.2 安全包 ✅**：`security.py`（台灣 PII 偵測——身分證內政部檢核碼/信用卡 Luhn/手機/Email + 遮罩、injection 啟發式）接進 `/ask`+`/agent` 輸出護欄；`redteam.py`（5 攻擊對映 OWASP LLM Top-10，確定性判定，實測 block_rate **1.000**）進 nightly CI；`docs/security.md` 對映+誠實邊界；離線 **72 passed**。
  ＋ **P2.3 case study ✅**：`docs/case_study.md` 升級（隱私分層 / CD / eval-CI-gate / OWASP + 3 個新 failure case #6-8 + §9 履歷 bullet）。**→ P2 全部完成（P2.1 品質 gate + P2.2 安全 + P2.3 敘事）。**
  ＋ **P3-A ✅ LangGraph agent + Langfuse tracing**：`agent_lg.py`（LangGraph StateGraph，與手刻並存、4 等價測試 + mutation）+ `@observe` tracing（env-gated、pytest 下自動關；修 conda SSL_CERT_FILE→certifi bug）+ ADR。langgraph/langfuse 選配依賴不進核心/Docker。真實 trace 已送 Langfuse。離線 76 passed。
  ＋ **P3-B ✅ 真實年報 corpus**：`fetch_corpus.py`（TWSE doc.twse 三步抓年報 F04）→ 4 家指標公司年報加進 data/ → **78 → 9,579 chunks（123×）**。檢索 eval 在真實規模下 **recall 仍 1.000/prec 0.432 PASS**（robust）、新公司內容端到端帶頁碼答對（台積電營收 758億美元 p.5）。golden **已補 4 題台積電年報 factual Q**（美元營收/淨利/EPS/董事長，事實+可檢索性驗證，29 題 recall 仍 1.000、生成答對帶頁碼）；其他公司未覆蓋。年報 PDF 走 LFS。
  下一步：① 你 **GitHub Settings→Secrets 加 key** ② push `origin`(--force；LFS，含年報 ~28MB)+`space`（HF build 較久=重 embed 9.6k chunks）③ **P3-C observability**（Grafana 系統 metrics / /metrics）④ 選：紅色 CI 截圖 / 擴 golden 到其他公司。
- **全專案稽核（多代理 + 對抗式驗證，43 確認問題全修）**：HIGH＝①多輪 history 數字污染數值護欄（grounded 改只收本輪題目+工具結果，不含 history assistant）②findata explicit-year partial-year 誤標「全年」+永久快取（改標「前N季累計」+ 只磁碟快取完整四季）；MEDIUM＝search_filings.last_pages 全域單例 race（改結構化回傳就地取頁碼）、core.py 重型依賴 lazy import（確定性測試免載 ML 堆疊，雲端 2.5s）、findata/pgstore 補離線測試、verify_numbers 接入確定性狀態機測試、BM25 快取 re-ingest 失效、422 前後端契約一致 + 前端 history 中毒過濾；LOW/nit＝verify_numbers 年份豁免改上下文感知（緊鄰「年」才豁免）、千分位嚴格三位、verify_citations 裸式多頁/範圍、_resolve_code 最長匹配、_sum 容錯等。雲端測試 **36→55 passed**。

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
- **多輪對話記憶（multi-turn golden：6 段對話 / 13 turns / 7 follow-ups，含 ON vs OFF ablation）**：
  - **解析準確率（follow-up, n=7）**：history-ON **0.857 [0.571,1.000]** vs OFF **0.286 [0.000,0.571]**；**Δ(ON−OFF) +0.571 [+0.143,+0.857] 顯著**（配對 bootstrap，CI 不跨 0）→ 記憶確實買到跨輪解析力。點估計跨 run 約 0.71~0.86（4B 在即時股價輪非決定性，有時更正後才呼叫 stock_price）；Δ「顯著」為穩健結論。
  - **答案準確率（全 13 turns）**：**1.000 [0.769,1.000]**（13/13；財報歷史值斷言精確數值、即時股價輪斷言實體+工具呼叫——股價每日變動不可硬斷言，嚴謹度由 resolution 指標承擔）。
  - **誠實校正（n=7 小樣本）**：①ON 被「嚴格 tool 比對」低估——mt5「那營收呢」實際把台積電+聯電+2023 都帶對且答案正確，只因走 2×lookup 而非 compare 被判 miss（語意解析口徑 ON≈0.857）；②OFF 被「好猜案例」高估——mt1「它」猜台積電、mt4「那聯電呢」猜 2023 剛好對（in-text/可猜）→ 真實 Δ 應更大。兩邊都讓 Δ 偏保守，結論方向穩固。
  - **數值幻覺護欄（verify_numbers，修 failure #6）**：mt6 第二輪幻覺已由護欄處理——更正後逼出 stock_price 拿真值（257.5），或 4B 固執不查時誠實拒答；**「裸顯示未溯源數字」已成 0**。解析/答案準確率「點估計不變」（mt6 仍非 257、仍判 miss），改變的是 **failure mode：靜默編造 → 修正或拒答**（非決定性：能否救回真值取決於更正 retry 是否誘發工具呼叫）。
- **③ CI**：雲端確定性套件 **36 passed**（+9 多輪 schema/計分邏輯、+6 verify_numbers 數值溯源，皆合成資料不跑 LLM）；本地 gate **4 passed**（golden grounding + context recall≥0.7 + 多輪解析端到端 + 數值護欄「修正或拒答」各 ~60s）。

## 四、failure-case 素材（root cause → fix → 數字/行為）

1. **頁碼引用造假**：取錯 metadata key 致 p.-1、LLM 吐似是而非頁碼 → 修正用 `metadata['page_number']` + 加 runtime 引用護欄。
2. **reranker 小樣本翻轉**：85→78 索引重建後 bge/jina 排序翻面 → n=12 不足定 reranker 終局；預設改 bge、降級「顯著」宣稱。
3. **hybrid 被 reranker 遮蔽**：檢索層顯著、端到端打平 → 拆兩層級量測才看清，效益隨語料規模上升。
4. **小模型條件式/拒答失誤**：4B 對「若 X 才 Y」易失敗、會把民國年誤算西元；golden r2「2025 營收」該拒答卻沒拒、h8「資本支出 979」已在 context 卻沒抽出 → 可定位的生成端缺口（更強抽取/拒答 prompt）。
5. **deterministic 揭穿 LLM judge**：faithfulness(4B)=0.667 < 客觀 correctness 0.893，judge 把答對的判不忠實 → 本地小模型 judge 不可靠，需獨立/更強 judge 校準。
6. **多輪意圖延續會誘發幻覺（已修）**：mt6「台積電股價→那鴻海呢」第二輪 4B 沒呼叫 stock_price、直接在 final 編出 112.50 元/+23.7%（實際 257.5）→ 多輪 context 誘發「憑記憶作答」。先被 answer-check 抓到，再以 **`core.verify_numbers` 護欄修正**（與 `verify_citations` 對稱：頁碼要在檢索集、數值要在「工具結果＋題目＋歷史」內）：agent final 含未溯源數字 → 退回逼一次工具查證；更正後仍沒查工具就**拒答不展示假數字**。實測：有時逼出 stock_price 得 257.5、有時拒答，**裸幻覺＝0**。誠實邊界：能否救回真值非決定性（取決於 4B 是否聽從更正），但「不展示未溯源數字」是硬保證。
7. **雲端 demo reranker 冷啟動下載（已修）**：Docker 化後首次 `/ask` 卡 ~140s、甚至 500——root cause：`ingest.py` build 時只快取 embedder，**reranker（bge-reranker-base）在 runtime 首次請求才從 HF Hub 下載**（未認證下載被限速 → 500）。修：Dockerfile build 時多跑一次 `core.retrieve` 暖機把 reranker 預載進 image。實測首次 `/ask` **140s→32s**、runtime 零下載（`Fetching` 0 次）。教訓：lazy-load 的模型容器化要 **build 時預載**，否則冷啟動依賴外部下載＝慢且脆。
8. **Groq 被 Cloudflare 擋（error 1010，已修）**：router/probe 用 `urllib` 打 Groq API 回 `403 error code: 1010`——root cause：Groq API 在 Cloudflare 後，預設 `Python-urllib` User-Agent 被當 bot 擋（**與 key/帳號無關**；Gemini 無此層故正常）。诊断法：讀 403 response body 看到 1010（Cloudflare 碼非 Groq 碼）+ 帶 `User-Agent: Mozilla/5.0` 重打即 200。修：`llm_router._post` 與 `probe.test_groq` 帶 UA。實測雙 provider auth OK + 強制 Gemini 失敗自動 fallback Groq 回答成功。grep sweep 確認其餘 urllib（Ollama localhost / FinMind / Gemini）不受影響。教訓：第三方 API 在 CDN/WAF 後時裸 UA 會被擋，錯誤碼要分清是 provider 還是 CDN。

## 五、已知問題 / 待辦

- **E2 一頁 case study 未產出**（把上面 failure cases 寫成顧問敘事；補 JD「需求→AI 方案」軸線）。
- **C1 雲端 router ✅ 已實作並端到端實證**（`llm_router.py`：`CITERAG_LLM_BACKEND=cloud`；Gemini Flash-Lite 主 + Groq fallback；逐家 json 模式對映 + 壞 JSON 歸零；離線 8 例 + Gemini 金鑰實連 auth OK + 容器內 `/ask` 全鏈路 200 帶引用）。Docker 化（`Dockerfile`/`.dockerignore`/`.env.example`/`DEPLOY.md`）build+smoke 驗證過。Groq fallback 金鑰已設定並實測（Cloudflare UA 修復後雙 provider 皆 auth OK、fallback 切換實證）。**待**：HF Spaces push（你的 HF 帳號）+ 公開前重生 Gemini 金鑰（曾外洩）。VLM 讀圖仍走本機（cloud demo 不含）。
- 測試集偏小（檢索 n=12 / golden 28）、僅 2 份 PDF → 擴題是 reranker/hybrid 終局與收窄 CI 的前提。
- `TOP_K=5`（已對齊 eval MAIN_K）；n8n/Dify、STT/TTS/影像生成 未碰（多模態 2/5）。
- 多輪只塞前文 Q/A、上限 3 輪；長對話可靠度未驗。multi-turn golden 已建（n=7 follow-up 偏小、CI 寬）→ 可擴題收窄、補 RAG 多輪與更多 recency/負例。
- `w0_results_agent.json` 為舊 schema，待重跑刷新。

## 六、檔案

- `PLAN.md`(SSOT) / `CLAUDE.md`(守則) / `HANDOFF.md`(本檔) / `README.md`(對外) / `requirements.txt` / `.gitignore`
- 部署：`Dockerfile`（cloud 預設、build 時建索引+暖機 reranker） / `.dockerignore` / `.env.example` / `DEPLOY.md`（HF Spaces 步驟）
- `rag/`：`core.py`（引擎：解析/嵌入/FAISS/**BM25+RRF**/重排/生成/`verify_citations`/**`verify_numbers`**/`vlm_b64`）/ `agent.py`（5 工具）/ **`findata.py`**（FinMind 結構化查詢：lookup/compare/stock_price）/ **`pgstore.py`**（pgvector 向量資料庫後端，可切換） / **`llm_router.py`**（生成端路由：本機 Ollama / 雲端 Gemini+Groq fallback，`CITERAG_LLM_BACKEND`） / `app.py`(Gradio) / `api.py`(FastAPI) / **`web/`**(自製前端) / `stats.py` / **`golden.py`+`golden.jsonl`** / `ingest.py` / `ask.py`
  - eval：`eval_retrieval.py` / `eval_rerank.py` / **`eval_hybrid.py`** / `eval_generation.py` / **`eval_rag_triad.py`** / `eval_agent.py` / `eval_agent_hard.py` / **`eval_multiturn.py`＋`golden_multiturn.jsonl`**（多輪解析 ON/OFF ablation，結果 `multiturn_results.json`）
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
