# CiteRAG — 帶頁碼引用的本地 RAG + Agent 文件問答系統

一套 **CPU-only、零雲端** 的繁體中文文件問答系統：把 PDF 變成「**帶頁碼引用、會拒答、擋幻覺**」的問答助手，並延伸出多工具 Agent 與視覺讀圖。重點不在堆語料規模，而在把一條 RAG 鏈路的每個工程決策**量化、可重現、能擋退步**。

整條鏈路（解析 → 切塊 → 嵌入 → 檢索 → 重排 → 生成 → 護欄 → 評測）皆手刻、不依賴 LangChain/RAGAS 等高階框架黑盒，以便完全掌握每一層的行為與取捨。

---

## 為什麼值得看

- **Hybrid 檢索**：dense 向量 + BM25 稀疏，經 RRF 融合 — 補純向量對精確詞/數字/股號的 silent fail。並用配對 bootstrap 量化其效益（檢索層 **ΔMRR +0.229，95% CI 不含 0**）。
- **引用幻覺護欄**：後處理剝除「不在本次檢索命中集合」的頁碼引用 — 擋 citation-shaped 幻覺（含多頁 `(p.3,4)` 逐頁判定）。
- **三層評測 + 信賴區間**：自寫 recall/MRR/nDCG、RAG Triad（faithfulness / context precision / recall）、Agent trajectory，全部附 **bootstrap 95% CI**，並對 0/1 退化情形改報 rule-of-three（不印假的 `[1,1]`）。
- **評測進 CI**：GitHub Actions 在每次 push 跑確定性測試（含護欄、統計、檢索原語、golden set schema），低於門檻擋下 — 防 regression。
- **誠實標籤**：小語料下若某優化端到端無顯著差異、或本地小模型當 judge 有偏差，一律照實報並說明原因，不誇大。

---

## 系統架構

```
                ┌──────────────────────────────────────────────────────────┐
  使用者  ──▶   │  介面層： Web UI(Gradio) ／ 自製前端＋REST API(FastAPI) ／ CLI │
                └───────────────────────────┬──────────────────────────────┘
                                            ▼
                ┌──────────────────────────────────────────────────────────┐
                │  RAG 引擎 (rag/core.py)                                    │
                │                                                          │
                │   query                                                  │
                │     ├─▶ 檢索   dense(FAISS) ＋ BM25(稀疏) ──RRF 融合──▶ 候選 │
                │     ├─▶ 重排   bge-reranker cross-encoder（top-N → top-k） │
                │     ├─▶ 生成   Ollama qwen3:4b（只依參考資料、帶頁碼、拒答） │
                │     └─▶ 護欄   剝除非檢索命中的頁碼引用（擋幻覺）            │
                └───────────────────────────┬──────────────────────────────┘
            ┌───────────────────────────────┼───────────────────────────────┐
            ▼                               ▼                               ▼
   Agent（程式碼編排單步）           VLM 讀圖（Gemma 3:4b）          評測（rag/eval_*.py）
   工具 + 參數驗證 + max_turns       影像 → 文字                    三層指標 + bootstrap CI
   + 引用護欄 + 強制收尾                                            + golden set + CI gate
```

**離線索引建立**：PDF → PyMuPDF4LLM（帶頁碼）→ RecursiveCharacterTextSplitter → bge-small-zh 嵌入 → FAISS。

---

## Agent 工具：結構化 DB（FinMind）+ RAG 混合

關鍵設計：**數字 ≠ RAG**。財報數字、股價、跨公司比較是「結構化資料」，用 RAG（語意檢索 PDF）既不精確、也無法彙總/比較；應走資料庫。Agent（qwen3:4b）讀問題後**自動分流**到 5 個工具：

| 工具 | 用途 | 後端 | 範例問題 |
|---|---|---|---|
| `search_filings` | 文件原文 / 解釋 | RAG（你的 PDF） | 「鴻海為什麼毛利率下滑」 |
| `lookup_metric` | 單一精確財務數字 | FinMind 資料庫 | 「台積電 2023 EPS」 |
| `compare` | 跨公司比較 / 排名 / 篩選 | FinMind 資料庫 | 「誰 EPS 最高」「哪些毛利率>30%」 |
| `stock_price` | 即時 / 近期股價 | FinMind 資料庫 | 「台積電股價、近一年漲幅」 |
| `create_note` | 建立追蹤筆記 | 記憶體 | 「順便記一筆」 |

結構化工具（lookup/compare/stock_price）涵蓋**全市場 ~2500 家**（不限手上 PDF）、精確零幻覺、可彙總；RAG 工具答「為什麼 / 怎麼說」的質性內容。FinMind 查詢帶磁碟快取以節省 API 額度。**這就是「數字走 DB、文字走 RAG、agent 自動分流」的混合架構。**

---

## 技術選型

| 階段 | 選擇 | 說明 |
|---|---|---|
| PDF 解析 | PyMuPDF4LLM | 轉 markdown、保留頁碼 |
| 切塊 | RecursiveCharacterTextSplitter（500/80） | 中文標點為分隔點 |
| 嵌入 | `BAAI/bge-small-zh-v1.5`（fastembed/ONNX, CPU） | 繁中、省 RAM |
| 稀疏檢索 | 手刻 Okapi BM25（中文 bigram + 數字整段 token） | 補精確詞 silent fail |
| 向量庫 | FAISS `IndexFlatIP`（正規化內積＝cosine） | 78 chunks 用暴力精確搜＝正解，非過早上向量 DB |
| 重排 | `BAAI/bge-reranker-base`（cross-encoder） | 小語料最大品質槓桿 |
| LLM | Ollama `qwen3:4b-instruct`（CPU） | 帶頁碼引用生成、`format=json` agent |
| VLM | Ollama `gemma3:4b` | 影像理解（讀銘牌/儀表） |
| 介面 | Gradio／FastAPI＋自製前端／CLI | 同源服務、無 CORS |
| 評測 | 自寫指標 + 純 stdlib bootstrap | 可重現、固定 seed |

LLM/VLM 推論走外部 **Ollama** 服務；嵌入與重排走 **fastembed**（CPU）。統計、BM25、引用護欄為純 stdlib，無額外依賴。

---

## 快速開始

前置：[Ollama](https://ollama.com)、conda、約 4GB 磁碟（模型）。

```bash
# 1) 環境
conda create -n rag python=3.12 -y && conda activate rag
pip install -r requirements.txt

# 2) 拉模型（Ollama 需在背景執行）
ollama pull qwen3:4b-instruct
ollama pull gemma3:4b

# 3) 建索引（讀 data/*.pdf → FAISS；首次會自動下載 bge 嵌入/重排權重）
cd rag
python ingest.py

# 4) 任選一種介面執行（皆需 cwd 在 rag/）
python app.py                                  # Gradio 三分頁 UI  → http://127.0.0.1:7860
uvicorn api:app --port 8000                    # REST API + 自製前端 → http://127.0.0.1:8000/app
python ask.py "鴻海 2022 全年 EPS 是多少？"       # CLI 問答
python agent.py "查鴻海 EPS 並幫我記一筆筆記"      # CLI Agent
```

**REST API**：`POST /ask`、`POST /agent`、`POST /vlm`、`GET /health`（外部依賴失敗回 503，非 500）。

預設語料為兩份公開文件（鴻海 2022Q4 法說會逐字稿、金管會興櫃市場專題）。**換語料**：把 PDF 放進 `data/`、重跑 `python ingest.py` 即可。

---

## 評測報告卡

語料：兩份公開 PDF、78 chunks。所有點估計附 bootstrap 95% CI（固定 seed、可重現）。

### 檢索（n=12，`rag/eval_retrieval.py` / `eval_rerank.py` / `eval_hybrid.py`）

| 設定 | hit@3 | recall@5 | MRR | nDCG@5 |
|---|---|---|---|---|
| dense（無重排） | 0.750 | 0.651 | 0.562 | 0.552 |
| dense ＋ **hybrid(BM25/RRF)** | 0.917 | 0.746 | **0.792** | **0.714** |
| dense ＋ **bge-reranker** | 1.000 | 0.868 | 0.833 | 0.813 |

- **Hybrid（檢索層）配對 bootstrap**：ΔMRR **+0.229** CI[+0.076, +0.396]（顯著）、ΔnDCG@5 **+0.163** CI[+0.029, +0.305]（顯著）。
- **Reranker 配對 bootstrap**：bge-reranker ΔMRR **+0.271** CI[+0.014, +0.507]（顯著）。
- 誠實註記：**hybrid 的效益在檢索層顯著、但端到端（再經 reranker）在此 78-chunk 小語料被洗平**（top-N 候選佔語料 26%，reranker 能補回 dense 漏召）。其端到端 ROI 隨語料規模上升 — 大語料時 gold 可能不在 dense top-N，reranker 救不回不在候選池的 chunk。

### 生成 — RAG Triad（golden set 28 題，`rag/eval_rag_triad.py`）

版本化 golden set（25 可答 / 3 應拒答、5 類別）。程式硬驗指標客觀可重現：

| 指標 | 值 | 說明 |
|---|---|---|
| answer correctness | **0.893** CI[0.786, 1.000] | 答案含正解（refuse 題＝正確拒答） |
| context recall | **1.000** | gold 是否被撈進 context（檢索層幾乎不漏） |
| context precision | 0.490 | 檢索 chunk 含 gold 的比例 |
| faithfulness（4B judge） | 0.667 | ⚠ noisy proxy（見下） |
| answer relevancy（judge） | 1.000 | noisy proxy |

分類 answer correctness：factual 1.000(12)、qualitative 1.000(4)、exact-term 0.875(8)、refuse 0.667(3)、trap 0.000(1)。

**RAG Triad 把「哪裡好/哪裡壞」解耦**：context recall=1.000 代表**檢索不是瓶頸**（gold 都撈到了）；少數失分集中在生成端（如資本支出題：979 已在 context 卻沒被正確抽出）與拒答紀律。而 faithfulness(4B judge)=0.667 明顯低於 0.893 的客觀正確率 — 抽查發現 judge 把多題「答對且有依據」也判為不忠實，**再次印證本地小模型當 judge 不可靠**（deterministic 指標揭穿主觀 judge）；故 faithfulness 僅供相對參考、需更強 judge 校準。

### Agent（`rag/eval_agent_hard.py`）

- 程式碼編排單步 function-calling，JSON 合法率高、終止可靠。
- 難情境（拒答／fallback／長鏈／條件式／模糊，8 情境）：**task success 0.875** CI[0.625, 1.000]。

---

## 設計決策（為什麼這樣做）

- **為什麼 78 chunks 用 FAISS flat 而非向量 DB**：暴力精確搜在此規模是最佳解；過早上 Qdrant/HNSW 是 over-engineering。能清楚說出「何時該升級」（增量、權限、規模）的觸發條件。
- **為什麼 Agent 用程式碼編排單步、而非自由 ReAct**：4B 自由多步端到端可靠度低；把流程順序寫在程式、每步只讓模型做一個決定（選工具+填參數），配 `format=json` + 參數型別驗證 + max_turns + 重複呼叫偵測 + 強制收尾，吃「單步高可靠」那段。
- **為什麼加引用護欄**：LLM 會產生「格式正確但來源錯」的頁碼（citation-shaped 幻覺）。後處理比對「被引頁碼 ∈ 本次檢索命中頁集合」，不在集合內即剝除 — runtime 真實生效，不只在評測。
- **為什麼自寫評測而非 RAGAS**：手刻 recall/MRR/nDCG 與 RAG Triad，是為了**徹底理解每個指標在量什麼**，而非把 `evaluate()` 當黑盒；指標定義對齊業界標準。

---

## Failure cases（踩坑 → 根因 → 修復）

真實的工程紀錄，比「一切都好」更有參考價值（完整顧問式情境→方案→成效→踩坑見 **[docs/case_study.md](docs/case_study.md)**）：

1. **頁碼引用造假**：早期取錯 metadata key 導致頁碼為 `-1`，但 LLM 仍吐出似是而非的頁碼 → 定位真實頁碼在 `metadata['page_number']` 修正，並加上 runtime 引用護欄。
2. **Reranker 結論在小樣本不穩定**：索引由 85→78 chunks（過濾圖片佔位）重建後，reranker 的 A/B 排序翻轉 → 證明 n=12 不足以下 reranker 終局裁定，需擴測試集；預設改為在當前資料較穩的 bge-reranker-base，並降級「顯著性」宣稱。
3. **Hybrid 被 reranker 遮蔽**：hybrid 在檢索層顯著、端到端卻打平 → 拆成兩層級量測才看清，並理解其效益隨語料規模上升。
4. **小模型條件式推理**：4B 對「若 X 才做 Y」的條件式任務易失敗（且會把民國年誤算成西元）→ 標記為「難步驟外送雲端」的明確案例。

---

## 評測 CI

`.github/workflows/ci.yml` 在每次 push / PR 跑 `pytest -m "not local"`：

- **雲端（確定性、秒級、無模型）**：統計原語（bootstrap/rule-of-three/percentile）、引用護欄、BM25/RRF、golden set schema。
- **本地（`pytest -m local`）**：需索引/重排的檢索 recall gate、golden set grounding 驗證。需 4B judge 的 faithfulness 門檻因免費 runner 跑不動，由本地評測產出後把關。

意義：改 chunk 大小／reranker／prompt 時，CI 自動攔截 regression。

---

## 專案結構

```
rag/
  core.py            RAG 引擎（解析/嵌入/FAISS/BM25+RRF/重排/生成/引用護欄/VLM）
  agent.py           程式碼編排單步 Agent（工具 + 護欄）
  app.py             Gradio 三分頁 UI       api.py    FastAPI REST + 自製前端（web/）
  ingest.py ask.py   建索引 / CLI 問答
  stats.py           共用統計原語（bootstrap / paired bootstrap / rule-of-three）
  golden.py golden.jsonl   版本化 golden set + 驗證器
  eval_retrieval.py eval_rerank.py eval_hybrid.py    檢索層評測
  eval_rag_triad.py eval_agent.py eval_agent_hard.py  生成/Agent 評測
tests/               pytest 套件（雲端確定性 + 本地 gate）
data/                來源 PDF（公開文件）        index/   FAISS 索引（git 忽略，跑 ingest 重建）
w0/                  動工前 benchmark（速度 / 單步成功率 / 雲端連通）
```

---

## 限制與下一步

- **語料刻意保持小**：78 chunks 是 toy 規模，用以驗證鏈路與工程取捨；不為堆規模而堆。
- 測試集 n=12（檢索）/ 28（golden）偏小，CI 寬 — 擴題是 reranker/hybrid 終局裁定的前提。
- 規劃中（CPU/零預算可做）：本地 vs 雲端 provider 路由（模型選型對照）、增量 ingestion（content-hash 偵測變更）、metadata 過濾。
- 「規模/權限」層（GPU serving、向量 DB 服務化、權限感知檢索、合規）為大語料/多租戶情境所需，本專案不實作。
