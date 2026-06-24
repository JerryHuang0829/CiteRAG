# 文件智能助手（Document Intelligence Copilot）專案計畫書

- 版本：v2.0
- 日期：2026-06-22
- 定位：AI 應用工程師（AI Systems / AI Engineer）轉職作品集 — **學習導向**
- 開發機：**Acer/ROG 備援機 i7-1260P（CPU-only、Intel UHD 內顯、16GB RAM）**
- 預算：**零預算（核心純本地，雲端僅用免費額度）**
- 工作碟：**E:（218GB；C: 僅剩 33.6GB，模型/容器/cache 一律放 E:）**

---

## 0. 文件目的

把這個 side project 從「想法」固化為「可執行、可交付、可展示」的單一事實來源（single source of truth, SSOT）。每完成一個週末 checkpoint 即回頭更新本文件的進度與**本機實測數字**（不可寫估計值充當實測）。

> 本版相對 v1.0 的根本變更：(1) 開發機從 RTX 4080 改為 **CPU-only i7-1260P**；(2) **零預算**；(3) 定位從「貼合單一公司製造業」改為**學習導向、通用、可投多家**；(4) 模型全面下修到 CPU 可跑的尺寸；(5) **Agent 設計從「自由多步 ReAct」改為「程式碼編排單步 + grammar + router 外送」**——這是讓 agent 在 4B/CPU 上「可用 vs 不可用」的分水嶺。

---

## 1. 定位與目標

### 1.1 一句話定位

以一套**通用 RAG + Agent 引擎**為核心（換語料即可重投多家公司），做成「**帶頁碼引用的文件問答 + 多步驟工具代理**」的助手。預設語料為**金融/投資公開文件**，但語料可抽換（維運手冊、技術文件、法規皆可）。

### 1.2 主要目標：學習 + 轉職 ready

1. **學會**這套 AI 應用工程框架（RAG / Agent / eval / 本地部署 / observability / 模型選型）——這是首要目的。
2. 產出一個**可投履歷、能撐面試深挖**的作品集，目標職類為 AI 應用 / AI Systems 工程師（不為單一公司客製）。

### 1.3 設計策略：通用引擎（A 骨）優先，外皮（B 皮）為後

- 開發期只做**通用引擎**（A 骨）：RAG + Agent + eval + router + 部署，投 N 家都能用。
- 多模態外皮（VLM 讀圖、語音）列為彈性尾，行有餘力或確認標的後再補。

### 1.4 差異化主軸（這台機器的限制反而是賣點）

CPU / 單機 / 零預算 / 小模型的限制，**正是 AI 應用工程師的真實日常**。把它做成有數據的工程判斷：
- **模型選型 trade-off**：0.6B/1.7B/4B/8B、bge-small vs bge-m3、reranker base vs v2-m3、本地 vs 免費雲端。
- **小模型工程護欄**：發現 4B 無法自由多步 → 建確定性編排層 → 量化成功率改善（這是「真實 failure → 工程解法 → 量化改善」的差異化故事）。
- **嚴謹 eval**：recall@k/nDCG/faithfulness + bootstrap 信賴區間（多數候選人缺的統計嚴謹度）。

---

## 2. 硬體與資源現實（本版核心約束）

| 項目 | 規格 | 意義 |
|---|---|---|
| CPU | i7-1260P（4P+8E / 16 線程、~28W 行動、AVX2、**無 AVX-512**） | 只能 CPU 推論 |
| GPU | Intel UHD 內顯（無 CUDA，疑單通道致降級） | **不用於推論**（memory-bandwidth-bound，iGPU 打平或輸 CPU） |
| RAM | 15.6 GB | 新硬瓶頸；可用約 10-11GB 給工作負載 |
| 磁碟 | C: 33.6GB / **E: 218GB** | 模型/Docker/cache 全搬 E: |
| OS | Windows 11 Pro 64-bit | Ollama 原生 Windows、Docker Desktop(WSL2 backend) |

**速度預期（估計，W0 必實測校正）**：Qwen3-4B Q4_K_M decode 約 **6-10 tok/s**（單通道下緣）；prefill 約 **40 tok/s**（agent 延遲主因）。所有數字在 W0 用 `llama-bench` / `ollama run --verbose` 實測後才寫入文件。

**最高 ROI 硬體查核**：RAM 是否單通道（工作管理員→效能→記憶體）。若單通道且可加 SODIMM，補一條變雙通道 → 頻寬 +30-50% → CPU tok/s 約 ×1.3-1.5、iGPU 升 Iris Xe。**屬可選**（需花一條 RAM 錢、且須確認非焊死）。

---

## 3. 系統架構

```
[使用者]
  │ ① 文字提問   ② 上傳圖片(銘牌/儀表)   ③ (彈性)語音
  ▼
┌──────────────────────────────────────────────┐
│  Web UI (Gradio，streaming 輸出)               │
└───────────────┬────────────────────────────────┘
                ▼
┌──────────────────────────────────────────────┐
│  Orchestration 層（你的 Python 程式碼編排流程） │
│  ‧ 決定步驟順序（非讓 LLM 自由 ReAct）          │
│  ‧ 每步只問 LLM 一件小事（單步 tool/slot）      │
│  ‧ grammar 約束 JSON + Pydantic 驗證 + 重試     │
│  ‧ Router：難步驟/視覺 → 免費雲端(env 開關)     │
└───┬───────────┬───────────┬────────────────────┘
    ▼           ▼           ▼
 [VLM 讀圖]  [RAG 引擎]   [mock 工具]
 雲端 Gemini  retrieve     開工單/查設備
 (本地Gemma   →rerank      (純函式)
  對照)       →cite         │
    │           ▼           │
    │      [向量庫 Qdrant/FAISS]
    │           ▲
    │      [embedding: fastembed+bge-small-zh(線上) / bge-m3(離線)]
    │           ▲
    │      [文件庫：金融/投資公開文件 PDF]
    ▼
 [LLM 推論層]
  本地 Ollama: Qwen3-4B-instruct(主力) / 1.7B / 0.6B / 8B(對照)
  雲端(免費,env 開關): Gemini Flash / Groq
```

主資料流（旗艦情境）：拍銘牌照 →〔VLM〕讀型號 → 向量庫比對手冊 →〔RAG+LLM〕答步驟+頁碼 →〔Orchestration〕使用者確認後開 mock 工單。**每一步順序由程式碼控制，LLM 只做單步決定。**

---

## 4. 技術選型（CPU / 零預算 stack）

| 階段 | 選擇 | 套件 / tag | 註記 |
|---|---|---|---|
| LLM 主力 | Qwen3-4B-instruct（2507 非思考、Q4_K_M） | `qwen3:4b-instruct` | 常駐溫熱；**禁用 thinking 版**（CPU 上延遲翻倍） |
| LLM router 對照 | Qwen3-1.7B / 0.6B（快弱）、8B（準慢、不常駐） | `qwen3:1.7b` `qwen3:0.6b` `qwen3:8b` | 選型/router demo |
| VLM 讀圖 | 雲端 Gemini Flash（主）/ 本地 Gemma 3:4b（對照） | `gemma3:4b` + Gemini 免費 | 本地數十秒/張，僅對照 |
| Embedding 線上 | fastembed + bge-small-zh-v1.5（384維） | `fastembed` | CPU 快、省 RAM |
| Embedding 離線 | bge-m3（多語、dense+sparse） | `FlagEmbedding`/ONNX | 對繁中較穩，建索引/對照用 |
| Reranker | bge-reranker-base（線上、可開關）/ v2-m3（離線對照） | `sentence-transformers` | 小語料最大品質槓桿 |
| 向量庫 | FAISS（起步）→ Qdrant 單容器（on_disk） | `faiss-cpu` / `qdrant-client` | |
| Agent runtime | Ollama 原生 Windows CPU | `ollama` | host 跑（非容器）；`format=json` |
| 雲端臂（可選） | Gemini Flash 1500 RPD / Groq 14400 RPD | `google-genai` / `groq` | env 開關、provider 抽象 |
| Eval | RAGAS + DeepEval + 自寫指標 + bootstrap | `ragas` `deepeval` `scipy` | 評測放獨立 conda env |
| Web UI | Gradio（streaming、多模態上傳） | `gradio` | |
| Observability | Arize Phoenix（in-process） | `arize-phoenix` | 取代 Langfuse 6 容器棧（16GB 會 thrash） |
| 部署 | docker-compose（app + Qdrant + Phoenix；Ollama 在 host） | Docker Desktop | host.docker.internal 連 Ollama |
| 設定/祕密 | pydantic-settings + python-dotenv | | .env 進 .gitignore |
| STT（彈性） | faster-whisper small/base int8 | `faster-whisper` | 非核心 |

Ollama CPU 設定：`OLLAMA_MODELS=E:\ollama\models`、`OLLAMA_KEEP_ALIVE=30m`、`OLLAMA_NUM_PARALLEL=1`、`OLLAMA_FLASH_ATTENTION=1`、`num_ctx=4096`、`num_thread=8`（W0 用 6/8/10 校正）。

---

## 5. Agent 設計（核心，決定可用 vs 不可用）

**鐵律：不讓 4B 自由多步 ReAct。** 4B 自由多步端到端成功率約 15-20%；單步 function-calling 約 90%。設計就是去吃單步那 90%。

### 5.1 程式碼編排單步（code-orchestrated single-step）
- 流程順序寫在 Python（你規劃），不交給 LLM 自由決定。
- 每步只暴露 1-2 個工具，只問 LLM「這一步輸出哪個工具 + 參數」一個決定。
- 旗艦流程的步驟（讀型號 → 查手冊 → 生成 → 確認開單）由 orchestration 串。

### 5.2 護欄（必做，缺一即可能翻車）
- **grammar / constrained decoding**：`format=json` 或 GBNF → 壞 JSON 歸零（不增延遲）。
- **Pydantic 驗證**：每步驗 tool name + 參數型別 + 必填；失敗重試 1 次後 fallback。
- **max_turns（如 6）+ 重複呼叫偵測** → 擋無限迴圈。
- **router 逃生艙**：難的規劃步驟 / 視覺 → env 開關外送免費 Gemini/Groq，本地 4B 只做簡單步。
- **引用機制驗證**：後處理檢查「被引頁碼/chunk_id ∈ 本次檢索命中集合」，對不上即標記/拒答（擋 citation-shaped 幻覺）。

### 5.3 可用 agent 的最低門檻（須同時滿足，否則降級）
1. 自家工具集端到端成功率 **≥80%**（跑 20-30 次 + bootstrap CI；≥70% 才勉強進場）。
2. 單任務（≤3 步）wall-clock **≤90 秒**、每步 ≤30 秒（放棄即時互動，定位半互動/可錄影）。
3. JSON 合法率 100%、max_turns 內必終止。
4. 頁碼命中率 ≥90%、faithfulness 附 judge 校準討論。
5. 16GB 下重型服務序列化、demo 動線不觸發 swap。

達不到 (1)(2) → 主秀降為「RAG QA + router 對照」，多步 agent 當輔助章節。

---

## 6. RAG 與 Eval 設計

### 6.1 RAG pipeline
PDF（PyMuPDF4LLM 帶頁碼）→ chunking（Recursive / Markdown header，512-1024 token + overlap）→ embedding（線上 bge-small-zh）→ 檢索（FAISS/Qdrant；離線可加 hybrid BM25+dense, RRF）→ rerank（bge-reranker-base，top-20→top-3~5）→ LLM 生成帶頁碼引用。

### 6.2 三層 Eval（學習乘數 + 差異化）
- **Retrieval（自寫）**：recall@k、MRR、nDCG@k（numpy，徹底理解定義）。
- **Generation**：faithfulness、answer relevance、citation 正確率（RAGAS / DeepEval，LLM-as-judge）。
- **Agent**：tool-selection accuracy、false-action rate（DeepEval ToolCorrectness）。
- **統計嚴謹（你的差異化）**：對 per-query 分數做 **bootstrap 95% CI**（`scipy.stats.bootstrap`）；報「recall@5 = 0.82 [0.74, 0.89], n=50」而非裸數字。
- **judge 誠實標籤**：本地小模型 judge 偏差大 → 雙軌（本地 + 免費雲端）對照、報 inter-judge agreement。

### 6.3 Ablation（變成展示內容而非成本）
chunking 策略、reranker 加/不加、bge-small vs bge-m3、4B vs 8B、本地 vs 雲端——每個都做成有數據的對照。

---

## 7. W0 Benchmark Gate（動工前必測，決定野心大小）

**規則：以下數字未在本機實測，不得寫入作品集文件**（呼應「未實測就寫文件」教訓）。W0 量完才決定 agent demo 規模。

1. RAM 通道數（單→雙通道 +30-50% 頻寬，最高 ROI）。
2. Qwen3-4B Q4_K_M 的 **prefill tok/s @ 2K/3K/4K** 與 decode tok/s（thread 6/8/10 取最快）。
3. 自家 3 步 mock 工單 task 在 `format=json` 下端到端成功率（20-30 次 + bootstrap CI）。
4. grammar 開啟前後：JSON 合法率 vs 語意正確率分開量。
5. 典型 RAG prompt 的 TTFT；KV cache 是否跨 turn 保留。
6. bge-small-zh vs bge-m3 在**自己的繁中語料**的 recall@k（公開只有簡體 benchmark，繁中錯配是最大盲區；可加 OpenCC 轉簡 A/B）。
7. Gemma 3:4b 本地讀一張銘牌端到端秒數；16GB 下重型服務同跑峰值 RAM。
8. 免費雲端臂在台灣 IP 可否直連 Gemini/Groq。

**決策**：成功率 ≥80% 且任務 ≤90s → agent 多步當主秀；<70% 或 >3 分鐘且雲端臂壓不下 → 主秀改 RAG+router，agent 縮為學習章節。**專案本身是 GO，只有 agent 野心隨實測伸縮。**

---

## 8. 執行路線（里程碑）

原則：每個 checkpoint 都有可演的東西；核心前置；W4 為 MVP gate。日期暫定（從開工首個週末起算，可調）。

| 週末 | 核心交付 | 能秀 | 狀態 |
|---|---|---|---|
| **W0 環境+gate** | E: 搬遷、Ollama+模型、§7 全部 benchmark、免費雲端連通 | 本機跑通一個 LLM + 一張實測數字表 | gate |
| **W1 最小 RAG** | 語料→PyMuPDF4LLM→chunk→bge-small→FAISS→帶頁碼答（streaming） | 「問→答+頁碼」 | Demo 1 雛形 |
| **W2 RAG eval** | 50 題測試集、recall@k/nDCG（自寫）+ faithfulness + bootstrap CI + chunking/reranker ablation | eval 報告 | ✅ Demo 3 |
| **W3 Agent（編排單步）** | orchestration + grammar + Pydantic + max_turns + 引用驗證 + router 逃生艙 + Phoenix | 「問→查→步驟→開工單」 | ✅ Demo 1（文字） |
| **W4 Router/選型** | 本地 0.6B/1.7B/4B/8B vs 免費雲端：品質/延遲/成本對照 + 決策表 | Demo 4 ⭐ | ✅ **MVP gate（可投）** |
| W5 廣度（彈性） | FAISS→Qdrant；VLM 雲端 Gemini + 本地 Gemma 對照讀銘牌→RAG；(可選 STT) | Demo 1 完整 + 多模態 | 彈性 |
| W6 打包（彈性） | docker-compose（app+Qdrant+Phoenix，Ollama host）、README+架構圖、**一頁 failure case**、2 分鐘影片 | Demo 5 + Demo 6 | 彈性 |

**MVP**：到 W4 即有 Demo 1（文字）/ 3 / 4 可投履歷。W5-W6 為補完整套的加分。

---

## 9. 展示項目（Demo）

| # | Demo | 評級 | 說明 |
|---|---|---|---|
| 1 | 文件問答（帶頁碼引用） | ✅ 完整 | 檢索毫秒級，與算力無關 |
| 3 | RAG eval（三層 + bootstrap CI） | ✅ 完整 | 你的統計嚴謹度差異化 |
| 4 | 本地 vs 雲端 router + 選型對照 | ✅ 完整 | CPU 慢反而讓「何時升級/外送」數據更真實 |
| 1' | 多步 Agent（編排單步） | ⚠️ 降級但可用 | 靠編排+grammar+router；野心由 W0 定 |
| 5 | docker-compose 本地部署 + observability | ✅ 完整 | 「純本地離線可跑」隱私故事更純 |
| 2 | VLM 讀銘牌 / (彈性)語音 | 🔶 雲端保底 | 本地 Gemma 對照、主路徑雲端 Gemini |
| 6 | 顧問式 case study + failure case | ✅ 完整 | failure→根因→修復→數字 |

---

## 10. 風險與對策

| 風險 | 影響 | 對策 |
|---|---|---|
| 4B 自由多步成功率 ~16% | agent 不可用 | **§5 編排單步 + grammar + router 外送**（最關鍵） |
| prefill 延遲（每步重吃 context） | 單任務數分鐘 | top-k 限縮、chunk 精簡、streaming、warm model、≤3 步 |
| 16GB 並發爆 RAM | swap 拖垮全機 | 重型服務序列化、離線預建索引、Phoenix in-process |
| C: 33.6GB 爆掉 | 開發中斷 | §12 四項全搬 E: 後才下載模型 |
| 繁中腳本錯配（簡體訓練模型） | recall 掉、盲區 | 離線臂用 bge-m3、W0 用自己語料實測 + OpenCC A/B |
| 免費雲端額度/區域波動 | router 雲端臂掛 | provider 抽象、env 開關、本地 fallback、只送公開資料 |
| 估計值當實測寫文件 | 可信度崩 | §7 gate：未實測不得寫；tok/s 標單/雙通道 |
| 範圍爆炸（週末 + 學習曲線） | 跑不完 | 嚴守 W4 MVP gate，W5-W6 可砍 |

---

## 11. 成功標準（Definition of Done）

- [ ] W0 benchmark 數字表（RAM 通道、prefill/decode tok/s、自家工具成功率 + CI）
- [ ] `docker compose up` 本地離線跑起 app + Qdrant + Phoenix（Ollama host）
- [ ] RAG eval 報告：recall@k/nDCG/faithfulness + **bootstrap CI**（改善前後）
- [ ] Agent 達 §5.3 可用門檻（或誠實降級並記錄原因）
- [ ] 本地 vs 雲端 router 對照含實測延遲/成本/品質 + 決策表
- [ ] 一頁 failure case（真實踩坑 → 量化根因 → 修復 → 數字 a→b）
- [ ] 2 分鐘 demo 影片（含預錄後備，避免現場單次延遲毀 demo）
- [ ] 乾淨 GitHub repo（README、架構圖、執行說明；模型權重不入庫）
- [ ] 最低門檻：W4 即有 Demo 1（文字）/ 3 / 4 可投

---

## 12. 附錄

### A. E: 槽搬遷清單（W0，下載模型前完成）
- `OLLAMA_MODELS = E:\ollama\models`（使用者環境變數，設後重啟 Ollama/重開機驗證）
- `HF_HOME = E:\hf`（+ `HF_HUB_DISABLE_SYMLINKS=1`）
- Docker Desktop → Settings → Resources → Disk image location → `E:\Docker`
- conda：`.condarc` 加 `envs_dirs: [E:\conda\envs]`、`pkgs_dirs: [E:\conda\pkgs]`；新 env 用 `--prefix`
- `.wslconfig`：`[wsl2] memory=4GB / processors=6 / autoMemoryReclaim=gradual`

### B. conda env 切分
- `rag`（核心）：torch(CPU) → fastembed → pymupdf4llm → faiss-cpu → qdrant-client → langchain-text-splitters → ollama → google-genai/groq → gradio → pydantic-settings
- `rageval`（評測，獨立）：ragas + deepeval + scipy + arize-phoenix
- 對照線（隔離）：sentence-transformers/FlagEmbedding（bge-m3/reranker 離線）、langgraph（agent 框架對照）

### C. 資料來源（全公開、零成本、注意授權）
- 金融/投資：公開財報（MOPS 公開資訊觀測站）、法說會逐字稿、證交所/櫃買規章（公開文件，避免版權教科書入公開 repo）
- （備選語料）技術文件官方 docs、台灣法規、工業設備手冊
- 銘牌/儀表照片：自拍 + 公開圖庫（只送公開資料給免費雲端）

### D. 模型 tag 速查
`qwen3:4b-instruct` / `qwen3:1.7b` / `qwen3:0.6b` / `qwen3:8b` / `gemma3:4b`；embedding `BAAI/bge-small-zh-v1.5`、`BAAI/bge-m3`；reranker `BAAI/bge-reranker-base`、`BAAI/bge-reranker-v2-m3`。

### E. 事實校正
- i7-1260P（Alder Lake mobile）**無 AVX-512**（部分網路來源誤稱有）。
- 「16GB 可跑 12B」多指 16GB VRAM/unified memory，非 16GB CPU 系統 RAM。

---

## 變更紀錄

| 版本 | 日期 | 變更 |
|---|---|---|
| v1.0 | 2026-06-15 | 初版：RTX 4080 + 6 Demo + W0-W6（製造業外皮、本地 GPU 部署） |
| v2.0 | 2026-06-22 | 改 CPU-only i7-1260P + 零預算；定位轉學習導向/通用；模型下修（Qwen3-4B/Gemma3:4b/bge-small/Phoenix）；**Agent 改編排單步+grammar+router 外送**；新增 W0 benchmark gate；語料預設金融文件 |
