# Case Study：把「逐字稿翻找」變成可稽核、可上線、有品質/安全把關的秒級問答

需求 → 方案 → 量化成效 → 生產化 → 踩坑紀錄。一頁看懂這套系統解決什麼問題、為什麼這樣設計、量化結果，以及從「本機研究專案」到「雲端產品」的工程決策。

🔗 **Live demo**：https://jerry0829-citerag.hf.space/app ｜ **Repo**：github.com/JerryHuang0829/CiteRAG

---

## 1. 情境與痛點

**使用情境**：投資研究 / 法遵單位分析師，日常要從**幾十頁的法說會逐字稿、主管機關規章 PDF** 裡找出某個數字或原文（例：「鴻海 2022 第四季毛利率？」「興櫃轉上市櫃幾家？」）。

**痛點**：翻找一個數字要 **5–10 分鐘**，找到還要**人工核對頁碼**才能引用；關鍵字搜尋抓不到語意，且要的是**「答案＋可稽核出處」**而非一堆命中段落。

**核心目標**：自然語言問 → **秒級得到答案＋頁碼出處**；文件沒有的**明確拒答而非編造**。

## 2. 成功標準

| 維度 | 驗收指標 |
|---|---|
| 找得到 | context recall（答案有無被撈進來） |
| 答得對 | answer correctness（答案含正解） |
| 可稽核 | citation 正確率（頁碼落在正解頁） |
| 不亂編 | faithfulness、拒答正確率 |
| **可上線** | 品質 SLO 自動把關、安全紅隊 block_rate |

**約束**：資料敏感（財務/法遵）+ 零預算 → **隱私分層**（見 §3）、CPU、開源/免費。

## 3. 方案決策（為什麼這樣選）

**研究/檢索層**
| 決策 | 為什麼 |
|---|---|
| RAG 而非 fine-tune | 知識會變、要可引用、零標註成本 |
| Hybrid 檢索（向量 + BM25 + RRF） | 純向量對精確詞/數字/股號 silent fail（「2317」「979 億」）→ 加字面比對 |
| cross-encoder 重排 | 小語料品質最大槓桿 |
| 引用護欄（剝除非命中頁碼） | LLM 會產「格式對但來源錯」的頁碼（citation-shaped 幻覺）→ 後處理擋 |
| 程式碼編排單步 Agent | 小模型自由多步 ~16% 不可靠 → 流程寫程式、每步只做一個決定（~90%） |
| 78 chunks 用 FAISS flat、不上向量 DB | 此規模暴力精確搜即最佳；過早上向量 DB 是 over-engineering（可切 pgvector 供 SQL 過濾） |
| 數字走結構化 DB（FinMind）而非 RAG | 財報數字/跨公司比較是結構化資料 → Agent 自動分流「數字→DB、文字→RAG」 |
| 數值溯源護欄 | 多輪時模型會憑記憶編數字 → 每個數字須溯源工具結果，否則逼查證/拒答 |

**生產/部署層（本次新增）**
| 決策 | 為什麼 |
|---|---|
| **隱私分層：embedding/檢索/PII 護欄本機、只有生成走雲端** | 敏感內容不外洩到 LLM；免費主機無 GPU 跑不動本地 LLM → 生成外送免費雲端 API（Gemini 主 + Groq fallback） |
| **雙 provider fallback** | 免費 tier 不穩（2025-12 Gemini 額度無預警砍 50-80%）→ 主掛自動切備援，demo 不開天窗 |
| **eval-as-CI-gate 而非只印分數** | 改 prompt/模型/chunk 會讓品質默默退步、unit test 抓不到 → 用凍結 SLO 在 CI 擋退步的 PR |
| **手刻台灣 PII 護欄而非套 Presidio** | Presidio 是 US-centric；台灣身分證需自訂 recognizer → 手刻**內政部檢核碼 + 信用卡 Luhn**，精準（財報數字零誤判）且零重依賴 |

## 4. 量化成效（golden 28 題 + 檢索 n=12 + 多輪 n=7，皆附 bootstrap CI）

- **找得到**：context recall **1.000** — 檢索不是瓶頸。
- **答得對**：answer correctness **0.893** [0.786, 1.000]（factual/qualitative 類 100%）。
- **找得準**：hybrid 檢索層 **ΔMRR +0.229**（CI 不含 0，顯著）。
- **接得上（多輪）**：對話記憶 vs 無，跨輪代名詞解析率 **Δ +0.571**（配對 bootstrap 顯著）。
- **擋得住（安全）**：OWASP 紅隊 5 攻擊 **block_rate 1.000**（確定性判定，非 LLM judge）。
- **上得線**：Docker 化容器 `/ask` 全鏈路帶頁碼引用；首次請求延遲 **140s→32s**（見 §6-6）。

> 把分析師「翻找 + 核對出處」的 5–10 分鐘，壓到一次秒級問答 + 可點頁碼；且整套**可上線、有品質與安全的自動把關**。

## 5. 生產化：從本機研究專案到雲端產品

- **雲端 LLM router**（`CITERAG_LLM_BACKEND` 切換）：Gemini 2.5 Flash-Lite 主 → 失敗/429/壞 JSON 自動 fallback Groq；純 `urllib` 零新依賴。實測：強制主 provider 失敗會自動切備援並正常作答。
- **CD（持續部署）**：Dockerfile（slim、build 時建索引 + 預載 reranker）→ **Hugging Face Spaces 免費 CPU、$0/月、scale-to-zero**、live URL。
- **CI + eval-gate（P2.1）**：GitHub Actions 3 jobs — 確定性測試 + **檢索品質 gate（每 PR 擋）** + 生成/安全 gate（nightly）。門檻凍結於 `slo.py`，**mutation 驗證「會擋」**（門檻拉高→gate 紅燈 exit 1）。
- **安全 / OWASP LLM Top-10（P2.2）**：PII 偵測+遮罩護欄接進 `/ask`+`/agent`；紅隊測試（注入 canary/系統提示外洩/越獄捏造/PII 外洩）量 block_rate；對映與誠實邊界見 `docs/security.md`。
- **測試**：72 個確定性測試（含 PII/injection/紅隊判定），跑在 CI。

## 6. 踩坑 → 根因 → 修復（真實工程紀錄）

1. **頁碼造假**：取錯 metadata key 致頁碼 `-1`、LLM 仍吐似是而非頁碼 → 改 `metadata['page_number']` + runtime 引用護欄（被引頁 ∈ 檢索命中頁）。
2. **小樣本 reranker 結論翻轉**：85→78 chunks 重建後 A/B 排序翻面 → n=12 不足以下終局裁定；不對外宣稱「顯著」。
3. **新技巧效益被遮蔽**：hybrid 檢索層顯著、端到端打平 → 拆兩層級量測才看清，效益隨語料規模上升。
4. **客觀指標反證主觀評審**：本地小模型當 faithfulness judge 分數（0.667）< 客觀正確率（0.893），抽查發現它把答對的判不忠實 → **小模型 judge 不可靠**。
5. **多輪誘發數值幻覺**：「台積電股價→那鴻海呢」第二輪憑記憶編股價 → `verify_numbers` 護欄（數字須溯源工具、刻意不採信對話歷史），逼查證/拒答；裸幻覺降為 0。
6. **容器冷啟動模型下載（140s→32s）**：Docker 化後首次 `/ask` 卡 ~140s 甚至 500 → 根因：build 時只快取 embedder，**reranker runtime 首次請求才從 HF Hub 下載**（且未認證下載被限速）→ 修：build 時多跑一次 `retrieve` 把 reranker 預載進 image。首次請求 **140s→32s**、runtime 零下載。
7. **第三方 API 被 CDN 擋（分清 provider vs CDN）**：Groq API 回 `403 error code: 1010` → **讀 403 response body** 發現 `1010` 是 **Cloudflare** 碼（非 Groq），根因是預設 `Python-urllib` User-Agent 被 WAF 當 bot 擋（與金鑰無關）→ 加 `User-Agent` 標頭即 200。教訓：錯誤碼要分清是 provider 還是它前面的 CDN/WAF。
8. **免費額度下的評測管線 429 burst**：生成 eval-gate 在迴圈連發呼叫 → 撞 free-tier RPM、雙 provider 同時 429 → crash → 修：**retry 退避 + 節流 + 單題失敗跳過**，並把生成/安全 gate 移到 nightly（不每 push 燒額度）。

## 7. 限制與下一步（誠實）

- 語料已從 toy（78 chunks）**擴充至 9,579 個真實年報 chunks**（4 家指標公司台積電/鴻海/聯發科/台泥，自 TWSE `doc.twse.com.tw` 抓取，見 `rag/fetch_corpus.py`；年報 F04 文字乾淨，台積電年報單份 22.5 萬中文字）。**檢索在 123× 規模下 context_recall 仍 1.000**（hybrid+reranker 對真實噪音 robust），新公司內容端到端答得出帶頁碼答案（台積電 2022 合併營收 758 億美元 p.5）。**誠實限制**：golden set 目前仍以 2 份原始文件為主、尚未涵蓋年報內容 → 擴 golden Qs 才是「在真實語料上做完整 eval」的下一步；測試集 n 仍偏小、CI 寬。
- 安全：**不宣稱「injection-proof」**（沒有系統是）——這是 defense-in-depth + 量測 block_rate；injection 偵測是啟發式、可被繞過；PII 為 regex+檢核碼（高精準、非 100% 召回）；攻擊集小。
- 雲端 demo 免費 tier 48h 無流量會睡、首次冷啟動數秒；VLM 讀圖走本機 Ollama、雲端不含。
- 規劃中：LangGraph + tracing（agent 可觀測性）、Grafana/Phoenix 儀表板、Terraform/Cloud Run（IaC）。

## 8. 可遷移的判斷

- **每個決策都量化**，不是「感覺」——含 bootstrap CI 與 mutation-verified 的 gate。
- **知道何時不該做什麼**：不過早上向量 DB、不放任小模型自由多步、不盲信 LLM judge、不宣稱絕對安全。
- **誠實標註不確定性**（小樣本、judge bias、免費 tier、啟發式限制）——比假裝完美更可信，也是金融場景的必要態度。

---

## 9. 履歷 / LinkedIn 可直接用的成就 bullet

- 設計並部署繁中財報 **RAG + Agent** 系統（FastAPI / Docker / Hugging Face Spaces），**隱私分層**：embedding/檢索本機、生成走雲端免費 API（Gemini 主 + Groq fallback），**$0/月、scale-to-zero**。
- 建立 **eval-as-CI-gate**：以凍結的 Faithfulness / Context-Recall **SLO 在 GitHub Actions 擋品質退步的 PR**，並以 mutation 驗證「會擋」機制。
- 實作 **OWASP LLM Top-10 紅隊測試**（prompt injection / jailbreak / system-prompt leakage / PII exfiltration）+ **台灣在地 PII 護欄**（身分證內政部檢核碼、信用卡 Luhn），**block_rate 量測而非假設**。
- 手刻 **hybrid 檢索（BM25 + dense + RRF）+ cross-encoder 重排 + 引用/數值幻覺護欄**；以 **bootstrap 95% CI** 量化每個工程決策（answer correctness 0.893、hybrid ΔMRR +0.229）。
- 生產除錯：**Cloudflare WAF 擋 API**（讀 403 body 分辨 CDN vs provider 錯誤）、**容器冷啟動模型下載 140s→32s**（build-time 預載）、**免費額度 429 burst**（retry 退避 + 節流）。
