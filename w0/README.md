# W0 Benchmark Gate — 實測手冊

對應 [PLAN.md §7](../PLAN.md)。**鐵律：這裡量出來的數字才能寫進作品集文件，估計值不算。**
目的：用真實數字決定 agent demo 野心大小（非決定 go/no-go——專案本身是 GO）。

## 前置
1. 裝好 Ollama（原生 Windows），確認 `OLLAMA_MODELS` 指向 E:（見 PLAN §12）。
2. 拉模型：
   ```
   ollama pull qwen3:4b-instruct
   ollama pull gemma3:4b        # 04/VLM 用，可稍後
   ```
   用 `ollama list` 確認 tag；若名稱不同，改各腳本頂部的 `MODEL`。
3. 01/02/03 **只需 Python（stdlib）+ Ollama**，零額外套件。04（可選、較重）才裝 `requirements_w0.txt`。
4. 在 conda env `rag` 內跑（`conda activate rag` 或 `--prefix`）。

## 執行順序與判讀

| 步驟 | 指令 | 量什麼 | 判讀 / gate |
|---|---|---|---|
| 00 硬體 | `powershell -ExecutionPolicy Bypass -File 00_hardware.ps1` | **RAM 單/雙通道**、CPU、GPU、磁碟 | 單通道 + 插槽有空 → 補一條 RAM 是最高 ROI（頻寬 +30-50%） |
| 01 LLM 速度 | `python 01_llm_bench.py` | prefill / decode tok/s × num_thread | 對照估計 decode 6-10、prefill ~40；挑最快 thread 寫進設定 |
| 02 **Agent 成功率** | `python 02_agent_success.py` | 單步 tool-call 成功率 + bootstrap CI | **≥80% 主秀／70-80% 勉強(難步驟外送)／<70% 降級** |
| 03 雲端連通 | `python 03_cloud_probe.py` | 台灣 IP 能否連 Gemini/Groq | 連不到 → router 改純本地策略 |
| 04 embedding（可選） | `python 04_embed_recall.py` | bge-small vs bge-m3 繁中 recall + 轉簡 A/B | 轉簡 recall 明顯較高 = 腳本錯配嚴重 → 離線臂用 bge-m3 |

## 跑完請貼回給我
- `w0_results_llm.json`、`w0_results_agent.json`（最關鍵）、03 主控台輸出、(04) `w0_results_embed.json`。
- 以及 00 的「已插記憶體條數 / 插槽總數 / RAM 速度」。

我會據這些數字定：agent demo 規模、主力是否從 4B 換 8B、是否值得補 RAM、router 預設走本地或雲端。

> 誠實補充：所有腳本的 toy 預設（02 的工具題、04 的 toy 語料）是為了「跑得起來」；02 的題目就是你旗艦情境的工具，數字有意義；04 的 toy 語料數字無意義，要換成你自己的 `w0_corpus.jsonl` / `w0_queries.jsonl` 才算數。
