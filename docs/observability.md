# 可觀測性（Observability）— 兩層

生產系統要能被看見：出錯/變慢/變貴都要能發現。本專案分兩層，互補：

| 層 | 工具 | 看什麼 | 在哪 |
|---|---|---|---|
| **LLM 語意層** | **Langfuse**（`@observe` tracing） | 每次 agent 的軌跡：retrieve→工具→LLM 決策→答案，每步的輸入/輸出/延遲 | `rag/agent_lg.py`（env-gated；有 `LANGFUSE_*` 才開） |
| **系統層** | **Prometheus `/metrics`**（`prometheus-fastapi-instrumentator`）→ Grafana | 請求數 / p50-p95-p99 延遲 / 狀態碼 / in-progress / 每 endpoint | `rag/api.py`（`/metrics`，已接） |

## 系統層 `/metrics`（已實作）
FastAPI app 暴露 `/metrics`（Prometheus 曝露格式），含 `http_requests_total`、`http_request_duration_seconds`（histogram，可算 p95）、`http_requests_inprogress` 等，依 handler/method/status 標籤。跑起來即可 `curl localhost:8000/metrics`。測試見 `tests/test_metrics.py`。

## 接 Grafana（把 /metrics 變儀表板）
兩條路，二選一：

**A. 本機 Prometheus + Grafana（零帳號、docker）**
```yaml
# prometheus.yml
scrape_configs:
  - job_name: citerag
    metrics_path: /metrics
    static_configs:
      - targets: ["host.docker.internal:8000"]
```
```bash
docker run -d -p 9090:9090 -v $PWD/prometheus.yml:/etc/prometheus/prometheus.yml prom/prometheus
docker run -d -p 3000:3000 grafana/grafana     # 加 Prometheus data source → 建 dashboard
```

**B. Grafana Cloud（免費、可分享連結）**
- 註冊 grafana.com（免費 tier）→ 用 **Grafana Alloy / Prometheus remote_write** 把 `/metrics` 推上去，或用 Grafana Cloud 的 scrape。
- 對 HF Spaces 上的 demo：Space 需 public（已是）；用 remote_write agent 從能碰到 /metrics 的地方推。

**推薦儀表板面板**：request rate、p95 latency、error rate（5xx/總數）、/ask vs /agent 延遲對比。

## 誠實邊界
- `/metrics` endpoint 已實作並測試；**Grafana 儀表板是「接上去」的一步**（需 Prometheus/Grafana Cloud 設定），本 repo 提供 endpoint + scrape 設定範例，未內含跑起來的 Grafana。
- 系統層看「快不快、錯不錯」；**內容品質看 eval-gate、LLM 行為看 Langfuse**——三者合起來才是完整觀測。
