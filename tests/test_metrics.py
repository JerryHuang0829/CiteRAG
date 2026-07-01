"""/metrics（Prometheus）系統觀測性 endpoint 測試（TestClient，不需 Ollama/index/雲端）。"""
from fastapi.testclient import TestClient

import api


def test_metrics_endpoint_prometheus_format():
    c = TestClient(api.app)
    c.get("/health")                                # 觸發一次請求以產生指標
    r = c.get("/metrics")
    assert r.status_code == 200
    body = r.text
    assert "# HELP" in body and "# TYPE" in body     # Prometheus 曝露格式
    assert "http_request" in body                    # 有 HTTP 請求指標（數/延遲/狀態）
