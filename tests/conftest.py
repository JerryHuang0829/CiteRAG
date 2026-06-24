"""pytest 共用設定：把 rag/ 加到 sys.path，並註冊 local marker（不依賴 pytest.ini 是否被讀到）。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rag"))


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "local: 需要 Ollama + 已建索引；本地跑（pytest -m local），雲端 CI 跳過")
