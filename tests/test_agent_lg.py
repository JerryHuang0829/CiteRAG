"""agent_lg（LangGraph 版）與手刻 agent 的行為等價測試——同一組 scripted 回應，斷言同樣護欄行為。

monkeypatch core.chat（不需 Ollama/雲端）；langgraph 未裝則整檔 skip（不擋預設 CI）。
"""
import json

import pytest

pytest.importorskip("langgraph")   # 選配依賴未裝 → 跳過

import agent_lg
import core
import findata


def _scripted(responses):
    it = iter(responses)
    return lambda messages, as_json=False: next(it)


def test_refuses_when_persistent_ungrounded_and_no_tool(monkeypatch):
    monkeypatch.setattr(core, "chat", _scripted([
        json.dumps({"action": "final", "answer": "鴻海股價為 112.50 元"}),
        json.dumps({"action": "final", "answer": "鴻海股價為 112.50 元"}),
    ]))
    final, trace = agent_lg.run("那鴻海呢？", [])
    assert trace == []
    assert "112.5" not in final
    assert "無法" in final or "查證" in final


def test_accepts_final_when_numbers_grounded_in_question(monkeypatch):
    monkeypatch.setattr(core, "chat", _scripted([
        json.dumps({"action": "final", "answer": "台積電 EPS 為 32.34 元"}),
    ]))
    final, _ = agent_lg.run("台積電 2023 EPS 是 32.34 元嗎？", [])
    assert "32.34" in final


def test_corrective_retry_induces_tool_then_grounded(monkeypatch):
    monkeypatch.setattr(findata, "price", lambda company, year=None: "鴻海(2317) 收盤 257.5 元")
    monkeypatch.setattr(core, "chat", _scripted([
        json.dumps({"action": "final", "answer": "鴻海股價約 999.9 元"}),
        json.dumps({"action": "tool", "tool": "stock_price", "args": {"company": "鴻海"}}),
        json.dumps({"action": "final", "answer": "鴻海股價為 257.5 元"}),
    ]))
    final, trace = agent_lg.run("那鴻海呢？", [])
    assert any(t["tool"] == "stock_price" for t in trace)
    assert "257.5" in final and "999" not in final


def test_grounded_does_not_include_history_assistant(monkeypatch):
    monkeypatch.setattr(core, "chat", _scripted([
        json.dumps({"action": "final", "answer": "它的股價是 112.50 元"}),
        json.dumps({"action": "final", "answer": "它的股價是 112.50 元"}),
    ]))
    history = [{"role": "user", "content": "鴻海股價？"},
               {"role": "assistant", "content": "鴻海股價為 112.50 元"}]
    final, _ = agent_lg.run("再說一次？", history)
    assert "112.5" not in final
