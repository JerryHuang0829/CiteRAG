"""agent 控制流 × verify_numbers 數值護欄的確定性測試（monkeypatch core.chat，不需 Ollama）。

把 4B 換成腳本化回應，鎖死狀態機：未溯源數字→退回逼工具→（查到放行 / 全程沒查則拒答）。
雲端確定性測試（@local 的端到端版另測 4B 真實行為）。
"""
import json

import agent
import core
import findata


def _scripted(responses):
    it = iter(responses)
    def fake_chat(messages, as_json=False):
        return next(it)
    return fake_chat


def test_refuses_when_persistent_ungrounded_and_no_tool(monkeypatch):
    # 兩次都直接吐含未溯源數字的 final、全程沒查工具 → 更正一次後仍未溯源 → 誠實拒答、不展示假數字
    monkeypatch.setattr(core, "chat", _scripted([
        json.dumps({"action": "final", "answer": "鴻海股價為 112.50 元"}),
        json.dumps({"action": "final", "answer": "鴻海股價為 112.50 元"}),
    ]))
    final, trace = agent.run("那鴻海呢？", [], verbose=False)
    assert trace == []                                  # 全程沒呼叫工具
    assert "112.5" not in final                         # 幻覺數字不裸顯示
    assert "無法" in final or "查證" in final            # 改誠實拒答


def test_accepts_final_when_numbers_grounded_in_question(monkeypatch):
    # 數字已在題目（grounded）→ 直接放行，不誤觸更正
    monkeypatch.setattr(core, "chat", _scripted([
        json.dumps({"action": "final", "answer": "台積電 EPS 為 32.34 元"}),
    ]))
    final, trace = agent.run("台積電 2023 EPS 是 32.34 元嗎？", [], verbose=False)
    assert "32.34" in final


def test_corrective_retry_induces_tool_then_grounded(monkeypatch):
    # 未溯源 final → 退回 → 改查工具 → 工具結果讓數字溯源 → 放行真值、丟棄臆測
    monkeypatch.setattr(findata, "price", lambda company, year=None: "鴻海(2317) 收盤 257.5 元")
    monkeypatch.setattr(core, "chat", _scripted([
        json.dumps({"action": "final", "answer": "鴻海股價約 999.9 元"}),
        json.dumps({"action": "tool", "tool": "stock_price", "args": {"company": "鴻海"}}),
        json.dumps({"action": "final", "answer": "鴻海股價為 257.5 元"}),
    ]))
    final, trace = agent.run("那鴻海呢？", [], verbose=False)
    assert any(t["tool"] == "stock_price" for t in trace)
    assert "257.5" in final and "999" not in final


def test_grounded_does_not_include_history_assistant(monkeypatch):
    # 上一輪模型講過的數字（在 history）不可當本輪數值佐證 → 仍須查工具，否則拒答（跨輪自我污染防線）
    monkeypatch.setattr(core, "chat", _scripted([
        json.dumps({"action": "final", "answer": "它的股價是 112.50 元"}),
        json.dumps({"action": "final", "answer": "它的股價是 112.50 元"}),
    ]))
    history = [{"role": "user", "content": "鴻海股價？"},
               {"role": "assistant", "content": "鴻海股價為 112.50 元"}]   # 含 112.50
    final, trace = agent.run("再說一次？", history, verbose=False)
    assert "112.5" not in final                         # history 的數字不算溯源 → 不被洗白
