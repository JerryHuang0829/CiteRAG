"""multi-turn golden：結構 schema ＋ 評測計分邏輯（雲端確定性，不跑 LLM）
＋ 多輪解析端到端 gate（@local，需 Ollama）。

計分邏輯（resolved/answer_ok）用合成 trace 測 → 守住「指標定義」不漂移，免跑 4B。
"""
import pytest

from eval_multiturn import load, resolved, answer_ok, _tools, _args_str

REQUIRED_TURN = {"q", "followup", "expect_tool", "expect_args", "expect_answer"}


# ---------- 雲端確定性：golden 結構 ----------
def test_multiturn_golden_loads_enough():
    convos = load()
    assert len(convos) >= 5
    n_fu = sum(1 for c in convos for t in c["turns"] if t["followup"])
    assert n_fu >= 5, f"follow-up 太少：{n_fu}"


def test_multiturn_ids_unique():
    ids = [c["id"] for c in load()]
    assert len(ids) == len(set(ids))


def test_multiturn_schema_and_first_turn():
    for c in load():
        assert c.get("turns"), f"{c['id']} 無 turns"
        assert not c["turns"][0]["followup"], f"{c['id']} 首輪不應是 follow-up"
        for t in c["turns"]:
            assert REQUIRED_TURN <= set(t), f"{c['id']} turn 缺欄位"
            assert isinstance(t["expect_args"], list) and t["expect_args"], f"{c['id']} expect_args 空"
            assert isinstance(t["expect_answer"], list) and t["expect_answer"], f"{c['id']} expect_answer 空"


# ---------- 雲端確定性：評測計分邏輯（合成 trace）----------
def _trace(tool, args):
    return [{"tool": tool, "args": args, "result": ""}]


def test_resolved_pass_when_entities_and_tool_match():
    turn = {"expect_tool": "lookup_metric", "expect_args": ["台積電", "2023"]}
    assert resolved(turn, _trace("lookup_metric", {"company": "台積電", "year": "2023", "metric": "營收"}))


def test_resolved_fail_when_entity_missing():
    turn = {"expect_tool": "lookup_metric", "expect_args": ["台積電", "2023"]}
    assert not resolved(turn, _trace("lookup_metric", {"company": "聯電", "year": "2023"}))


def test_resolved_fail_when_wrong_tool():
    turn = {"expect_tool": "compare", "expect_args": ["台積電"]}
    assert not resolved(turn, _trace("lookup_metric", {"company": "台積電"}))


def test_resolved_expect_not_args_catches_recency_error():
    turn = {"expect_tool": "lookup_metric", "expect_args": ["聯電"], "expect_not_args": ["台積電"]}
    assert resolved(turn, _trace("lookup_metric", {"company": "聯電", "metric": "營收"}))
    # 「它」誤解析成上上輪的台積電（recency 錯）→ 應判 unresolved
    assert not resolved(turn, _trace("lookup_metric", {"company": "台積電", "metric": "營收"}))


def test_resolved_fail_when_no_tool_called():
    # 直接 final、沒呼叫工具（如 mt6 幻覺案例）→ unresolved
    turn = {"expect_tool": "stock_price", "expect_args": ["鴻海"]}
    assert not resolved(turn, [])


def test_answer_ok_any_match():
    turn = {"expect_answer": ["2.162", "2.16 兆"]}
    assert answer_ok(turn, "台積電 2023 年營收為 2.162 兆元")
    assert not answer_ok(turn, "參考資料中查無此資訊")


# ---------- @local gate：多輪解析端到端（需 Ollama + FinMind 快取）----------
@pytest.mark.local
def test_multiturn_resolution_end_to_end():
    # 先問鴻海 2022 EPS，再以省略句「那 2021 年呢」接續；ON 帶 history 應把鴻海+2021 帶進 args。
    import agent
    q1 = "鴻海 2022 全年 EPS 多少？"
    f1, _ = agent.run(q1, [], verbose=False)
    history = [{"role": "user", "content": q1}, {"role": "assistant", "content": f1}]
    _, trace = agent.run("那 2021 年呢？", history, verbose=False)
    turn = {"expect_tool": "lookup_metric", "expect_args": ["鴻海", "2021"], "expect_answer": ["10.05"]}
    assert resolved(turn, trace), f"多輪解析失敗：tools={_tools(trace)} args={_args_str(trace)}"


@pytest.mark.local
def test_numeric_guardrail_no_silent_fabrication():
    # #6：股價多輪 follow-up 不可『沒查就編數字』。verify_numbers 護欄保證──
    # 要嘛逼出 stock_price 拿到真值（257.x），要嘛誠實拒答；絕不裸顯示臆測數字。
    import agent
    q1 = "台積電現在股價多少？"
    f1, _ = agent.run(q1, [], verbose=False)
    history = [{"role": "user", "content": q1}, {"role": "assistant", "content": f1}]
    f2, t2 = agent.run("那鴻海呢？", history, verbose=False)
    tools = _tools(t2)
    got_verified = ("stock_price" in tools) and ("257" in f2)
    refused = ("無法" in f2) or ("查證" in f2)
    assert got_verified or refused, f"既未查證也未拒答（疑似裸幻覺）：tools={tools} ans={f2}"
