"""llm_router 路由與雲端 fallback 的確定性測試（monkeypatch _post，不需網路/Ollama/金鑰）。

驗 backend 分流、Gemini→Groq fallback、as_json 壞 JSON 觸發 fallback、逐家 payload 形狀、兩家皆掛拋 CloudLLMError。
"""
import json
import urllib.error

import llm_router

_GEMINI_OK = {"candidates": [{"content": {"parts": [{"text": "嗨G"}]}}]}
_GROQ_OK = {"choices": [{"message": {"content": "嗨Q"}}]}


def _fake_post(gemini, groq):
    cap = []

    def post(url, body, headers):
        cap.append((url, body, headers))
        target = gemini if "googleapis" in url else groq
        if isinstance(target, Exception):
            raise target
        return target

    return post, cap


def _http429():
    return urllib.error.HTTPError("u", 429, "rate limit", {}, None)


def _set_cloud(monkeypatch):
    monkeypatch.setenv("CITERAG_LLM_BACKEND", "cloud")
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    monkeypatch.setenv("GROQ_API_KEY", "q")


def test_backend_default_is_ollama(monkeypatch):
    monkeypatch.delenv("CITERAG_LLM_BACKEND", raising=False)
    assert llm_router._backend() == "ollama"


def test_ollama_dispatch_calls_core_not_cloud(monkeypatch):
    monkeypatch.setenv("CITERAG_LLM_BACKEND", "ollama")
    import core
    monkeypatch.setattr(core, "_ollama_generate", lambda p, j=False: f"OLL:{p}:{j}")
    assert llm_router.route_generate("hi", as_json=True) == "OLL:hi:True"


def test_cloud_gemini_primary_no_fallback(monkeypatch):
    _set_cloud(monkeypatch)
    post, cap = _fake_post(_GEMINI_OK, _GROQ_OK)
    monkeypatch.setattr(llm_router, "_post", post)
    assert llm_router.route_chat([{"role": "user", "content": "hi"}]) == "嗨G"
    assert len(cap) == 1 and "googleapis" in cap[0][0]   # 只打 gemini


def test_cloud_falls_back_to_groq_on_gemini_error(monkeypatch):
    _set_cloud(monkeypatch)
    post, cap = _fake_post(_http429(), _GROQ_OK)
    monkeypatch.setattr(llm_router, "_post", post)
    assert llm_router.route_chat([{"role": "user", "content": "hi"}]) == "嗨Q"
    assert len(cap) == 2   # gemini 失敗→groq


def test_cloud_both_fail_raises(monkeypatch):
    _set_cloud(monkeypatch)
    post, _ = _fake_post(_http429(), _http429())
    monkeypatch.setattr(llm_router, "_post", post)
    try:
        llm_router.route_generate("hi")
        assert False, "應拋 CloudLLMError"
    except llm_router.CloudLLMError:
        pass


def test_as_json_invalid_gemini_triggers_fallback(monkeypatch):
    _set_cloud(monkeypatch)
    bad_gemini = {"candidates": [{"content": {"parts": [{"text": "這不是 json"}]}}]}
    good_groq = {"choices": [{"message": {"content": '{"ok":1}'}}]}
    post, cap = _fake_post(bad_gemini, good_groq)
    monkeypatch.setattr(llm_router, "_post", post)
    out = llm_router.route_chat([{"role": "user", "content": "hi"}], as_json=True)
    assert json.loads(out) == {"ok": 1}
    assert len(cap) == 2   # gemini 回非 JSON → fallback groq


def test_gemini_payload_role_and_json_shape(monkeypatch):
    _set_cloud(monkeypatch)
    good_gemini = {"candidates": [{"content": {"parts": [{"text": '{"a":1}'}]}}]}
    post, cap = _fake_post(good_gemini, _GROQ_OK)
    monkeypatch.setattr(llm_router, "_post", post)
    llm_router.route_chat([
        {"role": "system", "content": "S"},
        {"role": "user", "content": "U"},
        {"role": "assistant", "content": "A"},
    ], as_json=True)
    _url, body, _headers = cap[0]
    assert body["generationConfig"]["responseMimeType"] == "application/json"
    assert body["systemInstruction"]["parts"][0]["text"] == "S"          # system→systemInstruction
    assert [c["role"] for c in body["contents"]] == ["user", "model"]    # user→user, assistant→model
    parts = [p["text"] for c in body["contents"] for p in c["parts"]]
    assert "S" not in parts                                              # system 不混進 contents


def test_cloud_vlm_gemini_inline_image(monkeypatch):
    import base64
    _set_cloud(monkeypatch)
    gem = {"candidates": [{"content": {"parts": [{"text": "圖中文字"}]}}]}
    post, cap = _fake_post(gem, None)
    monkeypatch.setattr(llm_router, "_post", post)
    b64 = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"x" * 20).decode()   # PNG magic
    assert llm_router.route_vlm(b64, "讀圖") == "圖中文字"
    _url, body, _h = cap[0]
    part = body["contents"][0]["parts"][1]["inline_data"]
    assert part["mime_type"] == "image/png" and part["data"] == b64        # magic 判 mime + 圖傳對


def test_groq_payload_json_and_auth(monkeypatch):
    _set_cloud(monkeypatch)
    post, cap = _fake_post(_http429(), {"choices": [{"message": {"content": '{"a":1}'}}]})
    monkeypatch.setattr(llm_router, "_post", post)
    llm_router.route_chat([{"role": "user", "content": "U"}], as_json=True)
    _url, body, headers = cap[1]   # groq 是 fallback、第二打
    assert body["response_format"] == {"type": "json_object"}
    assert headers["Authorization"].startswith("Bearer ")
