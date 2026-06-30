"""生成端 LLM router：本機開發走 Ollama（預設），hosted demo 走免費雲端 API。

CITERAG_LLM_BACKEND = ollama(預設) | cloud
  cloud：Gemini 2.5 Flash-Lite 為主 → 失敗/429/壞 JSON 時 fallback Groq → 兩家皆掛則 raise。
只有「生成」這一步可外送；embedding 與檢索一律本機 bge-small-zh（資料不外洩）。
金鑰讀環境變數或 repo 根目錄 .env：GEMINI_API_KEY / GROQ_API_KEY。純 stdlib，零新依賴。

注意：Gemini 免費層會用 prompt 訓練 → 僅送公開資料（FinMind 衍生），勿送敏感/PII。
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path

_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"   # RAG_Agent_Copilot/.env
GEMINI_MODEL = os.environ.get("CITERAG_GEMINI_MODEL", "gemini-2.5-flash-lite")
# 預設用 probe 已驗證可達的 Groq 模型；繁中品質更佳可改設 CITERAG_GROQ_MODEL（見 README，須自行確認 console 模型 id）
GROQ_MODEL = os.environ.get("CITERAG_GROQ_MODEL", "llama-3.1-8b-instant")
_TIMEOUT = int(os.environ.get("CITERAG_CLOUD_TIMEOUT", "60"))


class CloudLLMError(RuntimeError):
    """雲端兩家 provider 皆失敗時拋出，與本機 OllamaError 區隔。"""


def _backend() -> str:
    # 每次呼叫時讀，方便測試與執行期切換
    return os.environ.get("CITERAG_LLM_BACKEND", "ollama").strip().lower()


def _getenv(key: str):
    # 先環境變數，後 repo 根目錄 .env（與 w0/03_cloud_probe.py 一致）
    v = os.environ.get(key)
    if v:
        return v
    try:
        for line in _ENV_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith(key + "="):
                return line.split("=", 1)[1].strip()
    except FileNotFoundError:
        pass
    return None


def _post(url: str, body: dict, headers: dict) -> dict:
    # 單一網路出口（測試只 monkeypatch 這裡即可離線）
    # User-Agent：Groq API 在 Cloudflare 後，會把預設 "Python-urllib" UA 當 bot 擋（HTTP 403 error 1010）；偽裝瀏覽器繞過。
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0", **headers},
    )
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
        return json.loads(r.read())


def _gemini_chat(messages: list[dict], as_json: bool) -> str:
    key = _getenv("GEMINI_API_KEY")
    if not key:
        raise CloudLLMError("缺 GEMINI_API_KEY")
    # role 對映：system→systemInstruction、assistant→model、其餘→user
    sys_txt = "\n".join(m["content"] for m in messages if m["role"] == "system")
    contents = [
        {"role": "model" if m["role"] == "assistant" else "user",
         "parts": [{"text": m["content"]}]}
        for m in messages if m["role"] != "system"
    ]
    gen_cfg: dict = {"temperature": 0.0 if as_json else 0.2}
    if as_json:
        gen_cfg["responseMimeType"] = "application/json"
    body: dict = {"contents": contents, "generationConfig": gen_cfg}
    if sys_txt:
        body["systemInstruction"] = {"parts": [{"text": sys_txt}]}
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{GEMINI_MODEL}:generateContent?key={key}")
    data = _post(url, body, {})
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError) as e:
        # 安全攔截/封鎖時無 candidates → 當失敗，交給 fallback
        raise CloudLLMError(f"Gemini 回應格式異常：{str(data)[:200]}") from e


def _groq_chat(messages: list[dict], as_json: bool) -> str:
    key = _getenv("GROQ_API_KEY")
    if not key:
        raise CloudLLMError("缺 GROQ_API_KEY")
    body: dict = {"model": GROQ_MODEL, "messages": messages,
                  "temperature": 0.0 if as_json else 0.2}
    if as_json:
        body["response_format"] = {"type": "json_object"}
    data = _post("https://api.groq.com/openai/v1/chat/completions", body,
                 {"Authorization": f"Bearer {key}"})
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise CloudLLMError(f"Groq 回應格式異常：{str(data)[:200]}") from e


def _cloud_chat(messages: list[dict], as_json: bool) -> str:
    errs = []
    for name, fn in (("gemini", _gemini_chat), ("groq", _groq_chat)):
        try:
            out = fn(messages, as_json)
            if as_json:
                json.loads(out)   # 守「壞 JSON 歸零」：非合法 JSON 視為失敗→fallback
            return out
        except (urllib.error.URLError, OSError, CloudLLMError, ValueError) as e:
            errs.append(f"{name}: {type(e).__name__}: {str(e)[:150]}")
    raise CloudLLMError("雲端生成兩家皆失敗 → " + " | ".join(errs))


def route_generate(prompt: str, as_json: bool = False) -> str:
    if _backend() == "cloud":
        return _cloud_chat([{"role": "user", "content": prompt}], as_json)
    import core   # lazy：避免與 core 的循環 import；ollama 路徑才需要
    return core._ollama_generate(prompt, as_json)


def route_chat(messages: list[dict], as_json: bool = False) -> str:
    if _backend() == "cloud":
        return _cloud_chat(messages, as_json)
    import core
    return core._ollama_chat(messages, as_json)
