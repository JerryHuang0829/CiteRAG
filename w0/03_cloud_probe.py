"""W0-03 免費雲端臂連通測試：確認台灣 IP 能否直連 Gemini / Groq 免費 API。

router 的雲端逃生艙若連不到，就要改純本地策略。
金鑰（皆有免費額度，選填）：環境變數或同層 .env 設 GEMINI_API_KEY / GROQ_API_KEY。
無金鑰時只測網路可達性。零額外套件。
"""
import json
import os
import time
import urllib.error
import urllib.request


def getenv_file(key):
    v = os.environ.get(key)
    if v:
        return v
    try:
        for line in open(".env", encoding="utf-8"):
            if line.strip().startswith(key + "="):
                return line.split("=", 1)[1].strip()
    except FileNotFoundError:
        pass
    return None


def reachable(url):
    try:
        with urllib.request.urlopen(urllib.request.Request(url), timeout=10) as r:
            return f"reachable (HTTP {r.status})"
    except urllib.error.HTTPError as e:
        return f"reachable (HTTP {e.code})"   # 4xx 也代表連得到
    except Exception as e:
        return f"FAIL: {e}"


def test_gemini(key):
    url = ("https://generativelanguage.googleapis.com/v1beta/models/"
           f"gemini-2.5-flash:generateContent?key={key}")
    body = json.dumps({"contents": [{"parts": [{"text": "say OK"}]}]}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=30) as r:
        r.read()
    return f"auth OK, {round(time.time() - t0, 2)}s"


def test_groq(key):
    url = "https://api.groq.com/openai/v1/chat/completions"
    body = json.dumps({"model": "llama-3.1-8b-instant",
                       "messages": [{"role": "user", "content": "say OK"}],
                       "max_tokens": 5}).encode()
    req = urllib.request.Request(url, data=body, headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=30) as r:
        r.read()
    return f"auth OK, {round(time.time() - t0, 2)}s"


def main():
    print("== 網路可達性（無金鑰也測）==")
    print("Gemini host:", reachable("https://generativelanguage.googleapis.com/"))
    print("Groq host  :", reachable("https://api.groq.com/"))

    print("\n== 金鑰驗證（有設才測）==")
    gk, qk = getenv_file("GEMINI_API_KEY"), getenv_file("GROQ_API_KEY")
    if gk:
        try:
            print("Gemini:", test_gemini(gk))
        except Exception as e:
            print("Gemini FAIL:", e)
    else:
        print("Gemini: 無 GEMINI_API_KEY，跳過（免費申請 aistudio.google.com）")
    if qk:
        try:
            print("Groq:", test_groq(qk))
        except Exception as e:
            print("Groq FAIL:", e)
    else:
        print("Groq: 無 GROQ_API_KEY，跳過（免費申請 console.groq.com）")


if __name__ == "__main__":
    main()
