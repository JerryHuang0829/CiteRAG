"""W0-01 LLM 推論實測：量 Qwen3-4B-instruct 在本機的 prefill / decode tok/s。

prefill(吃 context)是 agent 延遲主因；這個數決定 demo 是 2 分鐘還是 10 分鐘。
前置：Ollama 已跑、已 `ollama pull qwen3:4b-instruct`。零額外套件。
輸出：主控台表格 + w0_results_llm.json
"""
import json
import sys
import time
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")   # Windows cp950 無法編 ≥ 等字元，強制 UTF-8 輸出

OLLAMA = "http://localhost:11434/api/generate"
MODEL = "qwen3:4b-instruct"          # tag 不同請改（ollama list 查）
THREADS = [4, 6, 8]                  # i7-1260P 4 P-core；比哪個最快
CTX_TARGETS = [2000, 3000, 4000]     # 目標 prompt token 數（中文約 1 字 1 token）
NUM_PREDICT = 128                    # decode 取樣長度

FILLER = ("設備維修手冊：泵浦在運轉前需確認潤滑油位、軸封無洩漏、聯軸器對心。"
          "定期保養包含更換濾芯、檢查壓力錶讀數、記錄振動值與軸承溫度。")


def make_prompt(approx_tokens):
    body = (FILLER * (approx_tokens // len(FILLER) + 1))[:approx_tokens]
    return body + "\n\n問題：換濾芯的步驟是什麼？請簡短回答。"


def bench(prompt, num_thread):
    payload = {
        "model": MODEL, "prompt": prompt, "stream": False,
        "options": {"num_ctx": 4096, "num_thread": num_thread,
                    "num_predict": NUM_PREDICT, "temperature": 0},
        "keep_alive": "30m",
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(OLLAMA, data=data, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        res = json.loads(r.read())
    wall = time.time() - t0
    pe_n, pe_d = res.get("prompt_eval_count", 0), res.get("prompt_eval_duration", 0)
    ev_n, ev_d = res.get("eval_count", 0), res.get("eval_duration", 0)
    return {
        "prompt_tokens": pe_n,
        "prefill_tok_s": round(pe_n / (pe_d / 1e9), 1) if pe_d else None,
        "decode_tok_s": round(ev_n / (ev_d / 1e9), 1) if ev_d else None,
        "wall_s": round(wall, 1),
    }


def main():
    print(f"Model={MODEL}  (首次冷載稍等)\n")
    print(f"{'ctx~':>6} {'thread':>6} {'prefill_t/s':>12} {'decode_t/s':>11} {'wall_s':>7}")
    rows = []
    for ctx in CTX_TARGETS:
        p = make_prompt(ctx)
        for th in THREADS:
            try:
                r = bench(p, th)
            except Exception as e:
                print(f"{ctx:>6} {th:>6}  ERROR: {e}")
                continue
            r.update({"ctx_target": ctx, "num_thread": th})
            rows.append(r)
            print(f"{ctx:>6} {th:>6} {str(r['prefill_tok_s']):>12} "
                  f"{str(r['decode_tok_s']):>11} {r['wall_s']:>7}")
    with open("w0_results_llm.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print("\n→ 已存 w0_results_llm.json。最快的 thread 寫進 Ollama 設定。")
    print("→ 對照估計 decode 6-10 / prefill ~40；低很多代表單通道或降頻，先處理 RAM。")


if __name__ == "__main__":
    main()
