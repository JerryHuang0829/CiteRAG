"""問答：帶頁碼引用的 RAG 回答。
用法（在 rag/ 目錄）：
  python ask.py "2025 年的 EPS 是多少？"
  python ask.py            # 不帶參數 → 互動模式（空白離開）
"""
import sys

import core


def _safe_answer(q):
    # 把 Ollama 未啟動 / 索引未建等可預期錯誤印成一行人話，而非整頁 traceback
    try:
        core.answer(q)
    except (core.OllamaError, FileNotFoundError) as e:
        print(f"\n[錯誤] {e}")


def main():
    if len(sys.argv) > 1:
        _safe_answer(" ".join(sys.argv[1:]))
        return
    print("輸入問題（空白行離開）：")
    while True:
        try:
            q = input("\n> ").strip()
        except EOFError:
            break
        if not q:
            break
        _safe_answer(q)


if __name__ == "__main__":
    main()
