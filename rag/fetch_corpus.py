"""P3-B：抓真實台股年報（TWSE `doc.twse.com.tw` t57sb01 三步流程）到 data/，建真實 RAG 語料。

為何年報：實測年報 F04 文字乾淨（非掃描，台積電年報 22.5 萬中文字/份）；法說會簡報幾乎全圖（12 頁僅 757 字）故不採。
純 urllib GET，無 session/JS/cookie；末端 `/pdf/` URL 帶時戳故 step9→download 需連續。低量無限流，禮貌 sleep。
用法（rag/）：python fetch_corpus.py → 下載 CORPUS 清單到 ../data/，再跑 `python ingest.py` 重建索引。
"""
import re
import time
import urllib.request
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
BASE = "https://doc.twse.com.tw"
_H = {"User-Agent": "Mozilla/5.0"}

# 策展清單：(公司名, 股票代號, 民國年)。年報＝F04。
CORPUS = [
    ("台積電", "2330", "112"),
    ("鴻海", "2317", "112"),
    ("聯發科", "2454", "112"),
    ("台泥", "1101", "112"),
]


def _get(url: str) -> bytes:
    return urllib.request.urlopen(urllib.request.Request(url, headers=_H), timeout=60).read()


def fetch_annual(co_id: str, year_roc: str):
    """回 (pdf_bytes, filename) 或 None。三步：list(step1)→resolve(step9)→download(/pdf/)。"""
    lst = _get(f"{BASE}/server-java/t57sb01?step=1&colorchg=1&co_id={co_id}&year={year_roc}&mtype=F").decode("big5", "ignore")
    m = re.findall(rf'readfile2\("F","{co_id}","([0-9_]+F04\.pdf)"\)', lst)
    if not m:
        return None
    fn = m[0]
    pg = _get(f"{BASE}/server-java/t57sb01?step=9&kind=F&co_id={co_id}&filename={fn}").decode("utf-8", "ignore")
    href = re.search(r"/pdf/[\w]+\.pdf", pg)
    if not href:
        return None
    return _get(BASE + href.group(0)), fn


def main():
    DATA.mkdir(exist_ok=True)
    for name, co, yr in CORPUS:
        try:
            res = fetch_annual(co, yr)
            if not res:
                print(f"[skip] {name} {co} 民{yr}：找不到 F04")
                continue
            data, _fn = res
            out = DATA / f"annual_{co}_{yr}_{name}.pdf"
            out.write_bytes(data)
            print(f"[ok] {name} {co} 民{yr} 年報 {len(data) // 1024}KB -> {out.name}")
        except Exception as e:
            print(f"[err] {name} {co}: {repr(e)[:120]}")
        time.sleep(1.5)


if __name__ == "__main__":
    main()
