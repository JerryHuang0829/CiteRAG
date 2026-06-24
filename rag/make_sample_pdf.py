"""產生一份繁中金融範例 PDF 當 W1 smoke-test fixture（非真實語料）。
真實語料：把你的 PDF 丟到 ../data/ 取代即可（pipeline 與語料無關）。
用法（在 rag/ 目錄）：python make_sample_pdf.py
"""
from pathlib import Path

import fitz   # PyMuPDF；fontname="china-t" 為內建繁中字型

OUT = Path(__file__).resolve().parent.parent / "data" / "sample_finance.pdf"

PAGES = [
    ["範例公司 2025 年度財務摘要",
     "",
     "本文件為 RAG 流程測試用的範例財報摘要，內容為虛構，非真實資料。",
     "範例公司為一家虛構的製造業上市公司，主要產品為工業泵浦。"],
    ["第二節 經營績效",
     "",
     "2025 年第四季合併營收為新台幣 120 億元，較去年同期成長 15%。",
     "2025 全年每股盈餘（EPS）為新台幣 8.5 元。",
     "毛利率為 38%，營業利益率為 22%。"],
    ["第三節 股利與重要日期",
     "",
     "董事會決議每股配發現金股利新台幣 4 元。",
     "除息交易日為 2026 年 7 月 15 日。",
     "現金股利發放日為 2026 年 8 月 12 日。"],
]


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    for lines in PAGES:
        page = doc.new_page()
        y = 72.0
        for i, line in enumerate(lines):
            size = 16 if i == 0 else 12
            if line:
                page.insert_text((72, y), line, fontname="china-t", fontsize=size)
            y += size + 10
    doc.save(str(OUT))
    print(f"已產生 {OUT}（{len(PAGES)} 頁）")


if __name__ == "__main__":
    main()
