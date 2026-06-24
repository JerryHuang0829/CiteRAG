"""產生一張「設備銘牌」測試圖（PNG）給 VLM(Gemma) 讀圖示範用。
用法（rag/）：python make_sample_image.py → ../data/sample_nameplate.png
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent.parent / "data" / "sample_nameplate.png"
FONT_PATH = "C:/Windows/Fonts/msjh.ttc"   # 微軟正黑，含中英


def _font(sz):
    try:
        return ImageFont.truetype(FONT_PATH, sz)
    except Exception:
        return ImageFont.load_default()


def main():
    img = Image.new("RGB", (660, 380), (28, 38, 52))
    d = ImageDraw.Draw(img)
    d.rectangle([12, 12, 648, 368], outline=(200, 200, 200), width=3)
    lines = [
        ("設備銘牌 EQUIPMENT NAMEPLATE", 26),
        ("型號 MODEL : PUMP-X200", 24),
        ("序號 SERIAL: 2024-0913-TW", 24),
        ("額定壓力 PRESSURE: 4.2 bar", 24),
        ("運轉溫度 TEMP: 78 C", 24),
        ("製造商 MFR: 範例工業 Example Industrial", 20),
    ]
    y = 44
    for t, sz in lines:
        d.text((34, y), t, fill=(238, 238, 238), font=_font(sz))
        y += 50
    OUT.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUT)
    print("已產生", OUT)


if __name__ == "__main__":
    main()
