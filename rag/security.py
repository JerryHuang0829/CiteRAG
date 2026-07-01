"""安全防護原語（P2.2，對映 OWASP LLM Top-10）：PII 偵測/遮罩 + prompt-injection 啟發式偵測。

台灣在地 PII（身分證含內政部檢核碼、手機、信用卡含 Luhn、Email）——比 Presidio 的 US-centric
recognizer 更準且零重依賴。用作輸出護欄（與 verify_citations/verify_numbers 對稱）+ 紅隊測試判定。
誠實邊界：injection 偵測是啟發式（defense layer 1，非萬能）；PII 為 regex+檢核碼（降誤報，非 100% 召回）。
"""
import re

# --- 台灣身分證：1 英文字母 + (1|2) + 8 數字，且過內政部檢核碼 ---
# 字母對應碼（非字母順序，內政部規定）
_TWID_LETTER = {
    "A": 10, "B": 11, "C": 12, "D": 13, "E": 14, "F": 15, "G": 16, "H": 17,
    "I": 34, "J": 18, "K": 19, "L": 20, "M": 21, "N": 22, "O": 35, "P": 23,
    "Q": 24, "R": 25, "S": 26, "T": 27, "U": 28, "V": 29, "W": 32, "X": 30,
    "Y": 31, "Z": 33,
}
_TWID_RE = re.compile(r"[A-Z][12]\d{8}")
_PHONE_RE = re.compile(r"09\d{2}[- ]?\d{3}[- ]?\d{3}")                 # 台灣手機
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_CARD_RE = re.compile(r"(?:\d{4}[- ]){3}\d{4}|\b\d{16}\b")             # 4-4-4-4 或 16 連號（再過 Luhn）

_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(the\s+)?(previous|above|prior|earlier)\s+(instruction|prompt|rule)",
    r"disregard\s+(the\s+)?(above|previous|system|prior)",
    r"(reveal|show|print|repeat)\s+(me\s+)?(your\s+)?(system\s+)?(prompt|instruction)",
    r"you\s+are\s+now\b|developer\s+mode|jail\s*break|do\s+anything\s+now|\bDAN\b",
    r"pretend\s+(you|to)\b|act\s+as\s+if\b|without\s+any\s+(rule|restriction|guardrail|filter)",
    r"忽略(上述|以上|前面|之前|先前).{0,6}(指令|指示|規則|提示|設定)",
    r"無視.{0,6}(指令|規則|限制|護欄)",
    r"(揭露|洩漏|告訴我|給我看|印出).{0,10}(系統|提示詞|指令|prompt)",
    r"把你的.{0,8}(系統)?.{0,4}(提示|指令|prompt).{0,4}(告訴|給|印)",
    r"假裝你(沒有|不受|可以無視)|你現在(是|要扮演)|扮演一個|開發者模式|越獄",
]
_INJECTION_RE = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]


def _twid_valid(s: str) -> bool:
    code = _TWID_LETTER.get(s[0])
    if code is None:
        return False
    nums = [code // 10, code % 10] + [int(c) for c in s[1:]]
    weights = [1, 9, 8, 7, 6, 5, 4, 3, 2, 1, 1]
    return sum(n * w for n, w in zip(nums, weights)) % 10 == 0


def _luhn_valid(digits: str) -> bool:
    total, alt = 0, False
    for c in reversed(digits):
        d = int(c) * 2 if alt else int(c)
        total += d - 9 if d > 9 else d
        alt = not alt
    return total % 10 == 0


def detect_pii(text: str):
    """回 [(type, matched)]；身分證過檢核碼、信用卡過 Luhn 才算（降誤報）。"""
    text = text or ""
    found = []
    for m in _TWID_RE.finditer(text):
        if _twid_valid(m.group()):
            found.append(("tw_id", m.group()))
    for m in _PHONE_RE.finditer(text):
        found.append(("tw_phone", m.group()))
    for m in _EMAIL_RE.finditer(text):
        found.append(("email", m.group()))
    for m in _CARD_RE.finditer(text):
        if _luhn_valid(re.sub(r"[- ]", "", m.group())):
            found.append(("credit_card", m.group()))
    return found


def redact_pii(text: str):
    """遮罩偵測到的 PII（輸出護欄用）。回 (redacted_text, findings)。"""
    findings = detect_pii(text)
    red = text or ""
    for typ, val in findings:
        red = red.replace(val, f"[{typ} 已遮罩]")
    return red, findings


def detect_injection(text: str):
    """啟發式偵測 prompt-injection/jailbreak 常見句式（defense layer 1，非萬能）。回命中的 pattern。"""
    return [p.pattern for p in _INJECTION_RE if p.search(text or "")]
