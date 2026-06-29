"""FinMind 財務資料查詢 —— agent 的結構化「資料庫」後端（對照 RAG 的文字檢索）。

示範「數字/結構化問題 → 資料庫查（精確、可彙總、零幻覺）」vs RAG（質性文字）：
  「台積電 2023 EPS」 / 「哪些公司毛利率最高」 → 走這裡（FinMind 查表 / 彙總）
  「鴻海為什麼毛利率下滑」                    → 走 search_filings（RAG 讀文字）

- lookup(company, metric, year)         單一公司單一指標（精確）
- compare(metric, companies, year, ...)  多家比較/排名/篩選（RAG 做不到的彙總）
資料源：FinMind TaiwanStockFinancialStatements（季報；全年＝四季加總）+ TaiwanStockInfo（全市場名→碼）。
無 token 可用；有 token 放環境變數 FINMIND_TOKEN 提高額度。網路/額度失敗回可讀訊息。
"""
import json
import os
import re
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"

# 內建快取表（快、離線可用）；未命中再 lazy 抓 TaiwanStockInfo 全市場表
NAME_TO_CODE = {
    "台積電": "2330", "鴻海": "2317", "聯發科": "2454", "台達電": "2308",
    "聯電": "2303", "中華電": "2412", "富邦金": "2881", "國泰金": "2882",
    "台塑": "1301", "南亞": "1303", "中鋼": "2002", "長榮": "2603",
    "廣達": "2382", "日月光": "3711", "和碩": "4938", "緯創": "3231",
}
# 「主要大公司」預設比較範圍（compare 未指定公司時用）
DEFAULT_UNIVERSE = ["台積電", "鴻海", "聯發科", "台達電", "廣達", "日月光", "和碩", "緯創", "聯電", "中華電"]

_CACHE = {}
_ALL_CODES = None
_CACHE_FILE = Path(__file__).resolve().parent / ".findata_cache.json"   # 跨程序磁碟快取，省 FinMind 額度
_DISK = None


def _disk():
    global _DISK
    if _DISK is None:
        try:
            _DISK = json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            _DISK = {}
    return _DISK


def _disk_save():
    try:
        _CACHE_FILE.write_text(json.dumps(_DISK, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def _load_all_codes():
    global _ALL_CODES
    if _ALL_CODES is not None:
        return _ALL_CODES
    d = _disk()
    if "codes" in d:                                 # 磁碟快取命中（不打 API）
        _ALL_CODES = d["codes"]
        return _ALL_CODES
    try:
        url = FINMIND_URL + "?" + urllib.parse.urlencode({"dataset": "TaiwanStockInfo"})
        with urllib.request.urlopen(url, timeout=40) as r:
            rows = json.loads(r.read()).get("data", [])
        m = {}
        for row in rows:
            sid, name = str(row.get("stock_id", "")), row.get("stock_name")
            if name and sid.isdigit() and len(sid) == 4:
                m.setdefault(name, sid)              # 4 碼股票名→碼（全市場 ~2500 家）
        _ALL_CODES = m
        d["codes"] = m
        _disk_save()
    except Exception:
        _ALL_CODES = {}
    return _ALL_CODES


def _resolve_code(company: str):
    c = (company or "").strip()
    if c.isdigit():
        return c
    if c in NAME_TO_CODE:                             # 內建精確
        return NAME_TO_CODE[c]
    allc = _load_all_codes()
    if c in allc:                                     # 全市場精確
        return allc[c]
    # 退而求子串：取「最長」匹配名，避免短名前綴誤吃（台塑 vs 台塑化、南亞 vs 南亞科）
    best = None
    for name, code in NAME_TO_CODE.items():
        if name in c and (best is None or len(name) > len(best[0])):
            best = (name, code)
    for name, code in allc.items():
        if name and name in c and (best is None or len(name) > len(best[0])):
            best = (name, code)
    return best[1] if best else None


def _fetch(stock_id: str, year):
    key = (stock_id, str(year))
    if key in _CACHE:
        return _CACHE[key]
    d = _disk()
    dkey = f"fin:{stock_id}:{year}"
    if dkey in d:                                    # 磁碟快取命中（不打 API，省額度）
        _CACHE[key] = d[dkey]
        return d[dkey]
    params = {"dataset": "TaiwanStockFinancialStatements", "data_id": stock_id,
              "start_date": f"{year}-01-01", "end_date": f"{year}-12-31"}
    token = os.environ.get("FINMIND_TOKEN")
    if token:
        params["token"] = token
    try:
        url = FINMIND_URL + "?" + urllib.parse.urlencode(params)
        with urllib.request.urlopen(url, timeout=30) as r:
            rows = json.loads(r.read()).get("data", [])
    except Exception:
        rows = None
    if rows:                                          # 只快取成功結果（失敗 None/空不寫，下次可重試）
        _CACHE[key] = rows
        if sum(1 for r in rows if r.get("type") == "EPS") >= 4:   # 僅磁碟快取「完整四季」年度；
            d[dkey] = rows                                        # 當年未滿四季的 partial-year 不永久快取，補齊後重抓
            _disk_save()
    return rows


def _resolve_year(code: str, year):
    if year is not None:
        return str(year)
    now = datetime.now().year
    for y in range(now, now - 5, -1):                # 取最近有完整 4 季的年度
        rows = _fetch(code, y)
        if rows and sum(1 for r in rows if r.get("type") == "EPS") >= 4:
            return str(y)
    return str(now - 1)


def _sum(rows, t):
    return sum(r["value"] for r in rows if r.get("type") == t and isinstance(r.get("value"), (int, float)))


def _money(v):
    if abs(v) >= 1e12:
        return f"{v / 1e12:.3f} 兆元"
    if abs(v) >= 1e8:
        return f"{v / 1e8:.0f} 億元"
    return f"{v:,.0f} 元"


def _metric_value(rows, metric):
    """回傳該指標的全年數值（float），不支援回 None。"""
    m = (metric or "").strip().lower()
    if "eps" in m or "每股" in m:
        return _sum(rows, "EPS")
    if "毛利率" in m:
        rev = _sum(rows, "Revenue")
        return (_sum(rows, "GrossProfit") / rev * 100) if rev else None
    if "毛利" in m:
        return _sum(rows, "GrossProfit")
    if "營收" in m or "revenue" in m or "營業收入" in m:
        return _sum(rows, "Revenue")
    if "淨利" in m or "獲利" in m:
        return _sum(rows, "IncomeAfterTaxes")
    if "營業利益" in m:
        return _sum(rows, "OperatingIncome")
    return None


def _fmt_metric(metric, v):
    m = (metric or "").lower()
    if "eps" in m or "每股" in m:
        return f"{v:.2f} 元"
    if "毛利率" in m:
        return f"{v:.2f}%"
    return _money(v)


def lookup(company: str, metric: str, year=None) -> str:
    code = _resolve_code(company)
    if not code:
        return f"指標庫查無公司「{company}」（可給股號或公司名）。"
    year = _resolve_year(code, year)
    rows = _fetch(code, year)
    if rows is None:
        return "FinMind 查詢失敗（網路或額度）。"
    if not rows:
        return f"FinMind 查無 {company}({code}) {year} 年財報。"
    v = _metric_value(rows, metric)
    if v is None:
        return f"{company}({code}) {year}：暫不支援指標「{metric}」（支援 EPS／營收／毛利率／毛利／淨利／營業利益）。"
    nq = sum(1 for r in rows if r.get("type") == "EPS")
    period = f"前{nq}季累計" if 0 < nq < 4 else "全年"   # 當年未滿四季→誠實標「前N季累計」而非「全年」
    return f"{company}({code}) {year} {period}{metric} {_fmt_metric(metric, v)}"


def _fetch_price(stock_id: str):
    today = datetime.now().strftime("%Y-%m-%d")
    key = ("price", stock_id, today)
    if key in _CACHE:
        return _CACHE[key]
    d = _disk()
    dkey = f"price:{stock_id}:{today}"                # 股價每天變 → 以日期為快取 key（當日重複免打 API）
    if dkey in d:
        _CACHE[key] = d[dkey]
        return d[dkey]
    start = (datetime.now() - timedelta(days=400)).strftime("%Y-%m-%d")
    params = {"dataset": "TaiwanStockPrice", "data_id": stock_id, "start_date": start}
    token = os.environ.get("FINMIND_TOKEN")
    if token:
        params["token"] = token
    try:
        url = FINMIND_URL + "?" + urllib.parse.urlencode(params)
        with urllib.request.urlopen(url, timeout=30) as r:
            rows = json.loads(r.read()).get("data", [])
    except Exception:
        rows = None
    _CACHE[key] = rows
    if rows:
        d[dkey] = rows
        _disk_save()
    return rows


def price(company: str, year=None) -> str:
    code = _resolve_code(company)
    if not code:
        return f"查無公司「{company}」（可給股號或公司名）。"
    rows = _fetch_price(code)
    if rows is None:
        return "FinMind 查詢失敗（網路或額度）。"
    if not rows:
        return f"FinMind 查無 {company}({code}) 股價。"
    last = rows[-1]
    close, chg, d0 = last.get("close"), last.get("spread", 0), last.get("date")
    out = f"{company}({code}) {d0} 收盤 {close:.1f} 元（當日 {'+' if chg >= 0 else ''}{chg:.0f}）"
    if len(rows) > 200 and rows[0].get("close"):     # 近一年漲跌幅
        old = rows[0]["close"]
        out += f"；近一年 {'+' if close >= old else ''}{(close / old - 1) * 100:.1f}%"
    return out


def compare(metric: str, companies=None, year=None, threshold=None) -> str:
    """多家公司同一指標的比較/排名/篩選（RAG 做不到的彙總）。

    companies：逗號分隔的公司名/股號；省略或含「全/主要/大公司」→ 用 DEFAULT_UNIVERSE。
    threshold：如 ">30"、"<10"（搭配毛利率/EPS 篩選）。
    """
    raw = (companies or "").strip()
    if (not raw) or any(k in raw for k in ["全部", "全市場", "所有", "主要", "大公司"]):
        names = DEFAULT_UNIVERSE
    else:
        names = [x for x in re.split(r"[,，、\s]+", raw) if x]
    names = names[:20]                               # 上限，避免過多 API 呼叫

    year = _resolve_year(_resolve_code(names[0]) or "2330", year)
    results = []
    for nm in names:
        code = _resolve_code(nm)
        if not code:
            continue
        rows = _fetch(code, year)
        if not rows:
            continue
        v = _metric_value(rows, metric)
        if v is not None:
            results.append((nm, code, v))
    if not results:
        return f"查無可比較資料（{metric} {year}）。"

    op = num = None
    if threshold:
        mt = re.match(r"\s*(>=|<=|>|<)\s*([\d.]+)", str(threshold))
        if mt:
            op, num = mt.group(1), float(mt.group(2))
    if op:
        keep = {">": lambda x: x > num, "<": lambda x: x < num,
                ">=": lambda x: x >= num, "<=": lambda x: x <= num}[op]
        prefilter = results
        results = [t for t in results if keep(t[2])]
        if not results:
            rng = ", ".join(f"{n} {_fmt_metric(metric, v)}" for n, _, v in prefilter)
            return f"{year} {metric} 沒有公司符合 {threshold}（候選：{rng}）。"

    results.sort(key=lambda t: t[2], reverse=True)
    body = " ＞ ".join(f"{nm} {_fmt_metric(metric, v)}" for nm, code, v in results)
    cond = f"（篩選 {threshold}）" if op else ""
    return f"{year} {metric} 排名{cond}（高→低）：{body}"
