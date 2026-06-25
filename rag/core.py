"""W1 最小 RAG 核心：PDF → chunk → bge-small-zh embedding → FAISS → 帶頁碼引用回答。

語料與架構無關：把 PDF 放進 ../data/，跑 ingest.py 建索引，再用 ask.py 問。
embedding 走 fastembed(ONNX, CPU)；生成走本地 Ollama qwen3:4b-instruct(streaming)。
"""
from __future__ import annotations

import base64
import json
import math
import os
import re
import sys
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

# Windows 非管理員/未開發者模式下 HF cache 的 symlink 會失敗(WinError 1314)，
# 導致 snapshot 缺檔(tokenizer_config.json)；強制改用複製。必須在 import fastembed 前設。
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import faiss
import numpy as np
import pymupdf4llm
from fastembed import TextEmbedding
from langchain_text_splitters import RecursiveCharacterTextSplitter

sys.stdout.reconfigure(encoding="utf-8")   # Windows cp950 無法編部分字元，強制 UTF-8

# ---- 路徑（相對本檔，與 cwd 無關）----
BASE = Path(__file__).resolve().parent.parent          # CiteRAG/
DATA_DIR = BASE / "data"
INDEX_DIR = BASE / "index"
MODEL_CACHE = BASE / "hf_cache"                         # embedding 權重放 E:（不佔 C:）；symlink 停用故全新目錄重抓
INDEX_PATH = INDEX_DIR / "faiss.index"
CHUNKS_PATH = INDEX_DIR / "chunks.jsonl"

# ---- 設定 ----
EMBED_MODEL = "BAAI/bge-small-zh-v1.5"                  # fastembed 確認支援, dim 512
CHUNK_SIZE = 500
CHUNK_OVERLAP = 80
TOP_K = 4
USE_RERANK = True                                      # 預設開：rerank 對小語料是最大品質槓桿（見 eval_rerank）
# n=12 上 bge 與 jina 點估計近同、bge hit@3 略高且 ΔMRR 顯著（jina 觸 0）；選型待擴題收窄 CI 再定論
RERANK_MODEL = "BAAI/bge-reranker-base"
USE_HYBRID = True                                      # hybrid：dense + BM25(稀疏) 經 RRF 融合，補精確詞/數字 silent fail
RRF_K = 60                                             # Reciprocal Rank Fusion 常數（標準值）
USE_PGVECTOR = os.environ.get("CITERAG_PGVECTOR", "0") == "1"   # 1＝改用 pgvector 向量資料庫後端（預設 FAISS）

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
GEN_MODEL = "qwen3:4b-instruct"
VLM_MODEL = "gemma3:4b"                                 # 視覺模型（讀圖；影像→文字）
NUM_THREAD = 8                                          # W0 實測最快
NUM_CTX = 4096

_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
)


def _page_of(d: dict) -> int:
    # pymupdf4llm page_chunks 的頁碼在 metadata['page_number']（1-based，實測確認）
    md = d.get("metadata") or {}
    p = md.get("page_number")
    if p is None:
        p = md.get("page", d.get("page"))
    return int(p) if p is not None else -1


_IMG_PLACEHOLDER = re.compile(r"[^\n]*intentionally omitted[^\n]*\n?|!\[\]\([^)]*\)")


def _clean_text(text: str) -> str:
    # 移除 pymupdf4llm 的圖片佔位（==> picture ... intentionally omitted / markdown 圖片）
    return _IMG_PLACEHOLDER.sub("", text or "").strip()


def load_and_chunk(data_dir: Path = DATA_DIR) -> list[dict]:
    pdfs = sorted(data_dir.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"找不到 PDF，請放到 {data_dir}")
    chunks: list[dict] = []
    for pdf in pdfs:
        for d in pymupdf4llm.to_markdown(str(pdf), page_chunks=True):
            page = _page_of(d)
            text = _clean_text(d.get("text"))
            if not text:
                continue
            for piece in _SPLITTER.split_text(text):
                piece = piece.strip()
                if len(piece) < 10:
                    continue
                chunks.append({"text": piece, "source": pdf.name, "page": page})
    return chunks


_EMBEDDER = None     # 模組級快取，避免每次查詢重載模型
_RERANKER = None


def get_embedder() -> TextEmbedding:
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = TextEmbedding(model_name=EMBED_MODEL, cache_dir=str(MODEL_CACHE))
    return _EMBEDDER


def _normalize(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype("float32")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def embed_passages(embedder: TextEmbedding, texts: list[str]) -> np.ndarray:
    return _normalize(np.array(list(embedder.embed(texts)), dtype="float32"))


def embed_query(embedder: TextEmbedding, query: str) -> np.ndarray:
    # bge 查詢端會自動加指令前綴（query_embed），與文件端區分
    vec = np.array(list(embedder.query_embed([query]))[0], dtype="float32").reshape(1, -1)
    return _normalize(vec)


def build_index(chunks: list[dict]) -> None:
    vecs = embed_passages(get_embedder(), [c["text"] for c in chunks])
    if USE_PGVECTOR:                                    # pgvector 後端：寫進資料庫
        import pgstore
        pgstore.rebuild(chunks, vecs)
        return
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    index = faiss.IndexFlatIP(vecs.shape[1])           # 正規化向量 + 內積 = cosine
    index.add(vecs)
    faiss.write_index(index, str(INDEX_PATH))
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")


def load_index():
    if not INDEX_PATH.exists() or not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"找不到索引（{INDEX_PATH}）。請先在 rag/ 跑：python ingest.py"
        )
    index = faiss.read_index(str(INDEX_PATH))
    chunks = [json.loads(line) for line in open(CHUNKS_PATH, encoding="utf-8")]
    return index, chunks


def retrieve(query: str, k: int = TOP_K, use_rerank: bool | None = None,
             use_hybrid: bool | None = None, source: str | None = None) -> list[dict]:
    if use_rerank is None:
        use_rerank = USE_RERANK
    if use_hybrid is None:
        use_hybrid = USE_HYBRID
    if USE_PGVECTOR:                                    # pgvector 後端（支援 source metadata 過濾，FAISS 做不到）
        import pgstore
        qv = embed_query(get_embedder(), query)[0]
        cand = pgstore.search(qv, RERANK_CANDIDATES if use_rerank else k, source=source)
        if use_rerank and cand:
            return rerank_dicts(_live_reranker(), query, cand, k)
        return cand[:k]
    index, chunks = load_index()
    embedder = get_embedder()
    if use_rerank:
        # 候選（dense 或 hybrid dense+BM25/RRF）→ cross-encoder 重排 → 取 top-k
        cand = candidates(index, embedder, chunks, query, RERANK_CANDIDATES, use_hybrid)
        ranked = rerank_scored(_live_reranker(), query, cand, chunks)[:k]
        return [dict(chunks[i], score=float(s)) for i, s in ranked]
    qv = embed_query(embedder, query)
    scores, idxs = index.search(qv, k)
    return [dict(chunks[i], score=float(s)) for s, i in zip(scores[0], idxs[0]) if i >= 0]


RERANK_CANDIDATES = 20            # rerank 前先取的 FAISS 候選數


def faiss_topn(index, embedder, query: str, n: int) -> list[int]:
    qv = embed_query(embedder, query)
    _, idxs = index.search(qv, n)
    return [int(i) for i in idxs[0] if i >= 0]


def get_reranker(model_name: str):
    from fastembed.rerank.cross_encoder import TextCrossEncoder
    return TextCrossEncoder(model_name=model_name, cache_dir=str(MODEL_CACHE))


def rerank_order(reranker, query: str, texts: list[str]) -> list[int]:
    # 回傳 texts 重排後的索引順序（分數高→低）
    scores = list(reranker.rerank(query, texts))
    return sorted(range(len(texts)), key=lambda i: scores[i], reverse=True)


def rerank_scored(reranker, query: str, idxs: list[int], chunks) -> list[tuple]:
    # 回傳 [(chunk_idx, rerank_score), ...] 依分數高→低
    scores = list(reranker.rerank(query, [chunks[i]["text"] for i in idxs]))
    return sorted(zip(idxs, scores), key=lambda t: t[1], reverse=True)


def rerank_dicts(reranker, query: str, dicts: list[dict], k: int) -> list[dict]:
    # 對 chunk dict 清單重排（pgvector 後端用：候選直接是 dict，非索引）
    scores = list(reranker.rerank(query, [d["text"] for d in dicts]))
    ranked = sorted(zip(dicts, scores), key=lambda t: t[1], reverse=True)[:k]
    return [dict(d, score=float(s)) for d, s in ranked]


def _live_reranker():
    global _RERANKER
    if _RERANKER is None:
        _RERANKER = get_reranker(RERANK_MODEL)
    return _RERANKER


# ---- BM25 稀疏檢索（手刻 Okapi BM25）+ RRF 融合 ----
# 純向量(dense)對精確詞/數字/股號會 silent fail；加字面比對的 BM25 互補，RRF 融合兩路候選。
_CJK = re.compile(r"[一-鿿]")
_BM25_TOK = re.compile(r"[A-Za-z]+|\d+(?:\.\d+)?|[一-鿿]")


def bm25_tokens(text: str) -> list[str]:
    # 英文詞/數字(含小數)整段保留 → "2317"/"10.21"/"EPS" 成精確 token；中文加 bigram 增辨識度
    toks, prev = [], None
    for t in _BM25_TOK.findall(text or ""):
        if _CJK.match(t):
            if prev is not None:
                toks.append(prev + t)
            toks.append(t)
            prev = t
        else:
            toks.append(t.lower())
            prev = None
    return toks


class _BM25:
    def __init__(self, corpus_tokens, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in corpus_tokens]
        self.dl = [len(d) for d in corpus_tokens]
        self.N = len(corpus_tokens)
        self.avgdl = (sum(self.dl) / self.N) if self.N else 0.0
        df = Counter()
        for d in corpus_tokens:
            df.update(set(d))
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()}

    def scores(self, q_tokens) -> list[float]:
        out = [0.0] * self.N
        for t in q_tokens:
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i in range(self.N):
                f = self.tf[i].get(t, 0)
                if f:
                    out[i] += idf * f * (self.k1 + 1) / (
                        f + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avgdl))
        return out


_BM25_IDX = None


def _get_bm25(chunks):
    global _BM25_IDX
    if _BM25_IDX is None:
        _BM25_IDX = _BM25([bm25_tokens(c["text"]) for c in chunks])
    return _BM25_IDX


def bm25_topn(chunks, query: str, n: int) -> list[int]:
    scores = _get_bm25(chunks).scores(bm25_tokens(query))
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [i for i in order if scores[i] > 0][:n]


def _rrf(rankings, k: int = RRF_K) -> dict:
    # Reciprocal Rank Fusion：在多路排名都靠前的 chunk，融合分數越高（1/(k+rank)）
    fused = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return fused


def candidates(index, embedder, chunks, query: str, n: int, hybrid: bool) -> list[int]:
    # 重排前的候選集：dense top-n，或 hybrid(dense + BM25 經 RRF) top-n
    dense = faiss_topn(index, embedder, query, n)
    if not hybrid:
        return dense
    sparse = bm25_topn(chunks, query, n)
    fused = _rrf([dense, sparse])
    return sorted(fused, key=lambda i: fused[i], reverse=True)[:n]


def build_prompt(query: str, hits: list[dict]) -> str:
    refs = "\n\n".join(
        f"[來源{n}]（{h['source']}, p.{h['page']}）\n{h['text']}"
        for n, h in enumerate(hits, 1)
    )
    return (
        "你是文件問答助手。只能根據以下「參考資料」回答問題，"
        "並在每個重點後標註頁碼，例如 (p.3)。\n"
        "若參考資料中沒有答案，明確回答「參考資料中查無此資訊」，不要編造。\n\n"
        f"參考資料：\n{refs}\n\n"
        f"問題：{query}\n"
        "請用繁體中文作答並附頁碼引用："
    )


# 一個引用 token：括號式 (p.3,4)/（p.3、4） 或裸式 p.3；括號式可含逗號/頓號多頁
_CITE_TOKEN = re.compile(r"[（(]\s*[pP]\.?\s*([\d\s,、]+?)\s*[)）]|[pP]\.\s*(\d+)")
_CITE_NUM = re.compile(r"\d+")


def pages_in(text: str) -> set:
    # 抽出文字中所有引用頁碼：(p.3) /（p.3）/ p.3 /（p.3,4）多頁
    out = set()
    for inner, bare in _CITE_TOKEN.findall(text or ""):
        for n in _CITE_NUM.findall(inner or bare or ""):
            out.add(int(n))
    return out


def verify_citations(answer: str, hit_pages) -> tuple:
    """剝除答案中不在「本次檢索命中頁集合」內的頁碼引用（擋 citation-shaped 幻覺）。
    括號內多頁（p.3,4）逐頁判定：全合法保留、全範圍外整段剝除、部分則只留合法頁。
    回傳 (清理後答案, 被剝除頁碼清單)。"""
    allowed = {int(p) for p in hit_pages if p is not None and int(p) >= 0}
    stripped = []

    def _repl(m):
        inner, bare = m.group(1), m.group(2)
        if inner is None:                       # 裸式 p.N
            page = int(bare)
            if page in allowed:
                return m.group(0)
            stripped.append(page)
            return ""
        pages = [int(n) for n in _CITE_NUM.findall(inner)]   # 括號式可多頁
        keep = [p for p in pages if p in allowed]
        stripped.extend(p for p in pages if p not in allowed)
        if not keep:
            return ""
        if len(keep) == len(pages):
            return m.group(0)
        return "(p." + ",".join(str(p) for p in keep) + ")"

    cleaned = _CITE_TOKEN.sub(_repl, answer or "")
    cleaned = re.sub(r"[（(]\s*[,、]?\s*[)）]", "", cleaned)   # 清掉剝除後殘留的空/孤兒括號
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()
    return cleaned, stripped


class OllamaError(RuntimeError):
    """Ollama 連線或 HTTP 失敗時拋出，與程式邏輯錯誤區隔，讓 CLI 印可讀訊息而非裸 traceback。"""


def _open_ollama(url: str, payload: dict):
    # 統一開啟 Ollama 連線：連不上 / 模型未 pull(404) 轉成可讀的 OllamaError
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        return urllib.request.urlopen(req, timeout=600)
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="ignore")[:200]
        raise OllamaError(
            f"Ollama 回應 HTTP {e.code}（常見：模型「{payload.get('model')}」未 pull）。{detail}"
        ) from e
    except OSError as e:
        # URLError / ConnectionRefused / timeout 皆為 OSError 子類
        raise OllamaError(
            f"連不到 Ollama（{url}）：{e}。請確認 Ollama 在跑、且已 ollama pull {payload.get('model')}。"
        ) from e


def generate_iter(prompt: str):
    # 串流生成，逐 token yield（CLI 與網頁 UI 共用）
    payload = {
        "model": GEN_MODEL, "prompt": prompt, "stream": True,
        "options": {"num_ctx": NUM_CTX, "num_thread": NUM_THREAD, "temperature": 0.2},
        "keep_alive": "30m",
    }
    with _open_ollama(OLLAMA_URL, payload) as resp:
        for line in resp:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            tok = obj.get("response", "")
            if tok:
                yield tok
            if obj.get("done"):
                break


def generate_stream(prompt: str) -> str:
    out = []
    for tok in generate_iter(prompt):
        print(tok, end="", flush=True)
        out.append(tok)
    print()
    return "".join(out)


def generate(prompt: str, as_json: bool = False) -> str:
    # 非串流生成（eval 用）；as_json=True 走 grammar 約束輸出合法 JSON
    payload = {
        "model": GEN_MODEL, "prompt": prompt, "stream": False,
        "options": {"num_ctx": NUM_CTX, "num_thread": NUM_THREAD,
                    "temperature": 0.0 if as_json else 0.2},
        "keep_alive": "30m",
    }
    if as_json:
        payload["format"] = "json"
    with _open_ollama(OLLAMA_URL, payload) as resp:
        return json.loads(resp.read()).get("response", "")


def chat(messages: list[dict], as_json: bool = False) -> str:
    # /api/chat：給 messages、回一則訊息（Agent 用）；as_json 走 grammar 約束輸出合法 JSON
    payload = {
        "model": GEN_MODEL, "messages": messages, "stream": False,
        "options": {"num_ctx": NUM_CTX, "num_thread": NUM_THREAD,
                    "temperature": 0.0 if as_json else 0.2},
        "keep_alive": "30m",
    }
    if as_json:
        payload["format"] = "json"
    with _open_ollama(OLLAMA_CHAT_URL, payload) as resp:
        return json.loads(resp.read())["message"]["content"]


def vlm_b64(b64: str, question: str = "請讀出圖片中的所有文字與數值。") -> str:
    # 視覺模型（Gemma）讀 base64 影像 → 文字（API 端點與 read_image 共用）
    payload = {
        "model": VLM_MODEL, "prompt": question, "images": [b64], "stream": False,
        "options": {"num_thread": NUM_THREAD, "temperature": 0.2}, "keep_alive": "30m",
    }
    with _open_ollama(OLLAMA_URL, payload) as resp:
        return json.loads(resp.read()).get("response", "")


def read_image(image_path: str, question: str = "請讀出圖片中的所有文字與數值。") -> str:
    # 視覺模型（Gemma）讀圖：影像 → 文字。CPU 上較慢（數十秒）。
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return vlm_b64(b64, question)


def answer(query: str, k: int = TOP_K) -> None:
    hits = retrieve(query, k)
    print(f"\n— 檢索到 {len(hits)} 段（top-{k}）—")
    for h in hits:
        print(f"  [{h['score']:.3f}] {h['source']} p.{h['page']}: {h['text'][:40]}…")
    print("\n— 回答（streaming）—")
    raw = generate_stream(build_prompt(query, hits))
    cleaned, stripped = verify_citations(raw, {h["page"] for h in hits})
    if stripped:
        print(f"\n[引用護欄] 已剝除檢索範圍外頁碼 {stripped}，校正後：\n{cleaned}")
