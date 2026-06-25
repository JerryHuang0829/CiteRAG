"""pgvector 向量資料庫後端（FAISS 的可切換替代）。

預設用 FAISS（見 core.USE_PGVECTOR）；設環境變數 CITERAG_PGVECTOR=1 改用本後端。
相對 FAISS 多了：**持久化、SQL metadata 過濾**（如只查某來源文件）、可水平擴展——
但對小語料（78 chunks）FAISS flat 即最佳；pgvector 用於展示「何時該升級到向量資料庫」。

連線參數可由 PG_HOST/PG_PORT/PG_DB/PG_USER/PG_PASSWORD 覆寫（預設對應 docker 容器 citerag-pg）。
向量正規化 + 內積（<#>）= cosine，與 FAISS IndexFlatIP 一致。
"""
import os

import psycopg2
from psycopg2.extras import execute_values

DIM = 512   # BAAI/bge-small-zh-v1.5 維度
_CONN = None


def _conn():
    global _CONN
    if _CONN is None or _CONN.closed:
        _CONN = psycopg2.connect(
            host=os.environ.get("PG_HOST", "localhost"),
            port=os.environ.get("PG_PORT", "5432"),
            dbname=os.environ.get("PG_DB", "citerag"),
            user=os.environ.get("PG_USER", "postgres"),
            password=os.environ.get("PG_PASSWORD", "citerag"),
        )
    return _CONN


def _vec(arr) -> str:
    return "[" + ",".join(f"{float(x):.6f}" for x in arr) + "]"


def ensure_schema(cur):
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    cur.execute(f"""CREATE TABLE IF NOT EXISTS chunks (
        id serial PRIMARY KEY, text text, source text, page int, embedding vector({DIM}))""")
    cur.execute("CREATE INDEX IF NOT EXISTS chunks_embedding_idx "
                "ON chunks USING hnsw (embedding vector_ip_ops)")


def rebuild(chunks, embeddings) -> int:
    conn = _conn()
    cur = conn.cursor()
    ensure_schema(cur)
    cur.execute("TRUNCATE chunks RESTART IDENTITY")
    rows = [(c["text"], c["source"], c["page"], _vec(embeddings[i])) for i, c in enumerate(chunks)]
    execute_values(cur, "INSERT INTO chunks (text, source, page, embedding) VALUES %s",
                   rows, template="(%s,%s,%s,%s::vector)")
    conn.commit()
    return len(rows)


def search(query_vec, k: int, source: str | None = None) -> list[dict]:
    # 向量相似度（<#>＝負內積，正規化下＝cosine）+ 可選 source metadata 過濾（FAISS 做不到）
    cur = _conn().cursor()
    qv = _vec(query_vec)
    where = "WHERE source = %s" if source else ""
    params = [qv] + ([source] if source else []) + [qv, k]
    cur.execute(f"""SELECT text, source, page, (embedding <#> %s::vector) * -1 AS score
        FROM chunks {where}
        ORDER BY embedding <#> %s::vector
        LIMIT %s""", params)
    return [{"text": t, "source": s, "page": p, "score": float(sc)} for t, s, p, sc in cur.fetchall()]
