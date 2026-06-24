"""建索引：讀 ../data/*.pdf → chunk → bge-small-zh embedding → FAISS。
用法（在 rag/ 目錄）：python ingest.py
"""
import core


def main():
    chunks = core.load_and_chunk()
    n_src = len({c["source"] for c in chunks})
    print(f"共 {len(chunks)} 個 chunk，來自 {n_src} 份 PDF")
    core.build_index(chunks)
    print(f"索引已建：{core.INDEX_PATH}")
    print(f"chunk 清單：{core.CHUNKS_PATH}")


if __name__ == "__main__":
    main()
