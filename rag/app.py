"""網頁介面（本地 RAG + Agent），免 PowerShell。
分頁一：文件問答（RAG，帶頁碼引用）。
分頁二：Agent（多工具，顯示工具呼叫軌跡）。
用法（rag/）：python app.py  → http://127.0.0.1:7860
前置：Ollama 要在跑且看得到 qwen3:4b-instruct。
"""
import gradio as gr

import agent
import core


def rag_stream(q):
    q = (q or "").strip()
    if not q:
        yield "請輸入問題。"
        return
    yield "⏳ 檢索中…（首次會載入 reranker，稍久）"
    try:
        hits = core.retrieve(q)
    except Exception as e:
        yield f"檢索失敗：{e}（請先跑 ingest.py 建索引）"
        return
    src = "　|　".join(f"{h['source']} p.{h['page']}" for h in hits)
    yield "⏳ 已找到資料，生成回答中（本地 CPU 約 30–60 秒，請稍候）…"
    acc = ""
    try:
        for tok in core.generate_iter(core.build_prompt(q, hits)):
            acc += tok
            yield acc
    except Exception as e:
        yield acc + f"\n\n[生成失敗：{e}　請確認 Ollama 在跑]"
        return
    cleaned, stripped = core.verify_citations(acc, {h["page"] for h in hits})
    note = (f"\n\n> ⚠️ 引用護欄：已剝除 {len(stripped)} 個檢索範圍外頁碼 {stripped}"
            if stripped else "")
    yield cleaned + note + f"\n\n---\n**參考來源**：{src}"


def agent_stream(q):
    q = (q or "").strip()
    if not q:
        yield "請輸入需求。"
        return
    yield "⏳ Agent 思考中…（多步，每步約 30–60 秒，工具軌跡會逐步出現）"
    try:
        for log in agent.run_iter(q):
            yield log
    except Exception as e:
        yield f"Agent 失敗：{e}（請確認 Ollama 在跑）"


def vlm_answer(image_path, question):
    if not image_path:
        yield "請先上傳圖片。"
        return
    yield "⏳ Gemma 讀圖中…（本地 CPU，約 30–90 秒）"
    q = (question or "請讀出圖片中的所有文字與數值。").strip()
    try:
        yield core.read_image(image_path, q)
    except Exception as e:
        yield f"VLM 失敗：{e}（確認已 ollama pull gemma3:4b、Ollama 在跑）"


with gr.Blocks(title="CiteRAG") as demo:
    gr.Markdown("# 本地檢索增強問答系統 CiteRAG（帶頁碼引用的 RAG + Agent）\n"
                f"純本地 CPU + {core.RERANK_MODEL.split('/')[-1]} rerank，回答約 30–60 秒/步，請耐心等。")

    with gr.Tab("📄 文件問答 (RAG)"):
        rq = gr.Textbox(label="問題", placeholder="例：鴻海 2022 第四季毛利率是多少？")
        rbtn = gr.Button("送出", variant="primary")
        rout = gr.Markdown(label="答案")
        rbtn.click(rag_stream, rq, rout)
        rq.submit(rag_stream, rq, rout)
        gr.Examples(["鴻海 2022 全年 EPS 是多少？",
                     "鴻海 2022 第四季毛利率是多少？",
                     "興櫃股票市場是何時成立的？"], rq)

    with gr.Tab("🤖 Agent (多工具)"):
        gr.Markdown("Agent 會自己選工具：查財報原文 / 查財務指標 / 建追蹤筆記，並顯示每步軌跡。")
        aq = gr.Textbox(label="需求", placeholder="例：鴻海 2022 EPS 多少？順便幫我記一筆追蹤下季毛利率")
        abtn = gr.Button("執行", variant="primary")
        aout = gr.Markdown(label="軌跡與結果")
        abtn.click(agent_stream, aq, aout)
        aq.submit(agent_stream, aq, aout)
        gr.Examples(["鴻海 2022 EPS 多少？順便幫我記一筆追蹤下季毛利率",
                     "查興櫃市場何時成立，並把答案記成一筆筆記",
                     "鴻海 2022 全年營收多少？"], aq)

    with gr.Tab("🖼️ 圖片問答 (VLM / Gemma)"):
        gr.Markdown("上傳圖片（如設備銘牌/儀表），由 Gemma 視覺模型讀圖。"
                    "示範『影像→VLM』與『文字→LLM』的模型分流。")
        vimg = gr.Image(type="filepath", label="上傳圖片")
        vq = gr.Textbox(label="問題（留空＝讀出所有文字）",
                        placeholder="例：這台設備的型號和額定壓力是多少？")
        vbtn = gr.Button("讀圖", variant="primary")
        vout = gr.Markdown(label="結果")
        vbtn.click(vlm_answer, [vimg, vq], vout)


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
