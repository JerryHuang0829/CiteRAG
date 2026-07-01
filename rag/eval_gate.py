"""eval-as-CI-gate：對 golden set 跑品質指標，低於凍結 SLO（slo.py）即 exit 1（擋 CI / PR）。

預設只跑「檢索層」（不需 LLM，只需 index + embedder + reranker，可進 GitHub CI）。
加 --gen 也跑「生成層」（需 LLM：CITERAG_LLM_BACKEND=cloud 或本機 Ollama；建議 PR/nightly 才跑）。
門檻凍結於 slo.py。用法（rag/）：python eval_gate.py [--gen]
"""
import sys
import time

import core
from eval_retrieval import norm
from golden import load_golden
from slo import GENERATION_SLO, RETRIEVAL_SLO


def _rate(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _retry(call, tries=4, base=3):
    # 雲端 free-tier burst 易雙 provider 同時 429；退避重試避免單題 429 crash 整個 gate
    for i in range(tries):
        try:
            return call()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(base * (i + 1))   # 3,6,9s


def retrieval_metrics():
    items = [it for it in load_golden() if it["answerable"]]
    recalls, precs = [], []
    for it in items:
        hits = core.retrieve(it["question"])
        has_gold = [1 if any(norm(g) in norm(h["text"]) for g in it["gold"]) else 0 for h in hits]
        recalls.append(1 if any(has_gold) else 0)
        precs.append(_rate(has_gold))
    return {"context_recall": _rate(recalls), "context_precision": _rate(precs)}, len(items)


def generation_metrics(throttle=1.5):
    from eval_rag_triad import judge_faithful   # 重用既有 judge（judge 已自帶 except→None）
    items = [it for it in load_golden() if it["answerable"]]
    corr, faith = [], []
    skipped = 0
    for it in items:
        hits = core.retrieve(it["question"])
        ctx = "\n\n".join(f"(p.{h['page']}) {h['text']}" for h in hits)
        try:
            raw = _retry(lambda: core.generate(core.build_prompt(it["question"], hits)))
        except Exception:
            skipped += 1            # 重試後仍失敗（rate-limit/outage）→ 跳過該題，不 crash
            continue
        ans, _ = core.verify_citations(raw, {h["page"] for h in hits})
        corr.append(1 if any(norm(g) in norm(ans) for g in it["gold"]) else 0)
        f = judge_faithful(ctx, ans)
        if f is not None:
            faith.append(f)
        time.sleep(throttle)         # 節流：避免 burst 撞 free-tier RPM
    if skipped:
        print(f"  [warn] generation: {skipped}/{len(items)} 題 LLM 重試後仍失敗已跳過（rate-limit/outage）")
    return {"answer_correctness": _rate(corr), "faithfulness": _rate(faith)}, len(corr)


def _check(name, metrics, slo, n):
    print(f"=== {name} gate (golden answerable n={n}) ===")
    failed = []
    for key, floor in slo.items():
        got = metrics.get(key, 0.0)
        ok = got >= floor
        print(f"  [{'PASS' if ok else 'FAIL'}] {key:18} = {got:.3f}  (SLO >= {floor})")
        if not ok:
            failed.append(f"{key} {got:.3f}<{floor}")
    return failed


def main():
    failed = []
    m, n = retrieval_metrics()
    failed += _check("Retrieval", m, RETRIEVAL_SLO, n)
    if "--gen" in sys.argv:
        gm, gn = generation_metrics()
        failed += _check("Generation", gm, GENERATION_SLO, gn)
    if failed:
        print(f"\n[X] GATE FAILED: {' | '.join(failed)} -> 擋下 (exit 1)")
        sys.exit(1)
    print("\n[OK] GATE PASSED: 品質達標")


if __name__ == "__main__":
    main()
