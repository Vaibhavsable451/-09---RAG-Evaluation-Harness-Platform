import json, sys, re
from statistics import mean
from rank_bm25 import BM25Okapi
from src.corpus import load_sections
from src.metrics import hit_at_k, recall_at_k, reciprocal_rank

tok = lambda s: re.findall(r"[a-z0-9]+", s.lower())
sections = load_sections()
ids = list(sections)
bm25 = BM25Okapi([tok(sections[i]) for i in ids])
golden = [json.loads(l) for l in open("data/golden.jsonl", encoding="utf-8") if l.strip()]

rows = []
for g in golden:
    scores = bm25.get_scores(tok(g["q"]))
    ranked = [ids[i] for i in sorted(range(len(ids)), key=lambda i: -scores[i])]
    rows.append({"q": g["q"], "ranked": ranked, "hit@1": hit_at_k(ranked, g["relevant"], 1),
                 "hit@3": hit_at_k(ranked, g["relevant"], 3), "recall@3": recall_at_k(ranked, g["relevant"], 3),
                 "mrr": reciprocal_rank(ranked, g["relevant"])})
    print(f"{rows[-1]['hit@3']:.0f} rr={rows[-1]['mrr']:.2f}  {g['q']}  -> {ranked[:3]}")

summary = {m: round(mean(r[m] for r in rows), 3) for m in ["hit@1", "hit@3", "recall@3", "mrr"]}
print("\nRETRIEVAL:", summary)

if "--judge" in sys.argv:
    from src.judge import judge
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from src.config import get_llm
    llm, agg = get_llm(), []
    for g, r in zip(golden, rows):
        ctx = "\n".join(sections[i] for i in r["ranked"][:2])
        ans = (ChatPromptTemplate.from_template("Answer from context only.\n{c}\nQ: {q}") | llm | StrOutputParser()).invoke({"c": ctx, "q": g["q"]})
        s = judge(ctx, g["q"], g["reference"], ans); agg.append(s)
    print("GENERATION:", {k: round(mean(getattr(s, k) for s in agg), 2) for k in ["faithfulness", "relevance", "correctness"]})

if "--gate" in sys.argv:
    th = json.load(open("thresholds.json"))
    failed = {k: (summary[k], v) for k, v in th.items() if summary[k] < v}
    if failed:
        print("GATE FAILED:", failed); sys.exit(1)
    print("GATE PASSED")
