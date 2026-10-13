# 09 - RAG Evaluation Harness

You cannot reduce hallucination you do not measure. This harness evaluates retrieval and generation separately.

**Retrieval (offline, no API key)**: Hit@k, Recall@k, MRR using BM25 against a golden set.
**Generation (optional, LLM-as-judge)**: faithfulness, answer relevance, correctness vs reference.
**CI gate**: exits non-zero if metrics fall below thresholds in `thresholds.json`.

```bash
pip install -r requirements.txt
python -m src.run_eval                 # retrieval metrics, offline
python -m src.run_eval --judge         # + LLM-judge metrics
python -m src.run_eval --gate          # fail the build on regression
```
Edit `data/golden.jsonl` with real questions, their relevant chunk ids, and reference answers.
