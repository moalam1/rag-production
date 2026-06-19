#!/usr/bin/env python3.11
"""
eval/run_eval_local.py — LangSmith-independent gate runner.
Reuses predict() + evaluators from run_eval.py; runs locally, prints averaged
scores. No evaluate(), no LS_CLIENT, no trace upload — immune to 429.
"""
import os, sys, json, time, argparse
from types import SimpleNamespace

from eval.run_eval import (
    predict,
    faithfulness_evaluator,
    answer_relevancy_evaluator,
    source_retrieval_evaluator,
    not_blocked_evaluator,
)

GOLD_PATH = os.path.join(os.path.dirname(__file__), "gold_set.json")
EVALUATORS = [
    ("faithfulness",     faithfulness_evaluator),
    ("answer_relevancy", answer_relevancy_evaluator),
    ("source_retrieval", source_retrieval_evaluator),
    ("not_blocked",      not_blocked_evaluator),
]


def _shim(question, gold_answer, relevant_doc, prediction):
    example = SimpleNamespace(
        inputs={"query": question},
        outputs={"answer": gold_answer, "relevant_doc": relevant_doc},
    )
    run = SimpleNamespace(outputs=prediction or {})
    return run, example


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--category")
    ap.add_argument("--difficulty")
    args = ap.parse_args()

    gold = json.load(open(GOLD_PATH))
    if args.category:   gold = [g for g in gold if g.get("category")   == args.category]
    if args.difficulty: gold = [g for g in gold if g.get("difficulty") == args.difficulty]
    if args.limit:      gold = gold[: args.limit]

    print(f"LOCAL gate — {len(gold)} questions (no LangSmith)")
    print(f"API target: {os.getenv('API_GATEWAY_URL', '(default from .env)')}")
    print("-" * 70)

    sums = {k: 0.0 for k, _ in EVALUATORS}
    n = 0
    rows = []
    for i, g in enumerate(gold, 1):
        q = g["question"]
        try:
            pred = predict({"query": q})
        except Exception as e:
            pred = {"answer": f"ERROR: {e}", "sources": [], "blocked": False}
        run, example = _shim(q, g["ground_truth"], g.get("relevant_doc", ""), pred)

        scores = {}
        for key, fn in EVALUATORS:
            try:
                scores[key] = fn(run, example).get("score", 0.0)
            except Exception as e:
                scores[key] = 0.0
                print(f"  ! evaluator {key} errored on q{i}: {e}")
            sums[key] += scores[key]
        n += 1
        rows.append((g.get("id", f"q{i}"), g.get("difficulty", ""), scores))
        print(f"[{i:>2}/{len(gold)}] {g.get('id','')} "
              + " ".join(f"{k[:4]}={scores[k]:.2f}" for k, _ in EVALUATORS)
              + f"  | {q[:50]}")
        sys.stdout.flush()

    print("-" * 70)
    print("AGGREGATE (mean):")
    for k, _ in EVALUATORS:
        print(f"  {k:<18} {sums[k]/n:.3f}")
    print(f"  n = {n}")

    out = {
        "n": n,
        "means": {k: round(sums[k] / n, 4) for k, _ in EVALUATORS},
        "rows": [{"id": r[0], "difficulty": r[1], **r[2]} for r in rows],
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    json.dump(out, open("/tmp/eval_local_result.json", "w"), indent=2)
    print(f"\nFull results → /tmp/eval_local_result.json")


if __name__ == "__main__":
    main()
