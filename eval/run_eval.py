"""
eval/run_eval.py — LangSmith evaluation runner for Equinix RAG platform.

Usage:
  python3.11 run_eval.py                    # run all 50 questions
  python3.11 run_eval.py --category ai      # run only AI category
  python3.11 run_eval.py --difficulty easy  # run only easy questions
  python3.11 run_eval.py --limit 10         # run first 10 only

Requires:
  pip3.11 install langsmith openai requests

Environment variables (in .env):
  LANGCHAIN_API_KEY     = ls__xxxxxx
  LANGCHAIN_PROJECT     = equinix-rag-production
  OPENAI_API_KEY        = sk-xxxxx
  API_GATEWAY_URL       = https://lxhxqqh3r8.execute-api.us-east-1.amazonaws.com
  API_KEY               = your-api-key
"""
import os
import sys
import json
import time
import time as _time
import argparse
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from langsmith import Client
from langsmith.evaluation import evaluate
from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────
API_URL   = os.getenv("API_GATEWAY_URL", "https://lxhxqqh3r8.execute-api.us-east-1.amazonaws.com")
API_KEY   = os.getenv("API_KEY", "")
LS_CLIENT = Client()
OAI       = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DATASET_NAME = "equinix-rag-gold-set-v1"
GOLD_FILE    = os.path.join(os.path.dirname(__file__), "gold_set.json")


# ── Upload dataset to LangSmith (run once) ────────────────────────
def upload_dataset(gold: list, force: bool = False) -> str:
    """Upload gold set to LangSmith as a dataset."""
    existing = list(LS_CLIENT.list_datasets(dataset_name=DATASET_NAME))
    if existing and not force:
        print(f"Dataset '{DATASET_NAME}' already exists ({existing[0].id}) — skipping upload.")
        return existing[0].id

    if existing and force:
        LS_CLIENT.delete_dataset(dataset_id=existing[0].id)
        print(f"Deleted existing dataset.")

    dataset = LS_CLIENT.create_dataset(
        dataset_name=DATASET_NAME,
        description="50 gold question-answer pairs for Equinix RAG evaluation",
    )

    LS_CLIENT.create_examples(
        inputs  = [{"query": q["question"], "category": q["category"], "difficulty": q["difficulty"]} for q in gold],
        outputs = [{"answer": q["ground_truth"], "relevant_doc": q["relevant_doc"]} for q in gold],
        dataset_id = dataset.id,
    )

    print(f"Uploaded {len(gold)} examples to dataset '{DATASET_NAME}' (id: {dataset.id})")
    return dataset.id


# ── Predictor — calls your live RAG API ───────────────────────────
def predict(inputs: dict) -> dict:
    """Call the live RAG API and return the answer + sources."""
    query = inputs["query"]
    try:
        import time as _t
        resp = None
        for _attempt in range(5):
            resp = requests.post(
                f"{API_URL}/api/v1/search",
                json={"query": query},
                headers={"Content-Type": "application/json", "X-API-Key": API_KEY},
                timeout=90,
            )
            if resp.status_code != 429:
                break
            _wait = 2 ** _attempt            # 1,2,4,8,16s backoff
            _t.sleep(_wait)
        _t.sleep(0.5)                        # pace between questions
        resp.raise_for_status()
        data = resp.json()
        return {
            "answer":   data.get("answer", ""),
            "sources":  [s.get("clean_name", "") for s in data.get("sources", [])],
            "cached":   data.get("cached", False),
            "blocked":  data.get("blocked", False),
        }
    except Exception as e:
        return {"answer": f"ERROR: {e}", "sources": [], "cached": False, "blocked": False}


# ── Evaluators ────────────────────────────────────────────────────


# ════════════ SENTINEL HANDLING (behavioural tests, no source doc) ════════════
SENTINEL_DOCS = {"PRICING_REDIRECT", "NO_ANSWER"}
_REDIRECT_MARKERS = ("equinix.com/contact", "solutions architect", "custom quote", "contact our team")
_DECLINE_MARKERS  = ("could not find", "couldn't find", "not in", "not available",
                     "no specific information", "cannot find", "don't have", "do not have")

def _sentinel_score(relevant_doc, answer):
    if relevant_doc not in SENTINEL_DOCS:
        return None
    a = (answer or "").lower()
    if relevant_doc == "PRICING_REDIRECT":
        ok = any(m in a for m in _REDIRECT_MARKERS)
        return (1.0 if ok else 0.0, "redirected" if ok else "FAILED: no redirect (price hallucination risk)")
    ok = any(m in a for m in _DECLINE_MARKERS)
    return (1.0 if ok else 0.0, "declined" if ok else "FAILED: did not decline (hallucination risk)")

def _extract(run, example):
    rd = ""
    try: rd = (example.outputs or {}).get("relevant_doc","")
    except Exception:
        try: rd = example.get("relevant_doc","")
        except Exception: pass
    ans = ""
    try: ans = (run.outputs or {}).get("answer","")
    except Exception: pass
    return rd, ans

def faithfulness_evaluator(run, example) -> dict:
    _rd, _ans = _extract(run, example)
    _s = _sentinel_score(_rd, _ans)
    if _s is not None:
        return {'key': 'faithfulness', 'score': _s[0], 'comment': _s[1]}
    """
    Measures: did the answer make up anything not in the ground truth?
    Score 0-1: 1 = fully faithful, 0 = hallucinated.
    """
    answer       = (run.outputs or {}).get("answer", "")
    ground_truth = (example.outputs or {}).get("answer", "")
    question     = (example.inputs or {}).get("query", "")

    if answer.startswith("ERROR:") or not answer:
        return {"key": "faithfulness", "score": 0.0, "comment": "API error or empty answer"}

    resp = OAI.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"""You are an expert evaluator. Score how faithful the actual answer is to the reference answer.

Question: {question}
Reference answer: {ground_truth}
Actual answer: {answer}

Faithfulness means: does the actual answer contain only information that is consistent with the reference?
- Score 1.0: Actual answer is fully consistent with reference, no hallucinations
- Score 0.5: Mostly consistent but adds some unverifiable claims
- Score 0.0: Contains clearly wrong information or major hallucinations

Return ONLY a decimal number between 0 and 1. Nothing else."""
        }],
        max_tokens=5,
        temperature=0,
    )
    try:
        score = float(resp.choices[0].message.content.strip())
        score = max(0.0, min(1.0, score))
    except ValueError:
        score = 0.5

    return {"key": "faithfulness", "score": score}


def answer_relevancy_evaluator(run, example) -> dict:
    _rd, _ans = _extract(run, example)
    _s = _sentinel_score(_rd, _ans)
    if _s is not None:
        return {'key': 'answer_relevancy', 'score': _s[0], 'comment': _s[1]}
    """
    Measures: did the answer actually address the question?
    Score 0-1: 1 = directly answers, 0 = irrelevant or off-topic.
    """
    answer   = (run.outputs or {}).get("answer", "")
    question = (example.inputs or {}).get("query", "")

    if answer.startswith("ERROR:") or "couldn't find" in answer.lower():
        return {"key": "answer_relevancy", "score": 0.0, "comment": "No answer found"}

    resp = OAI.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"""You are an expert evaluator. Score how relevant the answer is to the question.

Question: {question}
Answer: {answer}

Relevancy means: does the answer directly address what was asked?
- Score 1.0: Answer directly and completely addresses the question
- Score 0.5: Answer is related but incomplete or partially off-topic
- Score 0.0: Answer does not address the question at all

Return ONLY a decimal number between 0 and 1. Nothing else."""
        }],
        max_tokens=5,
        temperature=0,
    )
    try:
        score = float(resp.choices[0].message.content.strip())
        score = max(0.0, min(1.0, score))
    except ValueError:
        score = 0.5

    return {"key": "answer_relevancy", "score": score}


def source_retrieval_evaluator(run, example) -> dict:
    _rd, _ans = _extract(run, example)
    _s = _sentinel_score(_rd, _ans)
    if _s is not None:
        return {'key': 'source_retrieval', 'score': _s[0], 'comment': _s[1]}
    """
    Measures: did the system retrieve the correct source document?
    Score 1 if relevant_doc appears in sources, 0 if not.
    """
    sources      = (run.outputs or {}).get("sources", [])
    relevant_doc = (example.outputs or {}).get("relevant_doc", "")

    if not relevant_doc or not sources:
        return {"key": "source_retrieval", "score": 0.0, "comment": "No sources returned"}

    # Check if any source name contains key words from the relevant doc
    relevant_words = set(relevant_doc.lower().split())
    for source in sources:
        source_words = set(source.lower().split())
        overlap = relevant_words & source_words
        if len(overlap) >= 2:
            return {"key": "source_retrieval", "score": 1.0, "comment": f"Found: {source}"}

    return {"key": "source_retrieval", "score": 0.0, "comment": f"Expected: {relevant_doc}, Got: {sources}"}


def not_blocked_evaluator(run, example) -> dict:
    """Checks that legitimate questions are not blocked by guardrails."""
    blocked = (run.outputs or {}).get("blocked", False)
    score   = 0.0 if blocked else 1.0
    return {"key": "not_blocked", "score": score, "comment": "Blocked by guardrail" if blocked else "OK"}


# ── Main ──────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Run RAG evaluation")
    parser.add_argument("--category",   help="Filter by category (ai, financial, product, definition, etc.)")
    parser.add_argument("--difficulty", help="Filter by difficulty (easy, medium, hard)")
    parser.add_argument("--limit",      type=int, help="Limit number of examples")
    parser.add_argument("--set-baseline", action="store_true", help="Freeze current scores as gate baseline")
    parser.add_argument("--upload",     action="store_true", help="Re-upload dataset to LangSmith")
    args = parser.parse_args()

    # Load gold set
    with open(GOLD_FILE) as f:
        gold = json.load(f)

    # Filter
    if args.category:
        gold = [q for q in gold if q["category"] == args.category]
        print(f"Filtered to {len(gold)} questions in category: {args.category}")
    if args.difficulty:
        gold = [q for q in gold if q["difficulty"] == args.difficulty]
        print(f"Filtered to {len(gold)} questions with difficulty: {args.difficulty}")
    if args.limit:
        gold = gold[:args.limit]
        print(f"Limited to {len(gold)} questions")

    print(f"Running evaluation on {len(gold)} questions...")

    # Upload dataset
    upload_dataset(gold, force=args.upload)

    # Run evaluation
    experiment_name = f"rag-eval-{datetime.now().strftime('%Y%m%d-%H%M')}"
    print(f"Experiment: {experiment_name}")

    results = evaluate(
        predict,
        data=DATASET_NAME,
        evaluators=[
            faithfulness_evaluator,
            answer_relevancy_evaluator,
            source_retrieval_evaluator,
            not_blocked_evaluator,
        ],
        experiment_prefix=experiment_name,
        max_concurrency=1,
    )

    # Print summary
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Experiment:       {experiment_name}")
    print(f"Questions tested: {len(gold)}")
    print(f"View in LangSmith: https://smith.langchain.com")
    print()

    scores = {}
    # Authoritative scores come from LangSmith (the live evaluate() return
    # shape is unreliable across client versions). Pull by experiment name.
    import time as _t
    _t.sleep(5)  # let feedback finalise
    try:
        for _run in LS_CLIENT.list_runs(project_name=getattr(results,"experiment_name",None) or experiment_name):
            for _fb in LS_CLIENT.list_feedback(run_ids=[_run.id]):
                if _fb.key and _fb.score is not None:
                    scores.setdefault(_fb.key, []).append(_fb.score)
    except Exception as _e:
        print("⚠ could not pull scores from LangSmith:", _e)

    for metric, vals in sorted(scores.items()):
        avg = sum(vals) / len(vals) if vals else 0
        status = "✅" if avg >= 0.80 else "⚠️" if avg >= 0.60 else "❌"
        print(f"{status} {metric:<25} {avg:.3f}  (n={len(vals)}, target: ≥0.80)")
    print()
    overall = sum(sum(v)/len(v) for v in scores.values()) / len(scores) if scores else 0
    print(f"Overall score: {overall:.3f}")
    import sys as _sys
    _sys.exit(gate_check(scores, update_baseline='--set-baseline' in _sys.argv))
    print("="*50)


# ════════════ BASELINE GATE ════════════
BASELINE_PATH = "eval/golden_baseline.json"
REGRESSION_TOLERANCE = 0.02
ABSOLUTE_FLOOR = 0.80

def _load_baseline():
    import json
    try: return json.load(open(BASELINE_PATH))
    except Exception: return None

def gate_check(scores: dict, update_baseline: bool = False) -> int:
    import json
    avg = {k: sum(v)/len(v) for k, v in scores.items() if v}
    if update_baseline:
        json.dump(avg, open(BASELINE_PATH, "w"), indent=2)
        print(f"\n\U0001F4CC Baseline frozen -> {BASELINE_PATH}")
        for k, v in avg.items(): print(f"   {k}: {v:.3f}")
        return 0
    baseline = _load_baseline()
    print("\n" + "=" * 52 + "\nGATE RESULT\n" + "=" * 52)
    failed = []
    f = avg.get("faithfulness", 0)
    if f < ABSOLUTE_FLOOR: failed.append(f"faithfulness {f:.3f} < floor {ABSOLUTE_FLOOR}")
    if baseline:
        for metric, base_val in baseline.items():
            cur = avg.get(metric, 0); drop = base_val - cur
            mark = "\u274C" if drop > REGRESSION_TOLERANCE else "\u2705"
            print(f"  {mark} {metric:<22} {cur:.3f}  (baseline {base_val:.3f}, d{-drop:+.3f})")
            if drop > REGRESSION_TOLERANCE: failed.append(f"{metric} regressed {base_val:.3f}->{cur:.3f}")
    else:
        print("  no baseline yet -- run with --set-baseline once happy")
        for metric, val in avg.items(): print(f"     {metric:<22} {val:.3f}")
    print("=" * 52)
    if failed:
        print("GATE FAILED:")
        for x in failed: print(f"   - {x}")
        return 1
    print("GATE PASSED")
    return 0


if __name__ == "__main__":
    main()
