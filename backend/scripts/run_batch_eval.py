"""
Run batch RAGAS evaluation against the eval dataset.
Usage: python -m backend.scripts.run_batch_eval
"""
import sys, os, json, requests
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv("backend/.env")

CHAT_URL  = "http://localhost:8000/api/chat/"
EVAL_URL  = "http://localhost:8000/api/eval/"
DATASET   = "backend/datasets/eval_dataset.json"

def run_batch():
    with open(DATASET) as f:
        dataset = json.load(f)

    results = []
    for i, item in enumerate(dataset):
        print(f"\n[Batch] Running {i+1}/{len(dataset)}: {item['question'][:60]}...")

        # step 1: get answer from Athena
        chat_resp = requests.post(CHAT_URL, json={
            "query": item["question"],
            "top_k": 5
        })
        chat_data = chat_resp.json()

        answer   = chat_data.get("answer", "")
        chunks   = [c["text"] for c in chat_data.get("vector_chunks", [])] if "vector_chunks" in chat_data else []
        model    = chat_data.get("model_used", "")

        chunks = [c["text"] for c in chat_data.get("vector_chunks", [])]

# remove the fallback — if chunks empty, skip eval for that question
        if not chunks:
            print(f"[Batch] ⚠️ No chunks returned — skipping eval for this question")
            continue

        # step 2: evaluate with RAGAS
        eval_resp = requests.post(EVAL_URL, json={
            "question":     item["question"],
            "answer":       answer,
            "contexts":     chunks,
            "ground_truth": item.get("ground_truth", ""),
            "model_used":   model
        })
        eval_data = eval_resp.json()

        result = {
            "question":          item["question"],
            "verdict":           eval_data.get("verdict"),
            "overall_score":     eval_data.get("overall_score"),
            "faithfulness":      eval_data.get("faithfulness"),
            "answer_relevancy":  eval_data.get("answer_relevancy"),
            "context_precision": eval_data.get("context_precision"),
        }
        results.append(result)

        print(f"[Batch]  Score: {result['overall_score']} | Verdict: {result['verdict']}")

    # summary
    print("\n" + "="*60)
    print("BATCH EVALUATION SUMMARY")
    print("="*60)
    scores = [r["overall_score"] for r in results if r["overall_score"] is not None]
    print(f"Total evaluated:  {len(results)}")
    print(f"Average score:    {round(sum(scores)/len(scores), 4) if scores else 'N/A'}")
    print(f"PASS  (≥0.75):   {sum(1 for r in results if r.get('overall_score') and r['overall_score'] >= 0.75)}")
    print(f"REVIEW (0.5-0.75): {sum(1 for r in results if r.get('overall_score') and 0.5 <= r['overall_score'] < 0.75)}")
    print(f"FAIL  (<0.5):    {sum(1 for r in results if r.get('overall_score') and r['overall_score'] < 0.5)}")
    print("="*60)

    for r in results:
        print(f"{r['verdict']:8} | {r['overall_score']} | {r['question'][:55]}")

if __name__ == "__main__":
    run_batch()