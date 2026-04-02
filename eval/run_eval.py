"""
RAG Evaluation Runner.
Usage: python eval/run_eval.py
"""

import statistics
import json
import sys
import statistics
from datetime import datetime
from pathlib import Path

# Add parent directory to path so we can import RAGSystem
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from RAGSystem import RAGSystem
from golden_dataset import GOLDEN_DATASET
from metrics import (
    context_precision,
    context_recall,
    hit_rate,
    mrr,
    answer_faithfulness,
    refusal_correctness,
)

def run_eval(top_k: int = 3) -> dict:
    
    print("initializing RAG system...")
    myRAG = RAGSystem()
    
    results = []
    for item in GOLDEN_DATASET:
        
        print(f"\n[{item['id']}]: {item['question']}")
        
        # step 1 retrieve chunks
        retrieved = myRAG.retrieve(item['question'])
        
        # step 2: extract sources and chunk text from retrieved results
        retrieved_sources = [meta["source"] for _, _, meta in retrieved[:top_k]]
        retrieved_texts   = [text for text, _, _ in retrieved[:top_k]]
        
        # step 3: Generate answer
        answer = myRAG.query(item['question'])
        
        # step 4: compute metrics
        precision = context_precision(retrieved_sources, item['relevant_sources'])
        recall = context_recall(retrieved_sources, item['relevant_sources'])
        hit = hit_rate(retrieved_sources, item['relevant_sources'])
        mean_reciprocal_rank = mrr(retrieved_sources, item['relevant_sources'])
        faithfulness = answer_faithfulness(answer, retrieved_texts)
        refusal = refusal_correctness(answer, item['expected_answer'])
        
        print(f" Sources Retrieved: {retrieved_sources}")
        print(f" Answer: {answer}")
        print(f" Precision: {precision:.2f}")
        print(f" Recall: {recall:.2f}")
        print(f" Hit Rate: {hit:.2f}")
        print(f" MRR: {mean_reciprocal_rank:.2f}")
        print(f" Faithfulness: {faithfulness:.2f}")
        print(f" Refusal: {refusal:.2f}")
        
        results.append({
            "id":               item["id"],
            "question":         item["question"],
            "category":         item["category"],
            "difficulty":       item["difficulty"],
            "retrieved_sources": retrieved_sources,
            "relevant_sources": item["relevant_sources"],
            "answer":           answer,
            "precision":        precision,
            "recall":           recall,
            "hit_rate":         hit,
            "mrr":              mean_reciprocal_rank,
            "faithfulness":     faithfulness,
            "refusal_correct":  refusal,
        })

    summary = summarize(results)
    save_results(results, summary)
    print_summary(summary)
    return summary

def summarize(results: list[dict]) -> dict:
    metrics = ["precision", "recall", "hit_rate", "mrr", "faithfulness", "refusal_correct"]
    summary = {}

    # Overall averages
    for m in metrics:
        values = [r[m] for r in results]
        summary[f"avg_{m}"] = round(statistics.mean(values), 3)

    # Breakdown by difficulty
    for difficulty in ["easy", "medium", "hard"]:
        subset = [r for r in results if r["difficulty"] == difficulty]
        if subset:
            summary[f"avg_mrr_{difficulty}"] = round(
                statistics.mean(r["mrr"] for r in subset), 3
            )

    summary["total_questions"] = len(results)
    return summary


def save_results(results: list[dict], summary: dict):
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    output_path = output_dir / f"eval_{timestamp}.json"

    with open(output_path, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)

    print(f"\nResults saved to: {output_path}")


def print_summary(summary: dict):
    print("\n" + "=" * 45)
    print("EVALUATION SUMMARY")
    print("=" * 45)
    print(f"  Questions evaluated : {summary['total_questions']}")
    print(f"  Avg Precision       : {summary['avg_precision']:.3f}")
    print(f"  Avg Recall          : {summary['avg_recall']:.3f}")
    print(f"  Avg Hit Rate        : {summary['avg_hit_rate']:.3f}")
    print(f"  Avg MRR             : {summary['avg_mrr']:.3f}")
    print(f"  Avg Faithfulness    : {summary['avg_faithfulness']:.3f}")
    print(f"  Refusal Accuracy    : {summary['avg_refusal_correct']:.3f}")
    print("-" * 45)
    print(f"  MRR easy   : {summary.get('avg_mrr_easy', 'n/a')}")
    print(f"  MRR medium : {summary.get('avg_mrr_medium', 'n/a')}")
    print(f"  MRR hard   : {summary.get('avg_mrr_hard', 'n/a')}")
    print("=" * 45)


if __name__ == "__main__":
    run_eval()