"""
Analyze query logs from logs/queries.jsonl
Usage: python eval/analyze_logs.py
"""

import json
from pathlib import Path
from collections import Counter


def load_logs(path: str = "logs/queries.jsonl") -> list[dict]:
    log_file = Path(path)
    if not log_file.exists():
        print(f"No log file found at {path}. Ask some questions first.")
        return []

    logs = []
    with open(log_file) as f:
        for line in f:
            line = line.strip()
            if line:
                logs.append(json.loads(line))
    return logs


def analyze(logs: list[dict]):
    if not logs:
        return

    total = len(logs)
    refused = [l for l in logs if l["refused"]]
    answered = [l for l in logs if not l["refused"]]

    print("=" * 50)
    print("QUERY LOG ANALYSIS")
    print("=" * 50)

    # ── Volume ────────────────────────────────────────
    print(f"\nTotal queries    : {total}")
    print(f"Answered         : {len(answered)}  ({len(answered)/total:.0%})")
    print(f"Refused          : {len(refused)}   ({len(refused)/total:.0%})")

    # ── Latency ───────────────────────────────────────
    retrieval_times  = [l["retrieval_ms"] for l in logs]
    generation_times = [l["generation_ms"] for l in logs]
    total_times      = [l["total_ms"] for l in logs]

    print(f"\n── Latency (ms) ──────────────────────────────")
    print(f"  Retrieval  avg: {avg(retrieval_times):>6.0f}   max: {max(retrieval_times)}")
    print(f"  Generation avg: {avg(generation_times):>6.0f}   max: {max(generation_times)}")
    print(f"  Total      avg: {avg(total_times):>6.0f}   max: {max(total_times)}")

    # ── Score distribution ────────────────────────────
    all_top_scores = [
        l["top_scores"][0]
        for l in logs
        if l["top_scores"] and l["top_scores"][0] != float("-inf")
    ]
    if all_top_scores:
        print(f"\n── Top Retrieval Scores ──────────────────────")
        print(f"  avg : {avg(all_top_scores):.3f}")
        print(f"  min : {min(all_top_scores):.3f}")
        print(f"  max : {max(all_top_scores):.3f}")

        low_confidence = [
            l for l in logs
            if l["top_scores"] and l["top_scores"][0] < -5
        ]
        if low_confidence:
            print(f"\n  Low confidence queries (top score < -5):")
            for l in low_confidence:
                print(f"    [{l['top_scores'][0]:.2f}] {l['question']}")

    # ── Most retrieved sources ────────────────────────
    all_sources = [s for l in logs for s in l["top_sources"]]
    if all_sources:
        print(f"\n── Most Retrieved Sources (top 5) ────────────")
        for source, count in Counter(all_sources).most_common(5):
            print(f"  {count:>3}x  {source}")

    # ── Refused questions ─────────────────────────────
    if refused:
        print(f"\n── Refused Questions ─────────────────────────")
        for l in refused:
            print(f"  - {l['question']}")

    print("\n" + "=" * 50)


def avg(values: list) -> float:
    return sum(values) / len(values) if values else 0.0


if __name__ == "__main__":
    logs = load_logs()
    analyze(logs)
