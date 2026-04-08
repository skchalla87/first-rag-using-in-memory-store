"""
Generate docs using a local Ollama model.
Usage:
    python generate_docs.py                    # generate all missing topics
    python generate_docs.py --model gemma2:27b # use a different model
    python generate_docs.py --dry-run          # print topics without generating
"""

import argparse
import json
import time
from pathlib import Path

import requests

DOCS_DIR = Path(__file__).parent / "docs"
OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "qwen2.5:14b"

# Topics to generate — edit freely
TOPICS = [
    # Algorithms & Data Structures
    "b_trees",
    "skip_lists",
    "merkle_trees",
    "consistent_hashing",
    "hyperloglog",
    "count_min_sketch",
    "locality_sensitive_hashing",
    "inverted_index",
    # Storage & Databases
    "columnar_storage",
    "time_series_databases",
    "graph_databases",
    "search_engine_internals",
    "storage_engines_comparison",
    "compaction_strategies",
    # Networking & APIs
    "grpc_and_protocol_buffers",
    "http2_and_http3",
    "websockets_and_sse",
    "api_gateway_patterns",
    "idempotency",
    # System Design Patterns
    "cqrs",
    "event_sourcing",
    "bulkhead_pattern",
    "retry_and_timeout_patterns",
    "distributed_job_scheduling",
    "work_stealing",
    "actor_model",
    "outbox_pattern",
    # Observability
    "metrics_and_monitoring",
    "structured_logging",
    "alerting_and_on_call",
    "slo_sla_sli",
    "chaos_engineering",
    "profiling_and_flame_graphs",
    # Security
    "oauth2_and_jwt",
    "zero_trust_security",
    "secrets_management",
    "tls_and_mtls",
    # AI / ML Systems
    "vector_databases",
    "approximate_nearest_neighbors",
    "embedding_models",
    "rag_architecture",
    "llm_inference_serving",
    "feature_stores",
    "model_versioning_and_deployment",
    # Infrastructure
    "kubernetes_fundamentals",
    "container_networking",
    "infrastructure_as_code",
    "blue_green_and_canary_deployments",
    "gitops",
    # Distributed Systems — Consistency & Correctness
    "pacelc_theorem",
    "linearizability_vs_serializability",
    "mvcc_multiversion_concurrency_control",
    "exactly_once_semantics",
    "conflict_resolution_strategies",
    # Distributed Systems — Failure & Recovery
    "byzantine_fault_tolerance",
    "split_brain_and_fencing_tokens",
    "cascading_failures",
    "phi_accrual_failure_detector",
    "distributed_snapshots_chandy_lamport",
    # Distributed Systems — Replication & Sync
    "chain_replication",
    "anti_entropy_and_read_repair",
    "hinted_handoff",
    "three_phase_commit",
    "change_data_capture",
    # Distributed Systems — Coordination & Topology
    "zookeeper_and_etcd_patterns",
    "distributed_hash_tables_chord",
    "cell_based_architecture",
    "geo_distributed_systems",
    # Distributed Systems — Performance
    "tail_latency_and_hedged_requests",
    "thundering_herd_and_cache_stampede",
    # Distributed Systems — Processing
    "stream_processing_windowing_watermarks",
    "distributed_file_systems",
]

PROMPT_TEMPLATE = """You are a senior distributed systems engineer writing internal technical documentation.

Write a detailed technical doc on: **{topic}**

Structure it as:
1. What it is (2-3 sentences, no fluff)
2. Core concepts / how it works
3. When to use it (use cases)
4. Trade-offs and limitations
5. Key takeaways (3-5 bullet points)

Write at a senior engineer level. Be specific, not generic. Aim for 600-800 words.
Do not add a title — start directly with the content.
"""


def topic_to_display(topic: str) -> str:
    return topic.replace("_", " ").title()


def existing_topics() -> set[str]:
    return {p.stem for p in DOCS_DIR.glob("*.txt")}


def generate(topic: str, model: str) -> str:
    prompt = PROMPT_TEMPLATE.format(topic=topic_to_display(topic))
    resp = requests.post(
        OLLAMA_URL,
        json={"model": model, "prompt": prompt, "stream": True},
        stream=True,
        timeout=60,  # 60s to receive first token; streaming keeps connection alive after that
    )
    resp.raise_for_status()
    chunks = []
    for line in resp.iter_lines():
        if not line:
            continue
        data = json.loads(line)
        chunks.append(data.get("response", ""))
        if data.get("done"):
            break
    return "".join(chunks).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--topic", help="Generate a single specific topic")
    parser.add_argument("--pause", type=int, default=15, help="Seconds to pause between topics (default: 15)")
    args = parser.parse_args()

    DOCS_DIR.mkdir(exist_ok=True)
    existing = existing_topics()

    if args.topic:
        targets = [args.topic]
    else:
        targets = [t for t in TOPICS if t not in existing]

    if not targets:
        print("All topics already generated.")
        return

    print(f"Model : {args.model}")
    print(f"Topics: {len(targets)} to generate")
    print(f"Output: {DOCS_DIR}\n")

    if args.dry_run:
        for t in targets:
            print(f"  {t}")
        return

    for i, topic in enumerate(targets, 1):
        display = topic_to_display(topic)
        print(f"[{i}/{len(targets)}] {display} ...", end=" ", flush=True)
        start = time.time()
        try:
            content = generate(topic, args.model)
            out = DOCS_DIR / f"{topic}.txt"
            out.write_text(content, encoding="utf-8")
            elapsed = time.time() - start
            words = len(content.split())
            print(f"{words} words in {elapsed:.1f}s")
        except Exception as e:
            print(f"FAILED — {e}")

        if i < len(targets):
            time.sleep(args.pause)

    print("\nDone.")


if __name__ == "__main__":
    main()
