GOLDEN_DATASET = [
    # ── EASY: single-document questions ──────────────────────
    {
        "id": "q001",
        "question": "What is the CAP theorem?",
        "expected_answer": "CAP theorem states a distributed system can only guarantee 2 of 3: Consistency, Availability, Partition tolerance.",
        "relevant_sources": ["cap_theorem", "pacelc_theorem"],
        "category": "consensus",
        "difficulty": "easy",
    },
    {
        "id": "q002",
        "question": "How does a circuit breaker work?",
        "expected_answer": "A circuit breaker monitors failures and stops requests to a failing service, allowing it time to recover.",
        "relevant_sources": ["circuit_breaker"],
        "category": "reliability",
        "difficulty": "easy",
    },
    {
        "id": "q003",
        "question": "What is a bloom filter used for?",
        "expected_answer": "A bloom filter is a probabilistic data structure to test set membership, with possible false positives but no false negatives.",
        "relevant_sources": ["bloom_filters"],
        "category": "data_structures",
        "difficulty": "easy",
    },
    {
        "id": "q004",
        "question": "What is the write-ahead log?",
        "expected_answer": "A write-ahead log records changes before applying them to the database, enabling crash recovery.",
        "relevant_sources": ["write_ahead_log"],
        "category": "storage",
        "difficulty": "easy",
    },
    {
        "id": "q005",
        "question": "What is the gossip protocol?",
        "expected_answer": "Gossip protocol is a peer-to-peer protocol where nodes periodically exchange state with random peers.",
        "relevant_sources": ["gossip_protocol"],
        "category": "protocols",
        "difficulty": "easy",
    },

    # ── MEDIUM: multi-document questions ─────────────────────
    {
        "id": "q006",
        "question": "How does leader election work in distributed systems?",
        "expected_answer": "Leader election lets nodes agree on a single coordinator using algorithms like Raft or Paxos.",
        "relevant_sources": ["leader_election", "consensus_algorithms"],
        "category": "consensus",
        "difficulty": "medium",
    },
    {
        "id": "q007",
        "question": "What are the differences between strong and eventual consistency?",
        "expected_answer": "Strong consistency guarantees reads see the latest write. Eventual consistency allows temporary divergence but guarantees eventual convergence.",
        "relevant_sources": ["consistency_models", "cap_theorem", "pacelc_theorem", "geo_distributed_systems"],
        "category": "consistency",
        "difficulty": "medium",
    },
    {
        "id": "q008",
        "question": "How do distributed transactions work?",
        "expected_answer": "Distributed transactions coordinate writes across multiple nodes using protocols like two-phase commit to ensure atomicity.",
        "relevant_sources": ["distributed_transactions", "two_phase_commit"],
        "category": "transactions",
        "difficulty": "medium",
    },

    # ── HARD: cross-document synthesis ───────────────────────
    {
        "id": "q009",
        "question": "How do CRDTs relate to eventual consistency?",
        "expected_answer": "CRDTs are data structures that merge concurrent updates without conflicts, making them a practical implementation of eventual consistency.",
        "relevant_sources": ["crdts", "consistency_models", "vector_clocks"],
        "category": "consistency",
        "difficulty": "hard",
    },
    {
        "id": "q010",
        "question": "What strategies exist for handling failures in distributed systems?",
        "expected_answer": "Strategies include circuit breakers, retries, timeouts, bulkheads, and the saga pattern.",
        "relevant_sources": ["fault_tolerance", "circuit_breaker", "saga_pattern"],
        "category": "reliability",
        "difficulty": "hard",
    },

    # ── OUT-OF-CORPUS: system should refuse ──────────────────
    {
        "id": "q011",
        "question": "How does React's virtual DOM work?",
        "expected_answer": None,
        "relevant_sources": [],
        "category": "out_of_corpus",
        "difficulty": "easy",
    },
    {
        "id": "q012",
        "question": "What is Redis and how is it used?",
        "expected_answer": None,
        "relevant_sources": [],
        "category": "out_of_corpus",
        "difficulty": "easy",
    },

    # ── EASY: new corpus additions ────────────────────────────
    {
        "id": "q013",
        "question": "How does Kubernetes work?",
        "expected_answer": "Kubernetes is a container orchestration platform that uses a declarative API to manage pods across nodes. Key components include the API server, etcd, scheduler, controller manager, and kubelet.",
        "relevant_sources": ["kubernetes_fundamentals"],
        "category": "infrastructure",
        "difficulty": "easy",
    },
    {
        "id": "q014",
        "question": "What is consistent hashing?",
        "expected_answer": "Consistent hashing maps nodes and keys onto a virtual ring so that only a small fraction of keys need to be remapped when a node is added or removed.",
        "relevant_sources": ["consistent_hashing"],
        "category": "data_structures",
        "difficulty": "easy",
    },
    {
        "id": "q015",
        "question": "What is an inverted index?",
        "expected_answer": "An inverted index maps each unique term to the list of documents containing it, enabling efficient full-text search.",
        "relevant_sources": ["inverted_index"],
        "category": "data_structures",
        "difficulty": "easy",
    },
    {
        "id": "q016",
        "question": "What are SLIs, SLOs, and SLAs?",
        "expected_answer": "SLIs are measurable indicators of system health, SLOs are quantitative targets set against SLIs, and SLAs are contractual commitments to customers based on those SLOs.",
        "relevant_sources": ["slo_sla_sli"],
        "category": "observability",
        "difficulty": "easy",
    },

    # ── MEDIUM: new multi-doc questions ──────────────────────
    {
        "id": "q017",
        "question": "How do LSM trees use SSTables and write-ahead logs?",
        "expected_answer": "LSM trees buffer writes in a memtable, persist them to immutable SSTables on disk, and use a write-ahead log to survive crashes before the memtable is flushed.",
        "relevant_sources": ["lsm_trees_and_sstables", "write_ahead_log"],
        "category": "storage",
        "difficulty": "medium",
    },
    {
        "id": "q018",
        "question": "How does CQRS relate to event sourcing?",
        "expected_answer": "CQRS separates read and write models; event sourcing is often used on the write side to persist state changes as an immutable event log that the read side can project from.",
        "relevant_sources": ["cqrs", "event_sourcing"],
        "category": "design_patterns",
        "difficulty": "medium",
    },

    # ── HARD: new cross-doc synthesis ────────────────────────
    {
        "id": "q019",
        "question": "How does MVCC achieve isolation without locking, and how does that compare to two-phase commit?",
        "expected_answer": "MVCC maintains multiple record versions so readers never block writers. Two-phase commit coordinates atomic commits across nodes but requires locks during the prepare phase, making it a complementary protocol rather than an alternative.",
        "relevant_sources": ["mvcc_multiversion_concurrency_control", "two_phase_commit", "distributed_transactions"],
        "category": "transactions",
        "difficulty": "hard",
    },
    {
        "id": "q020",
        "question": "What is the difference between TLS and mTLS, and when would you use mTLS?",
        "expected_answer": "TLS authenticates only the server; mTLS requires both client and server to present certificates, making it suitable for service-to-service communication in zero-trust or service-mesh environments.",
        "relevant_sources": ["tls_and_mtls", "service_mesh", "zero_trust_security"],
        "category": "security",
        "difficulty": "hard",
    },
]
