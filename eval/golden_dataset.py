GOLDEN_DATASET = [
    # ── EASY: single-document questions ──────────────────────
    {
        "id": "q001",
        "question": "What is the CAP theorem?",
        "expected_answer": "CAP theorem states a distributed system can only guarantee 2 of 3: Consistency, Availability, Partition tolerance.",
        "relevant_sources": ["cap_theorem"],
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
        "relevant_sources": ["consistency_models", "cap_theorem"],
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
        "question": "How does Kubernetes work?",
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
]
