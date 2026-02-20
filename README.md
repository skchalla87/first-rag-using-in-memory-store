## 📊 Test Results & Analysis

### Test Session: 2026-02-20

#### Query 1: "What is the CAP theorem?"
**Retrieval:**
- Top match: `cap_theorem` (similarity: 0.6313)
- 2nd: `clocks_and_time` (similarity: 0.4878)
- 3rd: `fault_tolerance` (similarity: 0.4847)

**Answer Quality:** ✅
- Accurately explained all three guarantees (Consistency, Availability, Partition Tolerance) and the core tradeoff.

**Learning:** The system correctly retrieved the most relevant document with the highest similarity. The two secondary matches (clocks, fault tolerance) are topically adjacent in distributed systems, showing the embeddings capture semantic proximity.

---

#### Query 2: "What's the difference between synchronous and asynchronous replication?"
**Retrieval:**
- Top match: `replication` (similarity: 0.6824)
- 2nd: `clocks_and_time` (similarity: 0.5697)
- 3rd: `consistency_models` (similarity: 0.5285)

**Answer Quality:** ✅
- Correctly compared both types: synchronous waits for acknowledgment (strong consistency, higher latency) vs asynchronous commits locally first (better performance, temporary inconsistency).

**Learning:** Highest similarity score across all tests (0.6824), suggesting query-document alignment is strongest when the question directly names a concept covered by a document. Secondary matches (clocks, consistency) are meaningfully related topics.

---

#### Query 3: "How does Kubernetes work?" (Out-of-domain test)
**Retrieval:**
- Top match: `clocks_and_time` (similarity: 0.4942)
- 2nd: `consistency_models` (similarity: 0.4423)
- 3rd: `consensus_algorithms` (similarity: 0.4365)

**Answer Quality:** ✅
- The model correctly stated that the answer is not in the provided context before offering general knowledge from its training data.

**Learning:** All similarity scores dropped below 0.50 — noticeably lower than in-domain queries (0.63–0.68). The system doesn't hallucinate from context; instead, the LLM transparently acknowledges the gap. This confirms the retrieval + generation pipeline handles unknown topics gracefully.

---

### Key Observations
- Retrieval quality: Good — the most relevant document consistently ranks #1 for in-domain queries.
- Answer accuracy: Consistently accurate — all three answers were correct and well-structured.
- Similarity threshold: In-domain top scores range 0.63–0.68; out-of-domain stays below 0.50, providing a natural threshold signal.
- Next improvements: 
    - Could add a similarity threshold cutoff (e.g., < 0.50) to skip retrieval and let the LLM answer from its own knowledge directly.
    - Add pgvector?