## 📊 RAG System - Test Results & Learnings Log

---

### Test Session: 2026-02-20 to 2026-02-21

#### 🐛 The "False Positive" RAG Discovery
Initially, the system *appeared* to work perfectly and answered questions like "What is the CAP theorem?" correctly. However, digging into the debug logs revealed four major bugs that created a **false positive RAG** (where the LLM was answering from its own training data, not the documents).

**Learning 1: The "Filename Chunking" Bug**
- **Issue:** The code iterated over `documents.keys()` instead of `documents.values()`. We were chunking and embedding the filenames (e.g., `"cap_theorem"`), not the file content!
- **Why it seemed to work:** The query semantically matched the string `"cap_theorem"`, which was fetched as context. The LLM saw the hint `"cap_theorem"` and answered from its own pre-trained knowledge.
- **Fix:** Switched to iterating over `documents.values()` (and eventually `documents.items()` to keep track of source filenames).

**Learning 2: The Strict Prompting Bug**
- **Issue:** Even after fixing the chunking, the LLM still answered correctly for queries where the retrieved context was completely unrelated (like data partitioning chunks for a CAP theorem question).
- **Fix:** Added a strict prompt rule: `"Answer the question using ONLY the context provided below. If the context does not contain the answer, respond with: 'I don't have information about this'"`

**Learning 3: The Chunking Strategy Problem**
- **Issue:** The original `DocumentChunker` split text strictly every 500 characters. This cut sentences and words in half, destroying semantic meaning for the embedding model.
- **Fix:** Built a `ParagraphChunker` that splits on blank lines (`\n\n`), preserving semantic boundaries so the embedding model encodes complete thoughts.

**Learning 4: The Parent Document Retrieval Fix**
- **Issue:** All 13 documents were heavily focused on distributed systems. As a result, all chunks scored very similarly (0.40–0.51). The core definition of CAP theorem ranked #22, while a chunk about "CAP theorem misunderstandings" ranked #4. If we only pulled top-k, the LLM got incomplete fragments and gave partial answers.
- **Fix:** Implemented **Parent Document Retrieval**. Now, if *any* chunk natively hits the top-K list, we automatically pull *all* sibling chunks from that source document. This ensures the LLM gets the entire surrounding context.

---

#### Post-Fix Verification Queries

#### Query 1: "What is the CAP theorem?"
**Retrieval:**
- Top match hits included: `consistency_models`, `distributed_transactions`, `cap_theorem`
- **Parent Retrieval Expansion:** Pulled in all 3 chunks of `cap_theorem` because chunk 1 hit the top 5 (rank 4).

**Answer Quality:** ✅ 
- Accurately quoted the definition directly from the source chunk: *The CAP theorem states that in a distributed system, you can have at most two of three properties...*

**Learning:** Semantic search across highly homogeneous documents results in tightly clustered similarity scores. Expanding retrieved chunks to include their full parent document is critical for preventing context fragmentation.

#### Query 2: "How does Kubernetes work?" (Out-of-domain test)
**Retrieval:**
- Expected: Low similarity scores across the board since Kubernetes isn't in the docs.

**Answer Quality:** ✅
- The model correctly replied: *"I don't have information about this in the provided documents."*

**Learning:** The strict prompt completely eliminated the LLM's tendency to hallucinate answers from its pre-training data.

---

### Key Observations (Session 1)
- Retrieval quality is heavily dependent on **chunk coherence** (paragraphs vs. raw characters).
- LLMs are eager to please and will "cheat" if the system prompt isn't strictly gated.
- **Parent Document Retrieval** elegantly solves the problem of dense similarity clustering in single-domain knowledge bases.

---

### Test Session: 2026-03-28

#### Improvement 1: Hybrid Retrieval — BM25 + Cosine Similarity

**Problem:** Pure semantic (cosine) search compresses query and chunk into independent vectors before comparing them. It can miss chunks that use the exact same terminology as the query, and can over-rank chunks that are semantically related but don't directly answer the question.

**Fix:** Added BM25 keyword search blended with cosine similarity at a **70% semantic / 30% keyword** split.

- BM25 index is built over all chunks at index time inside `InMemoryVectorStore`
- At query time, both scores are normalized to [0,1] and combined: `0.7 * cosine_norm + 0.3 * bm25_norm`
- The `query_text` parameter is passed through `RAGSystem.retrieve()` into `vector_store.search()`

**Observed Result:** Scores increased (e.g. 0.72 → 0.89 for top chunk on "what is leader election?") and the retrieved set shifted — chunks with direct keyword matches that were previously ranked lower surfaced into top-k.

**Learning:** BM25 and semantic search cover each other's blind spots. Semantic handles paraphrasing; BM25 handles exact terminology. Neither alone is optimal for mixed natural-language + technical queries.

---

#### Improvement 2: Cross-Encoder Re-ranking

**Problem:** Both cosine and BM25 score query and chunk **independently** — the model never sees both together. A chunk can score highly because it shares vocabulary or semantic space with the query without actually answering it.

**Fix:** After hybrid retrieval, a **cross-encoder** re-scores the top-20 candidates by reading query and chunk together as a single input, then reorders before passing top-k to the LLM.

- Model used: `cross-encoder/ms-marco-MiniLM-L-6-v2` (lightweight, runs locally)
- Pipeline: hybrid search → top-20 candidates → cross-encoder re-rank → top-3 to LLM
- Cross-encoder scores are raw logits (not [0,1]), used only for ranking not thresholding

**Learning:** Bi-encoders are fast but compare summaries of two texts. Cross-encoders are slower but understand the relationship between query and chunk — critical for distinguishing "related to topic" from "answers the question". Use bi-encoder for cheap candidate retrieval, cross-encoder for precise re-scoring of the shortlist.

**Why not cross-encode the full corpus?** Cross-encoders run inference on every (query, chunk) pair at query time — O(N) model forward passes. On thousands of chunks this is too slow. Restricting it to top-20 candidates keeps latency acceptable.

---

#### Improvement 3: Chunk Size Tuning

- Reduced default chunk size from `500 → 250` characters, overlap `30 → 50`
- Smaller chunks are more focused — less noise around the relevant sentence
- Higher overlap reduces the chance of a key sentence being split across chunk boundaries

---

#### Expanded Knowledge Base

Added 8 new distributed systems documents to `docs/`:
`service_discovery`, `distributed_tracing`, `message_queues_and_event_streaming`, `saga_pattern`, `rate_limiting_and_backpressure`, `gossip_protocol`, `bloom_filters`, `vector_clocks`

Total: 13 → 21 documents.

---

### Key Observations (Session 2)
- LLM output is non-deterministic — the same retrieval can produce different answer quality across runs due to temperature sampling. Use `temperature=0` when isolating whether a code change actually improved retrieval.
- Debug output and actual retrieval must use identical search parameters — a mismatch (e.g. missing `query_text` in the debug call) silently shows stale scores while retrieval is already improved.
- BM25 must be rebuilt over the **full** corpus on every `add_documents` call — IDF values depend on the entire document set.
- Re-ranking adds a model load cost at startup and inference cost per query, but only over the small candidate set — acceptable tradeoff for significantly better top-k precision.

### Known Limitations (Session 2)
- BM25 index rebuilds full corpus on every add_documents() call
- Acceptable for current corpus size (~21 docs)
- V2 (pgvector) will address this — BM25 index can be maintained
  incrementally or replaced with PostgreSQL full-text search (tsvector)
  combined with pgvector for true hybrid search at scale

---

### Test Session: 2026-03-31 — V2: Migration to PostgreSQL + pgvector

#### What Changed in V2

- **Persistent vector store:** Replaced `InMemoryVectorStore` (numpy + BM25) with `PgVectorStore` backed by PostgreSQL + pgvector extension.
- **Hybrid search in SQL:** BM25 is gone. Hybrid retrieval is now a single SQL query combining `pgvector` cosine similarity (`<=>`) with PostgreSQL native full-text search (`tsvector` / `ts_rank`), blended at 70% semantic / 30% keyword.
- **Separate ingestion script:** `load_documents.py` runs once to chunk, embed, and insert all docs into PostgreSQL. `RAGSystem` now connects to the DB at startup and queries immediately — no re-embedding on every run.
- **Parent retrieval via DB:** `get_chunks_by_source()` fetches all sibling chunks directly from PostgreSQL instead of filtering the in-memory candidate pool (which only held top-20).
- **Expanded corpus:** 21 → 33 documents. Added: `load_balancing`, `circuit_breaker`, `crdts`, `distributed_locking`, `write_ahead_log`, `lsm_trees_and_sstables`, `service_mesh`, `distributed_caching`, `backpressure`, `two_phase_commit`, `mapreduce`, `sidecar_pattern`.

---

#### Query 1: "what is cap theorem?"

**Debug ranking (top 5):**
```
1. [cap_theorem] chunk 0       score:  6.2224
2. [cap_theorem] chunk 5       score: -1.5494
3. [backpressure] chunk 16     score: -7.8531
4. [two_phase_commit] chunk 0  score: -9.6125
5. [distributed_locking] chunk 6 score: -10.3587
```

**Answer Quality:** ✅ Correct — answered directly from `cap_theorem` source.

**Learning:** Cross-encoder scores are raw logits (not bounded [0,1] like the old BM25+cosine scores). The gap between rank 1 (+6.22) and rank 2 (-1.55) is a strong signal — the model is highly confident the first chunk directly answers the query. `backpressure` appearing at rank 3 is a hybrid search false positive (keyword overlap), but the cross-encoder already assigned it -7.85, so it never influences the LLM context.

---

#### Query 2: "How does Kubernetes work?" (out-of-domain)

**Debug ranking (top 3):**
```
1. [sidecar_pattern] chunk 10   score: 4.6343
2. [design_patterns] chunk 2    score: 2.9732
3. [fault_tolerance] chunk 17   score: 0.9359
```

**Answer Quality:** ✅ Correctly refused — *"I don't have information about this in the provided documents."*

**Learning:** Kubernetes is mentioned in `sidecar_pattern.txt` (Kubernetes pod implementation) and `design_patterns.txt` (etcd reference), so retrieval scores are *positive* — not a failure. The cross-encoder correctly judges these chunks as *related to Kubernetes* but not *explaining how Kubernetes works*. The strict prompt then correctly blocks the LLM from hallucinating. This is a better result than V1 where the model partially answered from pre-training knowledge.

---

#### Query 3: "What is leader election?"

**Debug ranking (top 5):**
```
1. [leader_election] chunk 0       score:  0.8822
2. [leader_election] chunk 1       score:  0.8701
3. [leader_election] chunk 4       score: -0.3418
4. [consensus] chunk 11            score: -0.4851
5. [consensus_algorithms] chunk 6  score: -1.1891
```

**Answer Quality:** ✅ Accurate — retrieved correct source, answered concisely.

**Learning:** Even with accidental query prefix ("Enter your question: What is leader election?"), the cross-encoder correctly ranked `leader_election` chunks at the top. Noise in the query string did not break retrieval.

---

#### Query 4: "gossip protocol failure detection"

**Debug ranking (top 5):**
```
1. [gossip_protocol] chunk 10  score:  2.6124
2. [gossip_protocol] chunk 0   score:  2.3768
3. [gossip_protocol] chunk 1   score:  2.0572
4. [gossip_protocol] chunk 14  score:  0.7313
5. [gossip_protocol] chunk 11  score: -1.5675
```

**Answer Quality:** ✅ Precise — retrieved the exact failure detection section and answered correctly with source citation.

**Learning:** Multi-term technical queries ("gossip protocol failure detection") work well — both the semantic and keyword components align on the same document set. Top 4 hits all from `gossip_protocol`, with `fault_tolerance` mixing in at rank 6, which is semantically appropriate.

---

### Key Observations (Session 3 — V2)

- **No startup cost for embeddings:** The biggest practical improvement. V1 re-embedded all 21 docs on every run (~minutes). V2 connects to PostgreSQL and is query-ready immediately.
- **Cross-encoder score scale changed:** V1 scores were normalized [0,1] (BM25+cosine blend). V2 scores are raw cross-encoder logits, typically ranging -12 to +7. Confidence thresholds in `_confidence_level()` were designed for this scale (≥0 = Very High, etc.) and hold correctly.
- **Large score gaps are meaningful:** A rank-1 score of +6 with rank-2 at -1.5 means the cross-encoder is highly confident. Tight clustering (all scores around -2 to -3) means none of the candidates strongly answer the query.
- **Hybrid SQL search replaces BM25:** No more index rebuild on `add_documents`. PostgreSQL maintains `tsvector` automatically via a trigger or generated column. Scales to millions of rows without code changes.
- **Parent retrieval correctness:** V1 expanded parents by filtering the top-20 candidate pool — missing sibling chunks ranked below top-20. V2 calls `get_chunks_by_source()` directly against the DB, guaranteeing all chunks of a matched document are returned regardless of their individual scores.


### Session 4: 2026-04-02 — Evaluation Layer + Observability

#### What Was Built
- **eval/golden_dataset.py** — 12 curated Q&A pairs across easy/medium/hard/out-of-corpus
- **eval/metrics.py** — precision, recall, hit rate, MRR, faithfulness, refusal accuracy
- **eval/run_eval.py** — full eval runner with timestamped JSON result snapshots
- **eval/analyze_logs.py** — CLI log analyzer (latency, refusal rate, low-confidence queries)
- **RAGSystem.py** — added structured JSONL query logging + Arize Phoenix traces

#### Precision Problem Investigation
- Baseline precision was 0.575
- Identified `max_chunk_size=1000` was too large — chunks spanning multiple topics added noise
- Reduced to 500 → precision improved to 0.604
- Reduced top_k from 5 → 3 → precision improved to 0.708
- Key learning: chunk size directly affects embedding focus — smaller chunks embed one concept, larger chunks embed multiple, increasing false matches

#### Eval Results
| Metric      | Before | After |
|-------------|--------|-------|
| Precision   | 0.575  | 0.708 |
| Recall      | 0.819  | 0.806 |
| MRR         | 0.854  | 0.917 |
| Faithfulness| 0.480  | 0.594 |
| MRR medium  | 0.750  | 1.000 |

#### Key Learnings
- Measure retrieval separately from generation — retrieval bugs always cascade into generation failures
- parent_retrieval expands the result set for the LLM but inflates source count — don't measure precision after parent expansion
- Cross-encoder score gap between rank 1 and rank 3 is a proxy for retrieval confidence — tight clustering = ambiguous query
- Generation is the bottleneck (5.5s) not retrieval (0.9s) — optimize LLM choice before optimizing retrieval pipeline
- MRR hard = 0.5 is a known limitation — cross-doc synthesis requires query expansion or multi-hop retrieval, not tunable with current architecture
