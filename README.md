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

### Known Limitations
- BM25 index rebuilds full corpus on every add_documents() call
- Acceptable for current corpus size (~21 docs)
- V2 (pgvector) will address this — BM25 index can be maintained 
  incrementally or replaced with PostgreSQL full-text search (tsvector)
  combined with pgvector for true hybrid search at scale

### Next Steps
- Replace in-memory numpy store with a **persistent vector database** to avoid re-embedding on every run.