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

### Key Observations & Next Steps
- Retrieval quality is heavily dependent on **chunk coherence** (paragraphs vs. raw characters).
- LLMs are eager to please and will "cheat" if the system prompt isn't strictly gated.
- **Parent Document Retrieval** elegantly solves the problem of dense similarity clustering in single-domain knowledge bases.
- Next improvement: Implement `pgvector` for persistent, scalable storage instead of in-memory numpy arrays.