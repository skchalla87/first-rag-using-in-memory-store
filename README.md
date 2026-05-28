## RAG System — Distributed Systems Knowledge Base

A learning project building a production-grade RAG pipeline over 105 distributed systems documents. Built iteratively across sessions — each session documents what broke, what was fixed, and why. Read SESSIONS.md for the full learning log.

---

### What This Project Teaches

This isn't just "call an LLM with some docs". Each layer of the pipeline was designed, broken, debugged, and refined to address a real failure. The key lessons:

- Why character chunking destroys embedding quality (and what to use instead)
- Why cosine similarity alone fails on homogeneous corpora
- Why BM25 + semantic is better than either alone
- Why bi-encoders need a cross-encoder on top
- Why parent-document retrieval matters when chunks cluster tightly
- Why LLMs will hallucinate if your system prompt isn't strict
- Why eval must measure retrieval and generation separately

---

### Architecture

```
docs/*.txt
    │
    ▼
ParagraphChunker          ← splits on blank lines, merges up to 500 chars
    │                       (NOT character windows — see Design Decisions)
    ▼
EmbeddingGenerator        ← mxbai-embed-large via Ollama (1024-dim vectors)
    │
    ▼
PostgreSQL + pgvector     ← persistent store; chunks never re-embedded on restart
    │   (source, chunk_index, content, embedding, content_tsv)
    │
    │   ┌─────────────────── QUERY TIME ───────────────────────┐
    ▼   ▼
Hybrid SQL Search         ← 70% cosine (pgvector) + 30% keyword (tsvector)
    │                       returns top-20 candidates
    ▼
Cross-Encoder Re-ranker   ← reads (query, chunk) together, assigns raw logits
    │   cross-encoder/ms-marco-MiniLM-L-6-v2
    │   top-20 → re-ranked → top-3 to LLM
    ▼
LLM (llama3.1 via Ollama) ← strict prompt: answer ONLY from context
    │
    ▼
Answer + JSONL log entry  ← logs query, answer, latency, sources, refused?
    │
    ▼
Arize Phoenix traces      ← OpenTelemetry spans for rag-query and rag-retrieve
```

---

### Query Flow — Step by Step

What happens inside `rag.query("What is the CAP theorem?")` from first character to final answer.

```
User
 │
 │  question: str
 ▼
RAGSystem.query()                          opens rag-query OTel span
 │
 ├─ generates query_id (UUID[:8])          for JSONL log correlation
 ├─ starts retrieval timer (t0)
 │
 ▼
RAGSystem.retrieve()                       opens rag-retrieve child span
 │
 ├─ 1. EmbeddingGenerator.generate_embedding(query)
 │       → HTTP POST to Ollama /api/embeddings
 │       → returns numpy array shape (1024,)
 │
 ├─ 2. PgVectorStore.search(query_embedding, query_text, top_k=20)
 │       → runs hybrid SQL (see below)
 │       → returns 20 dicts: {source, chunk_index, content, cosine_score, text_score, combined_score}
 │
 ├─ 3. Build 20 pairs: [[query, chunk1], [query, chunk2], ...]
 │
 ├─ 4. CrossEncoder.predict(pairs)
 │       → single forward pass over all 20 (query, chunk) pairs
 │       → returns 20 raw logit floats (range roughly −12 to +7)
 │       → higher = "this chunk directly answers the query"
 │
 ├─ 5. Sort by logit score descending, slice top_k=3
 │
 ├─ 6. (if parent_retrieval=True)
 │       → collect unique sources from top-3
 │       → PgVectorStore.get_chunks_by_source(source) for each
 │       → replace results with ALL chunks from those sources
 │
 └─ returns List[(chunk_text, score, {source, chunk_index})]
 │
 ◀─ retrieval_ms recorded
 │
 ├─ 7. Build context string:
 │       "[Source: cap_theorem]\n<chunk text>\n\n[Source: consistency_models]\n<chunk text>..."
 │
 ├─ 8. Build prompt:
 │       system rules (answer ONLY from context, refuse if absent)
 │       + context string
 │       + "Question: ..."
 │
 ├─ starts generation timer (t0)
 │
 ▼
 9. ollama.generate(model="llama3.1", prompt=prompt)
       → HTTP POST to Ollama /api/generate
       → streams tokens, waits for done=true
       → returns {"response": "The CAP theorem states..."}
 │
 ◀─ generation_ms recorded
 │
 ├─ 10. _log_query() → appends one JSON line to logs/queries.jsonl
 │
 ├─ 11. set rag-query span attributes (answer, latencies, refused)
 │       → Phoenix collector receives both spans (rag-query + rag-retrieve)
 │
 └─ returns answer: str
```

#### The hybrid SQL query (step 2 in detail)

```sql
WITH semantic AS (
    -- cosine similarity for every row (full scan, uses IVFFlat index)
    SELECT id, source, chunk_index, content,
           1 - (embedding <=> <query_vector>) AS cosine_score
    FROM documents
),
keyword AS (
    -- full-text match only for rows that contain query terms
    SELECT id, ts_rank(content_tsv, plainto_tsquery('english', <query>)) AS text_score
    FROM documents
    WHERE content_tsv @@ plainto_tsquery('english', <query>)
)
SELECT s.*, s.cosine_score, COALESCE(k.text_score, 0),
       (0.7 * s.cosine_score + 0.3 * COALESCE(k.text_score, 0)) AS combined_score
FROM semantic s
LEFT JOIN keyword k ON s.id = k.id
ORDER BY combined_score DESC
LIMIT 20
```

`LEFT JOIN` means every document gets a combined score — documents with no keyword match get `text_score = 0`, so they still appear via cosine. Only the top-20 by combined score go to the cross-encoder.

#### What the cross-encoder actually does differently (step 4)

The embedding model encodes query and chunk *separately* into vectors, then measures the distance. It never sees them together. The cross-encoder receives `[CLS] query [SEP] chunk [SEP]` as a single sequence and runs full attention over both — it understands the *relationship*, not just the overlap. This is why it can demote a chunk that's semantically close but doesn't answer the question, and promote a chunk that directly answers it even if the wording differs.

The tradeoff: cross-encoder can't be used for full-corpus retrieval (one forward pass per chunk = O(N) at query time). That's why the pipeline is bi-encoder first (fast, approximate) → cross-encoder second (slow, precise) over just the top-20 shortlist.

#### Confidence levels (how `_confidence_level` maps scores to labels)

| Cross-encoder logit | Label | Meaning |
|---|---|---|
| ≥ 0 | Very High | Chunk directly answers the query |
| −5 to 0 | High | Chunk is highly relevant |
| −10 to −5 | Medium | Chunk is related but not a direct answer |
| < −10 | Low | Likely a false positive from hybrid search |

A **large gap** between rank-1 and rank-2 scores (e.g. +6.2 vs −1.5) means the cross-encoder is confident. **Tight clustering** (all scores around −3) means the query is ambiguous or the answer isn't in the corpus.

#### Two parallel outputs per query

| Output | Where | Persists? |
|---|---|---|
| Answer | returned to caller | No |
| JSONL log entry | `logs/queries.jsonl` | Yes — survives restarts |
| OTel trace (2 spans) | Phoenix at localhost:6006 | No — in-memory only |

---

### Stack

| Component | Choice | Why |
|---|---|---|
| Embeddings | mxbai-embed-large (Ollama) | 1024-dim, strong on technical text, runs locally |
| Vector DB | PostgreSQL + pgvector | Persistent, hybrid SQL search, no separate service |
| Full-text search | PostgreSQL tsvector + GIN index | Replaces BM25 — maintained automatically, scales |
| Re-ranker | cross-encoder/ms-marco-MiniLM-L-6-v2 | Lightweight, runs locally, understands query-chunk relationship |
| LLM | llama3.1 (Ollama) | Runs locally, no API cost during learning |
| Observability | Arize Phoenix | OpenTelemetry traces per query, free tier |

---

### Design Decisions (read this before touching the code)

#### 1. ParagraphChunker, not CharacterChunker

`DocumentChunker` splits on fixed character windows. It cuts words and sentences in half. The embedding model then encodes a meaningless fragment and produces a poor vector. `ParagraphChunker` splits on `\n\n` and merges small paragraphs up to 500 chars. Each chunk encodes one complete thought.

**The original bug:** the first version iterated `documents.keys()` (the filename strings) instead of `documents.values()` (the content). The LLM appeared to work because it saw `"cap_theorem"` as context and answered from its own training. Fixed in Session 1.

#### 2. Hybrid search: 70% semantic + 30% keyword

Pure cosine similarity fails when all docs are in the same domain (distributed systems). Every chunk scores 0.40–0.51 — no signal. BM25/tsvector rescues exact-terminology queries ("gossip protocol failure detection") that semantic search would rank poorly. Neither alone is optimal.

The SQL query runs a single pass with `LEFT JOIN` on the tsvector match so documents with no keyword match still appear via cosine. `COALESCE(k.text_score, 0)` handles the no-match case.

#### 3. Cross-encoder re-ranking over top-20

Bi-encoders (the embedding model) compare *summaries* of query and chunk independently. A chunk can have high cosine similarity because it shares the topic without actually answering the question. The cross-encoder reads `[query, chunk]` together as a single sequence — it understands the relationship, not just the overlap.

We can't cross-encode the full corpus (O(N) forward passes per query). The pattern: bi-encoder gets cheap top-20 candidates, cross-encoder does expensive precise re-ranking of just those 20.

Cross-encoder scores are raw logits (typically −12 to +7). A rank-1 score of +6 with rank-2 at −1.5 means high confidence. Tight clustering around −3 means the query is ambiguous.

#### 4. Parent document retrieval (optional, flag-controlled)

When all docs are in the same domain, chunk scores cluster tightly. The actual answer chunk may rank #22 while a tangentially related chunk ranks #4. `parent_retrieval=True` means: once *any* chunk from a source appears in top-k, pull *all* chunks from that source. This guarantees the LLM gets the full document context.

Disabled by default because it inflates the context sent to the LLM. Enable it for hard multi-hop questions. **Do not measure precision after parent expansion** — the extra chunks will look like noise.

V1 expanded parents by filtering the in-memory top-20 pool — it missed sibling chunks ranked below 20. V2 calls `get_chunks_by_source()` directly against the DB, so all chunks are guaranteed regardless of their individual scores.

#### 5. Strict prompt — no outside knowledge

The LLM will hallucinate confidently if the prompt doesn't gate it. The prompt has a hard rule: *"Answer using ONLY the context provided. If the context does not contain the answer, respond with: 'I don't have information about this in the provided documents.'"*

This was validated in Session 1: after adding the strict rule, out-of-corpus queries ("How does Kubernetes work?" before kubernetes docs existed) correctly refused even when retrieval returned vaguely related chunks with positive cross-encoder scores.

#### 6. Persistent store (V2 migration)

V1 used `InMemoryVectorStore` (numpy + BM25). Every run re-embedded all documents — minutes of startup time. V2 migrates to PostgreSQL + pgvector: ingest once with `load_documents.py`, then every subsequent `RAGSystem` startup connects to the DB and is query-ready immediately.

`load_documents.py` is idempotent: it checks `existing_sources()` before embedding, skips already-ingested files. A `UNIQUE (source, chunk_index)` constraint at the DB level is the final guard.

#### 7. IVFFlat index lists tuning

pgvector's IVFFlat index partitions vectors into clusters. At query time it only searches nearby clusters — O(lists) not O(N). Rule of thumb: `lists ≈ sqrt(row_count)`. With ~1000 rows, `lists = 30`. With 10K rows, rebuild to `lists = 100`. Rebuild command: `REINDEX INDEX idx_embedding`.

---

### Setup from Scratch

```bash
# 1. Start PostgreSQL with pgvector
docker compose up -d

# 2. Create schema (run once)
# Connect to DB and execute DB_Commands.sql
# docker exec -i rag-postgres psql -U raguser -d ragdb < DB_Commands.sql

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Pull Ollama models
ollama pull mxbai-embed-large
ollama pull llama3.1

# 5. Ingest documents (run once; idempotent — safe to re-run)
cd src
python load_documents.py

# 6. Run interactive query loop
python RAGSystem.py
```

---

### Key Files

```
src/
  DocumentChunker.py    — CharacterChunker (unused) + ParagraphChunker (active)
  EmbeddingGenerator.py — wraps Ollama embeddings API, returns numpy arrays
  PgVectorStore.py      — hybrid SQL search, add_documents, get_chunks_by_source
  RAGSystem.py          — orchestrates retrieve → rerank → generate → log
  load_documents.py     — one-time ingestion script; skips already-ingested sources
  observability.py      — Arize Phoenix setup helpers

eval/
  golden_dataset.py     — 20 curated Q&A pairs (easy/medium/hard/out-of-corpus)
  metrics.py            — precision, recall, hit_rate, MRR, faithfulness, refusal_correctness
  run_eval.py           — full eval runner; saves timestamped JSON to eval/results/
  analyze_logs.py       — CLI analyzer for logs/queries.jsonl (latency, refusal rate)
  results/              — timestamped eval snapshots (JSON)

docs/                   — 105 .txt knowledge base documents
logs/
  queries.jsonl         — auto-appended structured log for every query
generate_docs.py        — generates new docs via local Ollama model (idempotent)
DB_Commands.sql         — full schema: table, unique constraint, indexes, tsvector trigger
docker-compose.yml      — PostgreSQL + pgvector container
```

---

### Observability & Tracing

The system uses **Arize Phoenix** as a local OpenTelemetry trace collector. It runs in-process — no separate service to start.

#### How it starts

Inside `RAGSystem.__init__`:

```python
px.launch_app()   # starts Phoenix server on http://localhost:6006 (in-process, same Python process)
register()        # registers Phoenix as the OTLP exporter for opentelemetry-sdk
self.tracer = trace.get_tracer(__name__)  # standard OTel tracer for this module
```

`register()` (from `phoenix.otel`) does the OpenTelemetry boilerplate for you — it sets up the OTLP exporter pointed at Phoenix's collector endpoint. Without it, `trace.get_tracer()` would still work but spans would go nowhere.

**Note:** `src/observability.py` exists as a helper (`start_phoenix()`) but is **not used** — `RAGSystem` calls `px.launch_app()` and `register()` directly. If you refactor to call `observability.start_phoenix()` instead, you'll need to wire up `register()` inside it too.

#### Span structure

Each call to `rag.query(...)` produces two nested OpenTelemetry spans:

```
rag-query  (parent)
└── rag-retrieve  (child)
```

**`rag-retrieve` span attributes:**

| Attribute | Value |
|---|---|
| `query` | the raw query string |
| `top_sources` | list of top-3 source doc names after re-ranking |

**`rag-query` span attributes:**

| Attribute | Value |
|---|---|
| `query` | the raw query string |
| `answer` | the LLM's full answer text |
| `num_chunks` | how many chunks were passed to the LLM |
| `retrieval_ms` | time spent in retrieve() in milliseconds |
| `generation_ms` | time spent in ollama.generate() in milliseconds |
| `refused` | `true` if answer contains "don't have information" |

#### What Phoenix shows

Open `http://localhost:6006` while `RAGSystem` is running. You'll see:

- A trace timeline per query — the parent `rag-query` span with the `rag-retrieve` child nested inside, so you can see the retrieval vs. generation split at a glance
- All span attributes as key-value pairs — search by `refused=true` to find queries the system declined to answer, or sort by `generation_ms` to spot slow generations
- A full trace history across all queries in the current session

#### Also: structured JSONL log

In parallel with traces, every query is appended to `logs/queries.jsonl`:

```json
{
  "query_id": "a3f2c1d8",
  "timestamp": "2026-05-28T10:00:00",
  "question": "What is the CAP theorem?",
  "answer": "...",
  "retrieval_ms": 920,
  "generation_ms": 5400,
  "total_ms": 6320,
  "num_chunks": 3,
  "top_sources": ["cap_theorem", "consistency_models", "distributed_transactions"],
  "top_scores": [6.22, -1.55, -7.85],
  "refused": false
}
```

Use `python eval/analyze_logs.py` to aggregate this log — it reports latency percentiles, refusal rate, and surfaces low-confidence queries (where the top score is below a threshold).

The JSONL log persists across sessions; Phoenix traces only exist in-memory for the current run.

---

### Running Evaluation

```bash
# Full eval against 20 golden questions
python eval/run_eval.py

# Analyze query logs (latency, refusals, low-confidence queries)
python eval/analyze_logs.py
```

#### Eval Metrics Explained

All metrics work on **source names** (e.g. `cap_theorem`, `circuit_breaker`), not raw chunk text. Each question in the golden dataset has a `relevant_sources` list — that's the ground truth the metrics compare against.

---

**Precision** — of the sources you retrieved, what fraction were actually relevant?

```
precision = |retrieved_set ∩ relevant_set| / |retrieved_set|
```

Note: uses *sets* — duplicate sources in the retrieved list are deduplicated first.

> q001 "What is the CAP theorem?"
> retrieved = ["pacelc_theorem", "pacelc_theorem", "cap_theorem"]
> as set → {"pacelc_theorem", "cap_theorem"}
> relevant = {"cap_theorem", "pacelc_theorem"}
> intersection = {"cap_theorem", "pacelc_theorem"} → 2 hits out of 2 retrieved → **precision = 1.0**
>
> Contrast — before pacelc_theorem was added to relevant_sources:
> retrieved_set = {"pacelc_theorem", "cap_theorem"}, relevant_set = {"cap_theorem"}
> intersection = {"cap_theorem"} → 1 hit out of 2 retrieved → **precision = 0.5**

High precision = low noise sent to the LLM.
Low precision = the LLM is getting distracted by unrelated sources.

---

**Recall** — of all relevant sources that exist, how many did you actually retrieve?

```
recall = |retrieved_set ∩ relevant_set| / |relevant_set|
```

> q006 "How does leader election work?"
> retrieved = {"leader_election", "split_brain_and_fencing_tokens", "crdts"}
> relevant = {"leader_election", "consensus_algorithms"}
> intersection = {"leader_election"} → 1 out of 2 relevant docs found → **recall = 0.5**
>
> q010 "What strategies exist for handling failures?"
> retrieved = {"distributed_job_scheduling", "retry_and_timeout_patterns", "geo_distributed_systems"}
> relevant = {"fault_tolerance", "circuit_breaker", "saga_pattern"}
> intersection = {} → 0 out of 3 relevant docs found → **recall = 0.0** (complete miss)

High recall = the LLM has all the information it needs.
Low recall = the LLM is missing critical sources and will give an incomplete answer.

---

**Hit Rate** — did at least one relevant source appear anywhere in the retrieved list?

```
hit_rate = 1.0 if retrieved_set ∩ relevant_set is non-empty, else 0.0
```

This is binary — it only answers "did we find *anything* useful?" It doesn't care about how many or where.

> q006: retrieved includes "leader_election" which is in relevant → **hit_rate = 1.0**
> q010: retrieved has zero overlap with relevant → **hit_rate = 0.0**

**Important:** Hit rate 1.0 doesn't mean retrieval was good. q006 has hit_rate=1.0 but precision=0.33 and recall=0.5 — it found one right doc but also retrieved two wrong ones and missed one relevant doc entirely. Always read hit_rate alongside precision and recall, not instead of them.

---

**MRR (Mean Reciprocal Rank)** — how early in the ranking did the first relevant source appear?

```
MRR = 1 / rank_of_first_relevant_source
      0.0 if no relevant source appears at all
```

Rank 1 → 1.0 (best) | Rank 2 → 0.5 | Rank 3 → 0.33 | Not found → 0.0

> q001 (before fix) — retrieved list = ["pacelc_theorem", "pacelc_theorem", "cap_theorem"]
> relevant = ["cap_theorem"] only
> cap_theorem first appears at position 3 → **MRR = 0.33**
>
> q001 (after adding pacelc_theorem to relevant_sources)
> pacelc_theorem first appears at position 1 → **MRR = 1.0**
>
> q010 — no relevant source appears anywhere in top-3 → **MRR = 0.0**

MRR penalises burying the answer. If the right document is always there but always at rank 3, MRR = 0.33 even if hit_rate = 1.0. The cross-encoder's job is to keep MRR high by pushing the most relevant source to rank 1.

---

**Faithfulness** — are the content words in the answer actually present in the retrieved context?

```
content_words = {w for w in answer.split() if len(w) > 5}   # skip short/stopwords
faithfulness = count(w in context_text for w in content_words) / len(content_words)
```

This is a **word-overlap proxy**, not semantic grounding. It catches obvious hallucinations (LLM used words that don't appear anywhere in the retrieved text) but won't catch subtle paraphrasing or plausible-but-wrong facts.

> q014 "What is consistent hashing?"
> Answer quotes the source almost verbatim: "manages and distributes load across nodes efficiently"
> Almost every content word appears in the retrieved chunk → **faithfulness = 0.91**
>
> q011 "Apache Spark" (when answered from mapreduce.txt)
> Answer uses "intermediate", "memory", "computation" — all in mapreduce.txt but not in the spirit of the question
> Word overlap is moderate → **faithfulness = 0.47**

Low faithfulness = the LLM introduced words not in the context = likely hallucination or heavy paraphrasing.
High faithfulness = the LLM stayed close to the source text.

**Caveat:** Gemma4 writes longer, bullet-pointed answers that quote more source text verbatim. This mechanically raises faithfulness scores vs. llama3.1's shorter prose answers — the metric rewards verbosity, not accuracy.

---

**Refusal Correctness** — did the system handle the question correctly given whether it's in-corpus or not?

```
out-of-corpus question (expected_answer = None):
  1.0 → correctly refused ("I don't have information...")
  0.0 → wrongly answered from context

in-corpus question (expected_answer is set):
  1.0 → answered cleanly
  0.5 → partial answer + partial refusal (answered what context supported, refused the rest)
  0.0 → refused entirely despite having relevant context
```

> q015 "What is an inverted index?" — inverted_index.txt in corpus, retrieved correctly
> Answered fully → **refusal_correct = 1.0**
>
> q019 "How does MVCC compare to 2PC?" — mvcc retrieved, 2PC not retrieved
> Answer: "MVCC achieves isolation via snapshot isolation. I don't have information about how this compares to 2PC."
> Substantive content before the refusal phrase → **refusal_correct = 0.5** (correct behaviour)
>
> q006 "How does leader election work?" — leader_election retrieved at rank 1 (gemma4 regression)
> Answer: "I don't have information..." despite having context
> No content before refusal phrase → **refusal_correct = 0.0** (model failure)
>
> q011 "What is Redis?" — no redis.txt exists, but distributed_caching.txt mentions Redis by name
> Answered with Redis detail from that doc → **refusal_correct = 0.0** (false positive — answer exists in corpus by proxy)

**Important:** Measure precision/recall *before* parent retrieval expansion. Parent retrieval inflates the source count, making precision look artificially low.

#### Current Eval Metrics (Session 5 — 2026-05-28, 20 questions, 105 docs)

| Metric | Score |
|---|---|
| Precision | 0.792 |
| Recall | 0.804 |
| Hit Rate | 0.950 |
| MRR | 0.900 |
| Faithfulness | 0.681 |
| Refusal Accuracy | 0.925 |
| MRR easy | 1.000 |
| MRR medium | 0.800 |
| MRR hard | 0.750 |

**Before/after — key tuning milestones:**

| Milestone | Precision | Recall | MRR |
|---|---|---|---|
| Baseline (Session 4, chunk=1000, top_k=5) | 0.575 | 0.819 | 0.854 |
| Chunk size 1000 → 500 | 0.604 | — | — |
| top_k 5 → 3 | 0.708 | 0.806 | 0.917 |
| Corpus 33 → 105 docs, 20 questions (Session 5) | **0.792** | **0.804** | **0.900** |

**MRR hard = 0.750** reflects the remaining architectural gap: hard questions requiring cross-doc synthesis (e.g., "How does MVCC compare to two-phase commit?") are still bounded by single-pass retrieval. Fix requires query expansion or multi-hop retrieval.

---

### Generating More Docs

`generate_docs.py` generates new knowledge-base documents using a local Ollama model. It is idempotent — it checks what already exists and only generates missing topics.

```bash
python generate_docs.py              # generate all missing topics
python generate_docs.py --dry-run    # preview what would be generated
python generate_docs.py --model gemma2:27b  # use a different model
python generate_docs.py --topic merkle_trees  # generate a single topic
```

After generating new docs, re-run ingestion:
```bash
cd src && python load_documents.py   # skips already-ingested, adds new ones
```

After ingestion grows significantly, rebuild the IVFFlat index:
```bash
# lists ≈ sqrt(total_rows). Check with: SELECT COUNT(*) FROM documents;
REINDEX INDEX idx_embedding;
# or drop/recreate with updated lists value
```

---

### Project Evolution

| Session | Date | What Changed |
|---|---|---|
| 1 | Feb 2026 | Fixed chunking bug, strict prompt, ParagraphChunker, parent retrieval |
| 2 | Mar 28 2026 | Hybrid BM25 + cosine, cross-encoder re-ranking, chunk tuning, 21 docs |
| 3 | Mar 31 2026 | V2: PostgreSQL + pgvector, persistent hybrid SQL search, 33 docs |
| 4 | Apr 2 2026 | Eval layer, Arize Phoenix observability, chunk size tuning |
| 5 | May 28 2026 | Corpus 33 → 105 docs, idempotent ingestion, golden dataset 12 → 20 questions |

See SESSIONS.md for detailed notes on what broke and what was learned in each session.

---

### Known Limitations & Next Steps

#### Retrieval gaps (current)

- **q010 "failure strategies" — complete retrieval miss (precision/recall/MRR all 0).** The query "what strategies exist for handling failures?" is broad enough that the hybrid search returns retry patterns and job scheduling docs instead of `fault_tolerance`, `circuit_breaker`, `saga_pattern`. Root cause: vague natural-language queries don't align well with specific technical document names. Fix: query expansion — generate 2–3 sub-queries from the original and union their results before re-ranking.

- **q018 "CQRS + event sourcing" — `cqrs.txt` never retrieved (MRR 0.5).** The query contains "CQRS" but the tsvector match isn't boosting `cqrs.txt` above `message_queues` and `design_patterns`. Fix: either expand `relevant_sources` to accept `event_sourcing` alone as sufficient (the retrieved answer is correct), or increase BM25 weight for exact acronym matches.

- **q012 "What is Redis?" — system answers instead of refusing.** `distributed_caching.txt` and `distributed_locking.txt` mention Redis by name. This is in-corpus by proxy, so the system correctly answers from context but the eval treats it as a refusal failure. Fix: replace q012 with a topic that has zero mentions anywhere — e.g. "How does React's reconciliation algorithm work?"

#### Eval quality

- **Faithfulness is a word-overlap proxy.** Content words (>5 chars) in the answer are checked against the retrieved context. It rewards verbosity (gemma4 scores higher than llama3.1 simply by quoting more) and misses semantic hallucinations. Fix: LLM-as-judge — pass (answer, context) to a model and ask "is this answer supported by this context?" Costs one generation call per eval question.

- **MRR hard = 0.75 (was 0.50).** Improvement came from two new targeted hard questions (q019, q020), not from architectural changes. The original hard question q010 still scores 0. True cross-doc synthesis (combining `fault_tolerance` + `circuit_breaker` + `saga_pattern` into one answer) requires multi-hop retrieval — retrieve top docs, extract key facts, re-query with those facts, then synthesise.

#### Architecture gaps

- **No API layer.** The system is a CLI loop (`while True: input()`). A FastAPI service would expose `/query` (POST, takes `{"question": "..."}`, returns `{"answer": "...", "sources": [...], "retrieval_ms": ..., "generation_ms": ...}`), `/health` (GET, checks DB + Ollama reachability), and `/stats` (GET, aggregates `logs/queries.jsonl` — total queries, refusal rate, avg latency). This is the highest-impact change for portfolio visibility.

- **Generation latency unmeasured on gemma4.** Session 4 measured llama3.1 at ~5.5s generation / ~0.9s retrieval. After switching to gemma4, latency hasn't been re-measured. Run `python eval/analyze_logs.py` after a few queries to get the current p50/p95 breakdown. If gemma4 is slower, `llama3.2:3b` is a faster local alternative at lower quality.
