## RAG System — Distributed Systems Knowledge Base

A learning project building a RAG pipeline over 33 distributed systems documents.
Evolves across sessions: each session documents what broke, what was fixed, and why.

### Architecture
docs/*.txt → ParagraphChunker → EmbeddingGenerator → PostgreSQL + pgvector
                                                              ↓
User Query → Embed → Hybrid Search (70% semantic + 30% keyword) → Cross-Encoder Re-rank → LLM

### Stack
- Embeddings: mxbai-embed-large (Ollama)
- Vector DB: PostgreSQL + pgvector
- Re-ranking: cross-encoder/ms-marco-MiniLM-L-6-v2
- LLM: llama3.1 (Ollama)
- Observability: Arize Phoenix

### Setup
# 1. Start PostgreSQL
docker-compose up -d

# 2. Install dependencies
pip install -r requirements.txt

# 3. Ingest documents
python src/load_documents.py

# 4. Run
python src/RAGSystem.py

### Evaluation
python eval/run_eval.py        # run against golden dataset
python eval/analyze_logs.py    # analyze query logs

### Current Eval Metrics (Session 4 — 2026-04-02)
| Metric      | Score |
|-------------|-------|
| Precision   | 0.708 |
| Recall      | 0.806 |
| MRR         | 0.917 |
| Faithfulness| 0.594 |

### Project Structure
src/                  # core pipeline
eval/                 # evaluation layer
  golden_dataset.py   # ground truth Q&A pairs
  metrics.py          # precision, recall, MRR, faithfulness
  run_eval.py         # evaluation runner
  analyze_logs.py     # log analysis
  results/            # timestamped eval snapshots
logs/                 # auto-generated query logs (JSONL)
docs/                 # knowledge base (33 .txt files)

See SESSIONS.md for full learning log.
