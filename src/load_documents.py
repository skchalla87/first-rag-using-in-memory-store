"""
Script to load documents from the docs/ directory into PgVectorStore.
Chunks each document, generates embeddings, and inserts into PostgreSQL.
"""

from pathlib import Path
from DocumentChunker import ParagraphChunker
from EmbeddingGenerator import EmbeddingGenerator
from PgVectorStore import PgVectorStore


def load_documents(
    docs_dir: str = "./docs",
    embedding_model: str = "mxbai-embed-large",
    connection_string: str = "postgresql://raguser:ragpass@localhost:5432/ragdb",
):
    store = PgVectorStore(connection_string)
    chunker = ParagraphChunker()
    embedder = EmbeddingGenerator(embedding_model)

    docs_path = Path(docs_dir)
    txt_files = list(docs_path.glob("*.txt"))

    if not txt_files:
        print(f"No .txt files found in {docs_dir}")
        return

    print(f"Found {len(txt_files)} documents. Starting load...\n")

    for file_path in sorted(txt_files):
        source = file_path.stem
        text = file_path.read_text(encoding="utf-8")

        chunks = chunker.chunk_text(text)
        if not chunks:
            print(f"  [{source}] No chunks — skipping")
            continue

        print(f"  [{source}] {len(chunks)} chunks — generating embeddings...")
        embeddings = embedder.generate_embeddings(chunks)
        sources = [source] * len(chunks)

        store.add_documents(chunks, embeddings, sources)
        print(f"  [{source}] inserted.")

    total = store.count_documents()
    print(f"\nDone. Total documents in DB: {total}")
    store.close()


if __name__ == "__main__":
    load_documents()
