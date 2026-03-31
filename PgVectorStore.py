import psycopg2
import numpy as np
from pgvector.psycopg2 import register_vector

class PgVectorStore:
    def __init__(self, connection_string="postgresql://raguser:ragpass@localhost:5432/ragdb"):
        self.conn = psycopg2.connect(connection_string)

        # register the vector type
        register_vector(self.conn)
        
    def add_documents(self, chunks, embeddings, sources):
        """
        chunks: list of strings (chunk text)
        embeddings: list of numpy arrays (one per chunk)
        sources: list of strings (source filename per chunk)
        """
        cursor = self.conn.cursor()
        source_counters = {}

        for chunk, embedding, source in zip(chunks, embeddings, sources):
            chunk_index = source_counters.get(source, 0)
            source_counters[source] = chunk_index + 1

            cursor.execute(
                """
                INSERT INTO documents (source, chunk_index, content, embedding)
                VALUES (%s, %s, %s, %s)
                """,
                (source, chunk_index, chunk, embedding.tolist())
            )

        self.conn.commit()
        cursor.close()
        
    def close(self):
        self.conn.close()
        
    def search(self, query_embedding, query_text, top_k=20, semantic_weight=0.7):
        """
        Hybrid search: pgvector cosine similarity + PostgreSQL full-text search.
        Returns top_k results ranked by weighted combination.
        
        query_embedding: numpy array from your embedding model
        query_text: raw query string for full-text search
        top_k: number of results to return (these go to cross-encoder)
        semantic_weight: blend ratio (0.7 = 70% semantic, 30% keyword)
        """
        keyword_weight = 1 - semantic_weight
        cursor = self.conn.cursor()
        
        cursor.execute(
            """
            WITH semantic AS (
                SELECT id, source, chunk_index, content,
                    1 - (embedding <=> %s::vector) AS cosine_score
                FROM documents
            ),
            keyword AS (
                SELECT id,
                    ts_rank(content_tsv, plainto_tsquery('english', %s)) AS text_score
                FROM documents
                WHERE content_tsv @@ plainto_tsquery('english', %s)
            )
            SELECT s.id, s.source, s.chunk_index, s.content,
                s.cosine_score,
                COALESCE(k.text_score, 0) AS text_score,
                (%s * s.cosine_score + %s * COALESCE(k.text_score, 0)) AS combined_score
            FROM semantic s
            LEFT JOIN keyword k ON s.id = k.id
            ORDER BY combined_score DESC
            LIMIT %s
            """,
            (query_embedding.tolist(), query_text, query_text,
            semantic_weight, keyword_weight, top_k)
        )
        
        results = cursor.fetchall()
        cursor.close()
        
        return [
            {
                "id": row[0],
                "source": row[1],
                "chunk_index": row[2],
                "content": row[3],
                "cosine_score": row[4],
                "text_score": row[5],
                "combined_score": row[6]
            }
            for row in results
        ]
    def get_chunks_by_source(self, source):
        """
        Retrieve all chunks from a given source document.
        This enables parent document retrieval - if any chunk hits top-k,
        pull all sibling chunks for full context. 
        This is used based on flag for now.
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT id, source, chunk_index, content
            FROM documents
            WHERE source = %s
            ORDER BY chunk_index ASC
            """,
            (source,)
        )
        results = cursor.fetchall()
        cursor.close()
        
        return [
            {
                "id": row[0],
                "source": row[1],
                "chunk_index": row[2],
                "content": row[3]
            }
            for row in results
        ]
        
    def count_documents(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM documents")
        count = cursor.fetchone()[0]
        cursor.close()
        return count