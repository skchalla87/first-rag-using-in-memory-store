"""
RAG System - Main class that integrates chunking, embedding, and retrieval
"""

import ollama
from typing import List, Tuple

from DocumentChunker import DocumentChunker, ParagraphChunker
from EmbeddingGenerator import EmbeddingGenerator
from PgVectorStore import PgVectorStore
from sentence_transformers import CrossEncoder


class RAGSystem:
    """Main RAG system that combines retrieval and generation"""

    def __init__(
        self,
        embedding_model: str = "nomic-embed-text",
        llm_model: str = "llama3.1",
        chunk_size: int = 200,
        overlap: int = 50,
        chunker_type: str = "paragraph",  # "character" or "paragraph"
        parent_retrieval: bool = True,  # fetch ALL chunks from matched docs
        connection_string: str = "postgresql://raguser:ragpass@localhost:5432/ragdb"
    ):
        """
        Args:
            embedding_model: Ollama model for embeddings
            llm_model: Ollama model for text generation
            chunk_size: Size of document chunks (used only for character chunker)
            overlap: Overlap between chunks (used only for character chunker)
            chunker_type: Chunking strategy:
                - "character" : Fixed-size character windows with overlap (simple, can cut mid-word)
                - "paragraph" : Splits on blank lines, keeps paragraphs intact (better semantic coherence)
            parent_retrieval: If True, once any chunk from a document ranks in top-k,
                              ALL chunks from that document are retrieved to give the LLM
                              the full context of that document.
        """
        if chunker_type == "paragraph":
            self.chunker = ParagraphChunker()
        else:
            print("Using Character Chunker!!")
            self.chunker = DocumentChunker(chunk_size, overlap)
        # 1. Initialize the Cross-Encoder model (e.g., a lightweight ms-marco model)
        print("Loading Reranker Model...")
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        self.chunker_type = chunker_type
        self.parent_retrieval = parent_retrieval
        self.embedding_generator = EmbeddingGenerator(embedding_model)
        self.vector_store = PgVectorStore(connection_string)
        self.llm_model = llm_model


    @staticmethod
    def _confidence_level(score: float) -> str:
        if score >= 0:
            return "🟢 Very High"
        elif score >= -5:
            return "🟢 High"
        elif score >= -10:
            return "🟡 Medium"
        else:
            return "🔴 Low"


    def retrieve(
        self, query: str, top_k: int = 5, debug: bool = False
    ) -> List[Tuple[str, float, dict]]:
        """Retrieve relevant chunks for a query

        Args:
            query: The search query
            top_k: Number of top chunks to retrieve (before parent expansion)
            debug: If True, print ALL chunk scores to see full ranking

        Returns:
            List of (chunk_text, similarity_score, metadata) tuples
        """
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_embedding(query)

        # retrieve top 20 candidates from vector store
        initial_pool_size = 20
        candidate_docs = self.vector_store.search(
            query_embedding, query_text=query, top_k=initial_pool_size
        )

        # If no docs returned, just return empty
        if not candidate_docs:
            return []

        # Rerank the docs

        # Prepare pairs of (Query, Chunk) for the Cross-Encoder
        pairs = [[query, doc["content"]] for doc in candidate_docs]

        # predict the exact relevance scores
        rerank_scores = self.reranker.predict(pairs)

        # Combine the original docs with their new reranked scores
        reranked_results = []
        for idx, score in enumerate(rerank_scores):
            chunk_text = candidate_docs[idx]["content"]
            metadata = {
                "source": candidate_docs[idx]["source"],
                "chunk_index": candidate_docs[idx]["chunk_index"],
            }
            reranked_results.append((chunk_text, float(score), metadata))

        # Sort by the new reranker score (descending)
        reranked_results.sort(key=lambda x: x[1], reverse=True)

        if debug:
            print("\n🔬 [DEBUG] Full ranking of ALL chunks:")
            for rank, (_, score, meta) in enumerate(reranked_results, 1):

                source = meta.get("source", "unknown")
                chunk_idx = meta.get("chunk_index", "?")
                print(
                    f"   {rank:>3}. [{source}] chunk {chunk_idx} — score: {score:.4f}"
                )

        # 4. Return the final top_k documents to the LLM
        top_results = reranked_results[:top_k]
        if self.parent_retrieval:
            top_sources = {meta.get("source") for _, _, meta in top_results}
            expanded = []
            for source in top_sources:
                for row in self.vector_store.get_chunks_by_source(source):
                    expanded.append((
                        row["content"],
                        float("-inf"),
                        {"source": row["source"], "chunk_index": row["chunk_index"]},
                    ))
            top_results = expanded

        return top_results

    def query(self, question: str, top_k: int = 3, verbose: bool = True) -> str:
        """
        Query the RAG system with a question

        Args:
            question: User's question
            top_k: Number of relevant chunks to retrieve
            verbose: Whether to print retrieval details

        Returns:
            Generated answer
        """
        if verbose:
            print(f"\n🔍 Retrieving relevant context for: '{question}'")

        retrieved_docs = self.retrieve(question, top_k, debug=verbose)

        if verbose:
            print(f"\n📚 Retrieved {len(retrieved_docs)} relevant chunks:")
            for i, (doc, score, meta) in enumerate(retrieved_docs, 1):
                confidence = self._confidence_level(score)
                source = meta.get("source", "unknown")
                chunk_idx = meta.get("chunk_index", "?")
                print(
                    f"\n  {i}. 📄 Source: [{source}] chunk {chunk_idx} | Confidence: {confidence} (score: {score:.4f})"
                )
                print(f"     {doc[:150]}..." if len(doc) > 150 else f"     {doc}")

        # combine retrieved context (with source label for LLM)
        context = "\n\n".join(
            [
                f"[Source: {meta.get('source', 'unknown')}]\n{doc}"
                for doc, _, meta in retrieved_docs
            ]
        )

        # create prompt with context
        prompt = f"""You are a helpful assistant. Answer the question using ONLY the context provided below.

Rules:
- Base your answer strictly on the context. Do NOT use outside knowledge.
- If the context does not contain the answer, respond with: "I don't have information about this in the provided documents."
- Quote or reference the source when possible.

Context:
{context}

Question: {question}

Answer:"""

        # Generate response
        if verbose:
            print(f"\n 💭 Generating Answer...")
            # print(f"\n📝 Prompt being sent to LLM:\n{'='*60}\n{prompt}\n{'='*60}")

        response = ollama.generate(model=self.llm_model, prompt=prompt)
        return response["response"]

    def close(self):
        self.vector_store.close()


if __name__ == "__main__":
    print("=" * 60)
    print("Basic RAG System - Simple Test")
    print("=" * 60)

    # Choose chunking strategy:
    #   "paragraph" - splits on blank lines, keeps paragraphs intact (recommended)
    #   "character" - fixed-size windows with overlap (simpler, cuts mid-word)
    rag = RAGSystem(
        parent_retrieval=False,
        embedding_model="mxbai-embed-large",
    )

    # Get query from user
    question = input("\n❓ Enter your question: ")
    answer = rag.query(question)
    print(f"\n✅ Answer: {answer}")
    print("\n" + "=" * 60)
    rag.close()
