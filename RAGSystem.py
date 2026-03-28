"""
RAG System - Main class that integrates chunking, embedding, and retrieval
"""
from pathlib import Path
import ollama
from typing import List, Tuple, Dict

from DocumentChunker import DocumentChunker, ParagraphChunker
from EmbeddingGenerator import EmbeddingGenerator
from InMemoryVectorStore import InMemoryVectorStore
from sentence_transformers import CrossEncoder

class RAGSystem:
    """Main RAG system that combines retrieval and generation"""
    
    def __init__(self,
                 embedding_model: str = 'nomic-embed-text',
                 llm_model: str = "llama3.1",
                 chunk_size: int = 200,
                 overlap: int = 50,
                 chunker_type: str = "paragraph",   # "character" or "paragraph"
                 parent_retrieval: bool = True       # fetch ALL chunks from matched docs
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
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        self.chunker_type = chunker_type
        self.parent_retrieval = parent_retrieval
        self.embedding_generator = EmbeddingGenerator(embedding_model)
        self.vector_store = InMemoryVectorStore()
        self.llm_model = llm_model
        
    def load_documents(self, data_dir: str):
        """
        Load all .txt files from directory and index them
        
        Args:
            data_dir: Path to directory containing .txt files
        """
        print(f"\nLoading documents from {data_dir}...")
        
        # Read all .txt files
        documents = {}
        data_path = Path(data_dir)
        
        for file_path in data_path.glob("*.txt"):
            with open(file_path, 'r', encoding='utf-8') as f:
                documents[file_path.stem] = f.read()
        
        print(f"Found {len(documents)} documents")
        self.add_documents(documents)
        
    def add_documents(self, documents: Dict[str, str]):
        """Add Documents to RAG system
        
        Args:
            documents: Dict of {doc_name: content}
        """
        
        print(f"Adding {len(documents)} documents to the system...")
        
        # chunk all documents, tracking source name and chunk index
        all_chunks = []
        all_metadata = []
        for doc_name, doc_content in documents.items():
            chunks = self.chunker.chunk_text(doc_content)
            for chunk_index, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadata.append({"source": doc_name, "chunk_index": chunk_index})
            
        print(f"Created {len(all_chunks)} chunks from the documents")
        
        # Generate Embeddings for all the chunks
        print("Generating Embeddings...")
        embeddings = self.embedding_generator.generate_embeddings(all_chunks)
        
        # Add Embeddings to Vector Store
        self.vector_store.add_documents(all_chunks, embeddings, all_metadata)
        print("Documents added to Vector Store!!!")
        
    @staticmethod
    def _confidence_level(score: float) -> str:
        """Map a similarity score to a human-readable confidence level"""
        if score >= 0.70:
            return "🟢 Very High"
        elif score >= 0.60:
            return "🟢 High"
        elif score >= 0.50:
            return "🟡 Medium"
        else:
            return "🔴 Low"
    
    def retrieve(self, query: str, top_k: int = 5, debug: bool = False) -> List[Tuple[str, float, dict]]:
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
        candidate_docs =  self.vector_store.search(query_embedding, top_k=initial_pool_size, query_text=query)
        
        # If no docs returned, just return empty
        if not candidate_docs:
            return []
        
        # Rerank the docs
        
        # Prepare pairs of (Query, Chunk) for the Cross-Encoder
        pairs = [[query, doc[0]] for doc in candidate_docs]
        
        # predict the exact relevance scores
        rerank_scores = self.reranker.predict(pairs)
        
        # Combine the original docs with their new reranked scores
        reranked_results = []
        for idx, score in enumerate(rerank_scores):
            chunk_text = candidate_docs[idx][0]
            metadata = candidate_docs[idx][2]
            reranked_results.append((chunk_text, float(score), metadata))
        
         # Sort by the new reranker score (descending)
        reranked_results.sort(key=lambda x: x[1], reverse=True)
                
        # Get ALL results for debugging
        all_results = self.vector_store.search(query_embedding, top_k=15, query_text=query)
        
        if debug:
            print("\n🔬 [DEBUG] Full ranking of ALL chunks:")
            for rank, (doc, score, meta) in enumerate(reranked_results, 1):
                
                source = meta.get("source", "unknown")
                chunk_idx = meta.get("chunk_index", "?")
                print(f"   {rank:>3}. [{source}] chunk {chunk_idx} — score: {score:.4f}")
        
        # 4. Return the final top_k documents to the LLM
        return reranked_results[:top_k]
    
    def query(self, question: str, top_k : int = 3, verbose : bool = True) -> str:
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
                print(f"\n  {i}. 📄 Source: [{source}] chunk {chunk_idx} | Confidence: {confidence} (score: {score:.4f})")
                print(f"     {doc[:150]}..." if len(doc) > 150 else f"     {doc}")
                
        # combine retrieved context (with source label for LLM)
        context = "\n\n".join(
            [f"[Source: {meta.get('source', 'unknown')}]\n{doc}" for doc, _, meta in retrieved_docs]
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
            #print(f"\n📝 Prompt being sent to LLM:\n{'='*60}\n{prompt}\n{'='*60}")
            
        response = ollama.generate(model=self.llm_model, prompt=prompt)
        return response['response']
    
if __name__ == "__main__":
    print("=" * 60)
    print("Basic RAG System - Simple Test")
    print("=" * 60)
    
    # Choose chunking strategy:
    #   "paragraph" - splits on blank lines, keeps paragraphs intact (recommended)
    #   "character" - fixed-size windows with overlap (simpler, cuts mid-word)
    rag = RAGSystem(chunker_type="character", parent_retrieval=False, embedding_model='mxbai-embed-large')
    
    rag.load_documents('./docs')
    
    # Get query from user
    question = input("\n❓ Enter your question: ")
    answer = rag.query(question)
    print(f"\n✅ Answer: {answer}")
    print("\n" + "=" * 60)