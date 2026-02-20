"""
RAG System - Main class that integrates chunking, embedding, and retrieval
"""
from pathlib import Path
import ollama
from typing import List, Tuple

from DocumentChunker import DocumentChunker
from EmbeddingGenerator import EmbeddingGenerator
from InMemoryVectorStore import InMemoryVectorStore


class RAGSystem:
    """Main RAG system that combines retrieval and generation"""
    
    def __init__(self,
                 embedding_model: str = 'nomic-embed-text',
                 llm_model: str = "llama3.1",
                 chunk_size: int = 500,
                 overlap: int = 50
                 ):
        
        """
        Args:
            embedding_model: Ollama model for embeddings
            llm_model: Ollama model for text generation
            chunk_size: Size of document chunks
            overlap: Overlap between chunks
        """
        self.chunker = DocumentChunker(chunk_size, overlap)
        self.embedding_generator =  EmbeddingGenerator(embedding_model)
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
        
    def add_documents(self, documents: List[str]):
        """Add Documents to RAG system"""
        
        print(f"Adding {len(documents)} documents to the system...")
        
        # chuck all documents
        
        all_chunks = []
        for doc in documents:
            chunks = self.chunker.chunk_text(doc)
            all_chunks.extend(chunks)
            
        print(f"Created {len(all_chunks)} chunks from the documents")
        
        # Generate Embeddings for all the chunks
        print("Generating Embeddings...")
        embeddings = self.embedding_generator.generate_embeddings(all_chunks)
        
        # Add Embeddings to Vector Store
        self.vector_store.add_documents(all_chunks, embeddings)
        print("Documents added to Vector Store!!!")
        
    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Retrieve relevant documents for a query"""
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_embedding(query)
        
        # Search vector store
        results = self.vector_store.search(query_embedding, top_k)
        return results
    
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
            print("\n🔍 Retrieving relevant context for: '{question}'")
            
        retrieved_docs = self.retrieve(question, top_k)
        
        if verbose:
            print(f"\n📚 Retrieved {len(retrieved_docs)} relevant chunks:")
            for i, (doc, score) in enumerate(retrieved_docs, 1):
                print(f"\n  {i}. (Similarity: {score:.4f})")
                print(f"     {doc[:150]}..." if len(doc) > 150 else f"     {doc}")
                
        # combine retrieved context
        context = "\n\n".join([doc for doc, _ in retrieved_docs])
        
        # create prompt with context
        prompt = f"""You are a helpful assistant. Use the following context to answer the question. 
        if the answer is not in the context, say so.
        
        Context:
        {context}
        
        Question: {question}

        Answer:"""
        
        # Generate response
        if verbose:
            print(f"\n 💭 Generating Answer...")
            
        response = ollama.generate(model=self.llm_model, prompt=prompt)
        return response['response']
    
if __name__ == "__main__":
    # Simple test
    print("=" * 60)
    print("Basic RAG System - Simple Test")
    print("=" * 60)
    
    # Create RAG system
    rag = RAGSystem()
    
    rag.load_documents('./docs')
    
    # Get query from user
    question = input("\n❓ Enter your question: ")
    answer = rag.query(question)
    print(f"\n✅ Answer: {answer}")
    print("\n" + "=" * 60)