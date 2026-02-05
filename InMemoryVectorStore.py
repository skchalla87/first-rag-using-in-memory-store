from typing import List
import numpy as np

class InMemoryVectorStore:
    """Simple in-memory vector store with cosine similarity search"""
    
    def __init__(self):
        self.documents = []
        self.embeddings = None
        
    def add_documents(self, documents: List[str], embeddings: np.ndarray):
        """Add documents and their embeddings to the store"""
        self.documents.extend(documents)
        
        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])
            
    def cosine_similarity(slef, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        return dot_product / (norm1 * norm2)
    
    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Search for most similar documents using cosine similarity
        
        Args:
            query_embedding: Embedding vector of the query
            top_k: Number of top results to return
            
        Returns:
            List of (document, similarity_score) tuples
        """
        if self.embeddings is None or len(self.documents) == 0:
            return []
        
        # Calculate similarities for all documents
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            similarity = self.cosine_similarity(query_embedding, doc_embedding)
            similarities.append((self.documents[i], similarity))
        
        # Sort by similarity (highest first) and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]