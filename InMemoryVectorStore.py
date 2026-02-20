from typing import List, Tuple, Optional
import numpy as np

class InMemoryVectorStore:
    """Simple in-memory vector store with cosine similarity search"""
    
    def __init__(self):
        self.documents = []
        self.metadata = []        # stores source info per chunk e.g. {"source": "cap_theorem", "chunk": 2}
        self.embeddings = None
        
    def add_documents(self, documents: List[str], embeddings: np.ndarray, metadata: Optional[List[dict]] = None):
        """Add documents and their embeddings to the store
        
        Args:
            documents: List of text chunks
            embeddings: Corresponding embedding vectors
            metadata: Optional list of dicts with info like {"source": "cap_theorem", "chunk_index": 0}
        """
        self.documents.extend(documents)
        self.metadata.extend(metadata if metadata else [{} for _ in documents])
        
        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])
            
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        return dot_product / (norm1 * norm2)
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5, parent_retrieval: bool = False) -> List[Tuple[str, float, dict]]:
        """
        Search for most similar documents using cosine similarity.
        
        Args:
            query_embedding: Embedding vector of the query
            top_k: Number of top results to return
            parent_retrieval: If True, once a chunk from a document ranks in top-k,
                              ALL chunks from that document are included in the results.
                              This ensures the full context of a relevant document is retrieved,
                              not just one fragmented chunk.
            
        Returns:
            List of (chunk_text, similarity_score, metadata) tuples, sorted by score descending
        """
        if self.embeddings is None or len(self.documents) == 0:
            return []
        
        # Calculate similarities for all chunks
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            similarity = self.cosine_similarity(query_embedding, doc_embedding)
            similarities.append((self.documents[i], similarity, self.metadata[i]))
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        if not parent_retrieval:
            return similarities[:top_k]
        
        # Parent Document Retrieval:
        # Find which source documents appear in the top-k, then pull ALL their chunks
        top_k_results = similarities[:top_k]
        top_sources = {meta.get("source") for _, _, meta in top_k_results}
        
        # Expand: include every chunk belonging to those source documents
        expanded = [item for item in similarities if item[2].get("source") in top_sources]
        
        return expanded