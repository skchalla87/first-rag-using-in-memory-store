"""
Embedding Generator.
"""
import ollama
import numpy as np
from typing import List, Tuple
  
class EmbeddingGenerator:
        """Generates vector embeddings using Ollama's embedding model"""
        
        def __init__(self, model: str = "mxbai-embed-large"):
            """
            Args:
                model: Ollama embedding model to use
            """
            self.model = model
            
        def generate_embedding(self, text: str) -> np.ndarray:
            """
                Generate EßMbeddings
            """
            response = ollama.embeddings(model=self.model, prompt=text)
            return np.array(response['embedding'])
        
        def generate_embeddings(self, texts: List[str]) -> np.ndarray:
            """Generate embeddings for multiple texts"""
            embeddings = []
            for text in texts:
                embedding = self.generate_embedding(text)
                embeddings.append(embedding)
                
            return np.array(embeddings)

    