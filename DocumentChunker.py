class DocumentChunker:
    """Splits Documents into manageable chunks for better preservation"""
    
    def __init__(self, chunk_size: int =500, overlap: int=30):
        """
        Args:
            chunk_size: Number of characters per chunk
            overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        
    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            
            # only add non-empty chunks
            if chunk.strip():
                chunks.append(chunk.strip())
                
            start += self.chunk_size - self.overlap
            
        return chunks