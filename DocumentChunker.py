from typing import List

class DocumentChunker:
    """
    Strategy 1: Character-based chunking with overlap.
    Splits text into fixed-size character windows.
    
    Pro: Simple, consistent chunk sizes
    Con: Cuts mid-sentence/word, losing semantic coherence
    """
    
    def __init__(self, chunk_size: int = 500, overlap: int = 30):
        """
        Args:
            chunk_size: Number of characters per chunk
            overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        
    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping character-level chunks"""
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


class ParagraphChunker:
    """
    Strategy 2: Paragraph-based chunking.
    Splits text on blank lines (\\n\\n), keeping each paragraph intact.
    Merges small paragraphs together up to max_chunk_size.

    Pro: Semantically coherent chunks, no mid-sentence cuts
    Con: Variable chunk sizes, very long paragraphs won't be split
    """

    def __init__(self, max_chunk_size: int = 1000, min_chunk_size: int = 50):
        """
        Args:
            max_chunk_size: Merge paragraphs together until this char limit
            min_chunk_size: Skip paragraphs shorter than this (e.g. blank headers)
        """
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size

    def chunk_text(self, text: str) -> List[str]:
        """Split text into paragraph-based chunks"""
        # Split on blank lines
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        # Filter out very short paragraphs (e.g. lone titles with no content)
        paragraphs = [p for p in paragraphs if len(p) >= self.min_chunk_size]

        chunks = []
        current_chunk = ""

        for paragraph in paragraphs:
            # If adding this paragraph exceeds max size, save current and start fresh
            if current_chunk and len(current_chunk) + len(paragraph) + 2 > self.max_chunk_size:
                chunks.append(current_chunk.strip())
                current_chunk = paragraph
            else:
                current_chunk = current_chunk + "\n\n" + paragraph if current_chunk else paragraph

        # Don't forget the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks