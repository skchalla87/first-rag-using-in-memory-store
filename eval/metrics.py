"""
Individual metric functions for RAG evaluation.
All functions return a float between 0.0 and 1.0.
"""

def context_precision(retrieved_sources: list[str], relevant_sources: list[str]) -> float:
    """
    Of the chunks retrieved, what fraction are actually relevant?
    High precision = low noise sent to the LLM.

    Example: retrieved=["cap_theorem", "gossip", "replication"], relevant=["cap_theorem"]
    → 1/3 = 0.33
    retrieved_sources = what your RAG actually returned
    relevant_sources  = what the correct answer key says should appear
    intersection      = the overlap = correctly retrieved                       
    """
    if not retrieved_sources:
        return 0.0
    
    if not relevant_sources:
        return 1.0 # out of context question, nothing relevant exists, so nothing irrelevant retrieved
    
    relevant_set = set(relevant_sources)
    retrieved_set = set(retrieved_sources)
    
    return len(relevant_set & retrieved_set) / len(retrieved_set)

def context_recall(retrieved_sources: list[str], relevant_sources: list[str]) -> float:
    """
    Of all relevant docs that exist, what fraction did we retrieve?
    High recall = we didn't miss critical information.

    Example: retrieved=["cap_theorem"], relevant=["cap_theorem", "consensus_algorithms"]
    → 1/2 = 0.50
    """
    if not relevant_sources:
        return 1.0  # out-of-corpus question: nothing to recall

    retrieved_set = set(retrieved_sources)
    relevant_set = set(relevant_sources)
    return len(retrieved_set & relevant_set) / len(relevant_set)


def hit_rate(retrieved_sources: list[str], relevant_sources: list[str]) -> float:
    """
    Did at least one relevant doc appear anywhere in the retrieved set?
    Binary: 1.0 (hit) or 0.0 (miss).

    Useful because precision/recall can look ok even when the top result is wrong.
    """
    if not relevant_sources:
        return 1.0  # out-of-corpus: no relevant docs to miss

    return 1.0 if set(retrieved_sources) & set(relevant_sources) else 0.0


def mrr(retrieved_sources: list[str], relevant_sources: list[str]) -> float:
    """
    Mean Reciprocal Rank — rewards finding the right doc early.
    Score = 1/rank of the first relevant result.

    Rank 1 → 1.0  (best)
    Rank 2 → 0.5
    Rank 3 → 0.33
    Not found → 0.0

    Example: retrieved=["gossip", "cap_theorem", "replication"], relevant=["cap_theorem"]
    → cap_theorem is at rank 2, so MRR = 0.5
    """
    if not relevant_sources:
        return 1.0  # out-of-corpus: no relevant docs to rank

    relevant_set = set(relevant_sources)
    for rank, source in enumerate(retrieved_sources, start=1):
        if source in relevant_set:
            return 1.0 / rank
    return 0.0

def answer_faithfulness(answer: str, retrieved_chunks: list[str]) -> float:
    """
    Are the key words in the answer actually present in the retrieved context?
    Approximates: did the LLM answer from the context, or did it hallucinate?

    Method: extract content words (len > 5) from the answer,
    check what fraction appear in the combined context text.

    This is a simple proxy — not perfect, but useful without an LLM-as-judge.
    """
    
    if not answer or not retrieved_chunks:
        return 0.0
    
    context_text = " ".join(retrieved_chunks).lower()
    words_in_answer = set(answer.lower().split())
    
    # skip the short/stopwords
    content_words = {w for w in words_in_answer if len(w) > 5}
    
    if not content_words:
        return 1.0 # answer has no content words to check
    
    supported = sum(1 for w in content_words if w in context_text)
    
    return supported / len(content_words)


def refusal_correctness(answer: str, expected_answer) -> float:
    """
    For out-of-corpus questions (expected_answer=None):
    Did the system correctly refuse to answer?

    Returns 1.0 if it refused, 0.0 if it hallucinated an answer.
    """
    refused = "don't have information" in answer.lower() or \
              "not in the provided" in answer.lower() or \
              "cannot find" in answer.lower()

    if expected_answer is None:
        # We wanted a refusal
        return 1.0 if refused else 0.0
    else:
        # We wanted an actual answer, not a refusal
        return 0.0 if refused else 1.0