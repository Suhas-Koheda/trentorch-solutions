import numpy as np


def retrieval_cosine_gate(q: np.ndarray, documents: np.ndarray) -> np.ndarray:
    """
    query: array of shape (d,).
    documents: array of shape (n, d).

    Returns an array of shape (n,): cosine similarity between query and
    each document row. If either the query or a document has zero norm,
    that document's similarity is 0.
    """
    # TODO: compute dot(query, doc) / (||query|| * ||doc||) per document,
    # guarding against zero norms.
    results = [
        q.dot(d) / (np.linalg.norm(q) * np.linalg.norm(d))
        if np.linalg.norm(q) != 0 and np.linalg.norm(d) != 0
        else 0.0
        for d in documents
    ]
    return np.array(results)
