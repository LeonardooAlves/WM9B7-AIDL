"""
Minimal demo of BAAI/bge-m3 model's three embedding types with simple retrieval.
Demonstrates dense, sparse (lexical), and ColBERT (multi-vector) embeddings.
"""

import numpy as np
from FlagEmbedding import BGEM3FlagModel
from sklearn.metrics.pairwise import cosine_similarity


def demo_bge_m3_embeddings():
    # Sample document collection
    documents = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with multiple layers.",
        "Natural language processing helps computers understand human language.",
        "Computer vision enables machines to interpret visual information.",
        "Reinforcement learning trains agents through rewards and penalties.",
    ]

    # Initialize BGE-M3 model
    print("Loading BAAI/bge-m3 model...")
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

    # Get all three types of embeddings
    print("\nGenerating embeddings...")
    embeddings = model.encode(
        documents,
        batch_size=8,
        max_length=512,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=True,
    )

    # Extract different embedding types
    dense_embeddings = embeddings["dense_vecs"]
    sparse_embeddings = embeddings["lexical_weights"]
    colbert_embeddings = embeddings["colbert_vecs"]

    print(f"Dense embeddings shape: {dense_embeddings.shape}")
    print(f"Number of sparse embeddings: {len(sparse_embeddings)}")
    print(f"Number of ColBERT embeddings: {len(colbert_embeddings)}")

    # Query for retrieval
    query = "What is deep learning?"
    print(f"\nQuery: {query}")

    # Get query embeddings
    query_embeddings = model.encode(
        [query],
        batch_size=1,
        max_length=512,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=True,
    )

    # Dense retrieval
    print("\n--- Dense Embedding Retrieval ---")
    dense_query = query_embeddings["dense_vecs"]
    dense_scores = cosine_similarity(dense_query, dense_embeddings)[0]
    dense_ranked = np.argsort(dense_scores)[::-1]

    for i, idx in enumerate(dense_ranked[:3]):
        print(f"{i+1}. Score: {dense_scores[idx]:.4f} - {documents[idx]}")

    # Sparse retrieval (simplified lexical matching)
    print("\n--- Sparse Embedding Retrieval ---")
    query_sparse = query_embeddings["lexical_weights"][0]
    sparse_scores = []

    for doc_sparse in sparse_embeddings:
        # Calculate overlap score between query and document sparse representations
        common_tokens = set(query_sparse.keys()) & set(doc_sparse.keys())
        score = sum(query_sparse[token] * doc_sparse[token] for token in common_tokens)
        sparse_scores.append(score)

    sparse_ranked = np.argsort(sparse_scores)[::-1]

    for i, idx in enumerate(sparse_ranked[:3]):
        print(f"{i+1}. Score: {sparse_scores[idx]:.4f} - {documents[idx]}")

    # ColBERT retrieval (MaxSim)
    print("\n--- ColBERT Embedding Retrieval ---")
    query_colbert = query_embeddings["colbert_vecs"][0]
    colbert_scores = []

    for doc_colbert in colbert_embeddings:
        # MaxSim: for each query token, find max similarity with any document token
        similarities = np.dot(query_colbert, doc_colbert.T)
        maxsim_score = np.mean(np.max(similarities, axis=1))
        colbert_scores.append(maxsim_score)

    colbert_ranked = np.argsort(colbert_scores)[::-1]

    for i, idx in enumerate(colbert_ranked[:3]):
        print(f"{i+1}. Score: {colbert_scores[idx]:.4f} - {documents[idx]}")


if __name__ == "__main__":
    demo_bge_m3_embeddings()
