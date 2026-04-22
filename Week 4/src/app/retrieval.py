from pathlib import Path
import time

from FlagEmbedding import BGEM3FlagModel
from munch import Munch

from database import retrieve_unified


def embed_query(
    model: BGEM3FlagModel,
    query: str,
    max_length: int,
) -> dict:
    return model.encode(
        [query],
        max_length=max_length,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=True,
    )


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict]],
    top_k: int,
    k: int = 60,
    normalize: bool = True,
) -> list[dict]:
    """
    Merge multiple ranked lists using Reciprocal Rank Fusion (RRF).

    RRF combines rankings from different retrieval methods by assigning scores
    based on rank positions. Chunks appearing high in multiple lists get higher scores.

    Formula: RRF_score(chunk) = Σ [1 / (k + rank)] across all lists

    Args:
        ranked_lists : List of ranked result lists from different retrieval methods.
                      Each inner list contains dicts with 'chunk_id' and 'chunk_text'.
        top_k        : Number of top results to return after fusion.
        k            : RRF smoothing constant (default=60). Higher values reduce
                      the difference between high and low ranks.
        normalize    : If True, normalize scores to 0-1 range (default=False).

    Returns:
        List of dicts with keys: chunk_id, chunk_text, rrf_score.
        Sorted by rrf_score in descending order.

    Example:
        >>> dense_results = [{"chunk_id": 1, "chunk_text": "..."}, ...]
        >>> sparse_results = [{"chunk_id": 2, "chunk_text": "..."}, ...]
        >>> colbert_results = [{"chunk_id": 1, "chunk_text": "..."}, ...]
        >>> fused = reciprocal_rank_fusion([dense_results, sparse_results, colbert_results], top_k=5)
    """
    # ── Step 1: Initialize score accumulators ────────────────────────────
    rrf_scores: dict[int, float] = {}  # chunk_id -> accumulated RRF score
    chunk_texts: dict[int, str] = {}  # chunk_id -> chunk text (for output)

    # ── Step 2: Accumulate RRF scores across all retrieval methods ───────
    for method_results in ranked_lists:
        # Process each chunk in this method's ranked list
        for rank, item in enumerate(method_results, start=1):
            chunk_id = item["chunk_id"]

            # Calculate RRF contribution: 1 / (k + rank)
            # - Rank 1 contributes most: 1/(k+1)
            # - Rank 10 contributes less: 1/(k+10)
            rrf_contribution = 1.0 / (k + rank)

            # Add this method's contribution to the chunk's total score
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + rrf_contribution

            # Store chunk text (same for all occurrences)
            chunk_texts[chunk_id] = item["chunk_text"]

    # ── Step 3: (Optional) Normalize scores to 0-1 range ─────────────────
    if normalize:
        # Maximum possible score: chunk appears at rank 1 in all lists
        max_possible_score = len(ranked_lists) * (1.0 / (k + 1))

        # Normalize all scores
        rrf_scores = {
            chunk_id: score / max_possible_score
            for chunk_id, score in rrf_scores.items()
        }

    # ── Step 4: Sort by RRF score (descending) and take top_k ────────────
    # Convert dict to list of (chunk_id, score) tuples
    sorted_chunks = sorted(
        rrf_scores.items(),
        key=lambda x: x[1],  # Sort by score
        reverse=True,  # Highest scores first
    )[:top_k]

    # ── Step 5: Format output ─────────────────────────────────────────────
    return [
        {"chunk_id": chunk_id, "chunk_text": chunk_texts[chunk_id], "rrf_score": score}
        for chunk_id, score in sorted_chunks
    ]


def ensemble_retrieve(
    query: str,
    model: BGEM3FlagModel,
    config: Munch,
    database_url: str,
    sql_dir: Path,
) -> list[dict]:
    """
    Embed query, run all three retrievers, and fuse results with RRF.

    Args:
        query        : User question string.
        model        : Loaded BGE-M3 embedding model.
        config       : App config (Munch).
        database_url : PostgreSQL connection string.
        sql_dir      : Path to SQL query files.

    Returns:
        Top-k fused chunks as a list of dicts with method info.
    """
    start_time = time.time()
    top_k = config.retrieval.top_k

    q_out = embed_query(
        model=model,
        query=query,
        max_length=config.embedding.max_length,
    )
    
    embed_time = time.time()

    # Get results from unified query (single DB call)
    dense_results, sparse_results, colbert_results = retrieve_unified(
        database_url=database_url,
        sql_dir=sql_dir,
        q_dense=q_out["dense_vecs"][0],
        q_sparse=q_out["lexical_weights"][0],
        q_colbert=q_out["colbert_vecs"][0],
        top_k=top_k,
    )

    retrieval_time = time.time()
    ranked_lists = [dense_results, sparse_results, colbert_results]

    # Get fused results
    fused_results = reciprocal_rank_fusion(ranked_lists=ranked_lists, top_k=top_k)
    
    fusion_time = time.time()
    
    print(f"Timing breakdown: embed={embed_time-start_time:.3f}s, retrieval={retrieval_time-embed_time:.3f}s, fusion={fusion_time-retrieval_time:.3f}s, total={fusion_time-start_time:.3f}s")

    # Add method information and fix score key
    all_results = []

    # Add individual method results with method info
    for result in dense_results:
        all_results.append(
            {
                "chunk_id": result["chunk_id"],
                "chunk_text": result["chunk_text"],
                "score": result["score"],
                "method": "dense",
            }
        )

    for result in sparse_results:
        all_results.append(
            {
                "chunk_id": result["chunk_id"],
                "chunk_text": result["chunk_text"],
                "score": result["score"],
                "method": "sparse",
            }
        )

    for result in colbert_results:
        all_results.append(
            {
                "chunk_id": result["chunk_id"],
                "chunk_text": result["chunk_text"],
                "score": result["score"],
                "method": "colbert",
            }
        )

    # Add RRF fused results
    for result in fused_results:
        all_results.append(
            {
                "chunk_id": result["chunk_id"],
                "chunk_text": result["chunk_text"],
                "score": result["rrf_score"],
                "method": "rrf",
            }
        )

    # Return top unique results, prioritizing RRF
    seen_chunks = set()
    final_results = []

    # First add RRF results
    for result in fused_results:
        if result["chunk_id"] not in seen_chunks:
            final_results.append(
                {
                    "chunk_id": result["chunk_id"],
                    "chunk_text": result["chunk_text"],
                    "score": result["rrf_score"],
                    "method": "rrf",
                }
            )
            seen_chunks.add(result["chunk_id"])

    return final_results[:top_k]
