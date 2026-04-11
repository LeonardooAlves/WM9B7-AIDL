from pathlib import Path

from FlagEmbedding import BGEM3FlagModel
from munch import Munch

from database import retrieve_colbert, retrieve_dense, retrieve_sparse


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
) -> list[dict]:
    """
    Merge multiple ranked lists using Reciprocal Rank Fusion.

    Args:
        ranked_lists : Each list is a ranked set of retrieval results.
        top_k        : Number of results to return.
        k            : RRF smoothing constant.

    Returns:
        Fused list of dicts with keys chunk_id, chunk_text, rrf_score.
    """
    rrf_scores: dict = {}
    chunk_texts: dict = {}

    for results in ranked_lists:
        for rank, item in enumerate(results, start=1):
            cid = item["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank)
            chunk_texts[cid] = item["chunk_text"]

    fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [
        {"chunk_id": cid, "chunk_text": chunk_texts[cid], "rrf_score": score}
        for cid, score in fused
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
    top_k = config.retrieval.top_k

    q_out = embed_query(
        model=model,
        query=query,
        max_length=config.embedding.max_length,
    )

    # Get results from each method
    dense_results = retrieve_dense(
        database_url=database_url,
        sql_dir=sql_dir,
        q_dense=q_out["dense_vecs"][0],
        top_k=top_k,
    )
    
    sparse_results = retrieve_sparse(
        database_url=database_url,
        sql_dir=sql_dir,
        q_sparse=q_out["lexical_weights"][0],
        top_k=top_k,
    )
    
    colbert_results = retrieve_colbert(
        database_url=database_url,
        sql_dir=sql_dir,
        q_colbert=q_out["colbert_vecs"][0],
        top_k=top_k,
    )

    ranked_lists = [dense_results, sparse_results, colbert_results]
    
    # Get fused results
    fused_results = reciprocal_rank_fusion(ranked_lists=ranked_lists, top_k=top_k)
    
    # Add method information and fix score key
    all_results = []
    
    # Add individual method results with method info
    for result in dense_results:
        all_results.append({
            "chunk_id": result["chunk_id"],
            "chunk_text": result["chunk_text"],
            "score": result["score"],
            "method": "dense"
        })
    
    for result in sparse_results:
        all_results.append({
            "chunk_id": result["chunk_id"],
            "chunk_text": result["chunk_text"],
            "score": result["score"],
            "method": "sparse"
        })
    
    for result in colbert_results:
        all_results.append({
            "chunk_id": result["chunk_id"],
            "chunk_text": result["chunk_text"],
            "score": result["score"],
            "method": "colbert"
        })
    
    # Add RRF fused results
    for result in fused_results:
        all_results.append({
            "chunk_id": result["chunk_id"],
            "chunk_text": result["chunk_text"],
            "score": result["rrf_score"],
            "method": "rrf"
        })
    
    # Return top unique results, prioritizing RRF
    seen_chunks = set()
    final_results = []
    
    # First add RRF results
    for result in fused_results:
        if result["chunk_id"] not in seen_chunks:
            final_results.append({
                "chunk_id": result["chunk_id"],
                "chunk_text": result["chunk_text"],
                "score": result["rrf_score"],
                "method": "rrf"
            })
            seen_chunks.add(result["chunk_id"])
    
    return final_results[:top_k]
