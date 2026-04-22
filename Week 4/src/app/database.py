import json

import numpy as np
import psycopg2
from munch import Munch
from pathlib import Path


def build_database_url(config: Munch, password: str) -> str:
    return (
        f"postgresql://{config.supabase.user}:{password}@"
        f"{config.supabase.host}:{config.supabase.port}/{config.supabase.database}"
    )


def get_connection(database_url: str) -> psycopg2.extensions.connection:
    return psycopg2.connect(database_url)


def retrieve_dense(
    database_url: str,
    sql_dir: Path,
    q_dense: np.ndarray,
    top_k: int,
) -> list[dict]:
    sql = (sql_dir / "09_query_dense.sql").read_text()
    vec_str = f"[{','.join(map(str, q_dense.tolist()))}]"
    conn = get_connection(database_url=database_url)
    cur = conn.cursor()
    try:
        cur.execute(sql, (vec_str, vec_str, top_k))
        return [
            {"chunk_id": r[0], "chunk_text": r[1], "score": float(r[2])}
            for r in cur.fetchall()
        ]
    finally:
        cur.close()
        conn.close()


def retrieve_sparse(
    database_url: str,
    sql_dir: Path,
    q_sparse: dict,
    top_k: int,
) -> list[dict]:
    sql = (sql_dir / "10_query_sparse.sql").read_text()
    sparse_json = json.dumps({str(k): float(v) for k, v in q_sparse.items()})
    conn = get_connection(database_url=database_url)
    cur = conn.cursor()
    try:
        cur.execute(sql, (sparse_json, top_k))
        return [
            {"chunk_id": r[0], "chunk_text": r[1], "score": float(r[2])}
            for r in cur.fetchall()
        ]
    finally:
        cur.close()
        conn.close()


def retrieve_colbert(
    database_url: str,
    sql_dir: Path,
    q_colbert: np.ndarray,
    top_k: int,
) -> list[dict]:
    sql = (sql_dir / "11_query_colbert.sql").read_text()
    candidate_pool = top_k * 3
    chunk_scores: dict = {}
    conn = get_connection(database_url=database_url)
    cur = conn.cursor()
    try:
        for token_vec in q_colbert:
            vec_str = f"[{','.join(map(str, token_vec.tolist()))}]"
            cur.execute(sql, (vec_str, candidate_pool))
            for chunk_id, chunk_text, max_sim in cur.fetchall():
                entry = chunk_scores.setdefault(
                    chunk_id, {"chunk_text": chunk_text, "sims": []}
                )
                entry["sims"].append(float(max_sim))
    finally:
        cur.close()
        conn.close()

    results = [
        {
            "chunk_id": cid,
            "chunk_text": d["chunk_text"],
            "score": float(np.mean(d["sims"])),
        }
        for cid, d in chunk_scores.items()
    ]
    return sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]


def retrieve_unified(
    database_url: str,
    sql_dir: Path,
    q_dense: np.ndarray,
    q_sparse: dict,
    q_colbert: np.ndarray,
    top_k: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Single SQL query that retrieves from all three methods at once.
    Returns tuple of (dense_results, sparse_results, colbert_results).
    """
    sql = (sql_dir / "12_query_unified.sql").read_text()
    
    # Prepare parameters
    dense_vec_str = f"[{','.join(map(str, q_dense.tolist()))}]"
    sparse_json = json.dumps({str(k): float(v) for k, v in q_sparse.items()})
    # Use first token for ColBERT (simplified for now)
    colbert_vec_str = f"[{','.join(map(str, q_colbert[0].tolist()))}]"
    
    conn = get_connection(database_url=database_url)
    cur = conn.cursor()
    try:
        cur.execute(sql, (
            dense_vec_str, dense_vec_str, dense_vec_str, top_k,  # dense params
            sparse_json, top_k,  # sparse params  
            colbert_vec_str, colbert_vec_str, top_k  # colbert params
        ))
        
        all_results = cur.fetchall()
        
        dense_results = []
        sparse_results = []
        colbert_results = []
        
        for chunk_id, chunk_text, score, method, rank in all_results:
            result = {"chunk_id": chunk_id, "chunk_text": chunk_text, "score": float(score)}
            
            if method == "dense":
                dense_results.append(result)
            elif method == "sparse":
                sparse_results.append(result)
            elif method == "colbert":
                colbert_results.append(result)
                
        return dense_results, sparse_results, colbert_results
        
    finally:
        cur.close()
        conn.close()
