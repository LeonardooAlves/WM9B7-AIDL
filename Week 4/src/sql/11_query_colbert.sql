-- Simple ColBERT query - finds max similarity per chunk for a single query token
SELECT 
    chunk_id,
    chunk_text,
    MAX(1 - (token_vector <=> %s::vector)) AS max_sim
FROM embeddings_colbert
GROUP BY chunk_id, chunk_text
ORDER BY max_sim DESC
LIMIT %s;