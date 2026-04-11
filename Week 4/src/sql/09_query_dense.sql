-- Dense retrieval: cosine similarity via the <=> operator.
-- 1 - distance converts cosine distance to cosine similarity.
SELECT
    chunk_id,
    chunk_text,
    1 - (embedding <=> %s::vector) AS score
FROM embeddings_dense
ORDER BY embedding <=> %s::vector
LIMIT %s;