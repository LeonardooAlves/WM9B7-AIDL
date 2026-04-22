SELECT
    d.chunk_id,
    d.chunk_text,
    COALESCE(SUM(
        (q.weight::float) * (d.lexical_weights ->> q.token)::float
    ), 0) AS score
FROM
    embeddings_sparse d,
    jsonb_each_text(%s::jsonb) AS q(token, weight)
WHERE d.lexical_weights ? q.token
GROUP BY d.chunk_id, d.chunk_text
ORDER BY score DESC
LIMIT %s;