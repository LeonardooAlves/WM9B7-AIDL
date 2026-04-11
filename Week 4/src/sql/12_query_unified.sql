-- Unified query that retrieves results from all three embedding methods
-- Returns results with method labels for RRF fusion
WITH dense_results AS (
    SELECT
        chunk_id,
        chunk_text,
        1 - (embedding <=> %s::vector) AS score,
        'dense' AS method,
        ROW_NUMBER() OVER (ORDER BY embedding <=> %s::vector) AS rank
    FROM embeddings_dense
    ORDER BY embedding <=> %s::vector
    LIMIT %s
),
sparse_results AS (
    SELECT
        d.chunk_id,
        d.chunk_text,
        COALESCE(SUM(
            (q.weight::float) * (d.lexical_weights ->> q.token)::float
        ), 0) AS score,
        'sparse' AS method,
        ROW_NUMBER() OVER (ORDER BY COALESCE(SUM(
            (q.weight::float) * (d.lexical_weights ->> q.token)::float
        ), 0) DESC) AS rank
    FROM
        embeddings_sparse d,
        jsonb_each_text(%s::jsonb) AS q(token, weight)
    WHERE d.lexical_weights ? q.token
    GROUP BY d.chunk_id, d.chunk_text
    ORDER BY score DESC
    LIMIT %s
),
colbert_results AS (
    SELECT 
        chunk_id,
        chunk_text,
        MAX(1 - (token_vector <=> %s::vector)) AS score,
        'colbert' AS method,
        ROW_NUMBER() OVER (ORDER BY MAX(1 - (token_vector <=> %s::vector)) DESC) AS rank
    FROM embeddings_colbert
    GROUP BY chunk_id, chunk_text
    ORDER BY score DESC
    LIMIT %s
)
SELECT chunk_id, chunk_text, score, method, rank
FROM dense_results
UNION ALL
SELECT chunk_id, chunk_text, score, method, rank
FROM sparse_results
UNION ALL
SELECT chunk_id, chunk_text, score, method, rank
FROM colbert_results
ORDER BY method, rank;