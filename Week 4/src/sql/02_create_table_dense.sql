-- Dense embeddings table.
--
-- Each chunk gets a single fixed-length vector (1024 dims from bge-m3).
-- The HNSW index makes approximate nearest-neighbour search fast at scale;
-- without it every query would scan all rows (O(N) — too slow for large corpora).
--
-- Cosine similarity is the standard metric for normalised text embeddings:
--   1 - (embedding <=> query_vec)  →  1.0 = identical, 0.0 = orthogonal

DROP TABLE IF EXISTS embeddings_dense;

CREATE TABLE embeddings_dense (
    id SERIAL PRIMARY KEY,
    chunk_id INTEGER NOT NULL UNIQUE, -- matches index in chunks list
    chunk_text TEXT NOT NULL,
    embedding vector (1024) NOT NULL
);

-- HNSW index for cosine similarity search.
-- m        : number of neighbours per node in the graph (higher = more accurate, more memory)
-- ef_construction : beam width during index build (higher = better recall, slower build)
CREATE INDEX idx_dense_embedding_hnsw ON embeddings_dense USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);