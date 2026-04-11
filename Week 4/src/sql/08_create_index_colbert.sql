CREATE INDEX idx_colbert_token_hnsw
    ON embeddings_colbert
    USING hnsw (token_vector vector_cosine_ops)
    WITH (m = 8, ef_construction = 32);  -- Reduced from m=16, ef_construction=64 for faster creation