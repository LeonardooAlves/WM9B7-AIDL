-- Sparse (lexical) embeddings table.
--
-- bge-m3 sparse embeddings are dictionaries: {token_id: importance_weight}.
-- Most tokens have zero weight for any given chunk, so storing a full 30k-dim
-- float array would waste huge amounts of space. jsonb stores only non-zero
-- entries and compresses them efficiently.
--
-- jsonb also supports GIN (Generalised Inverted Index) indexing, which enables
-- fast lookups like "find all chunks that contain token 12345" — the same
-- operation performed during sparse retrieval.

DROP TABLE IF EXISTS embeddings_sparse;

CREATE TABLE embeddings_sparse (
    id SERIAL PRIMARY KEY,
    chunk_id INTEGER NOT NULL UNIQUE,
    chunk_text TEXT NOT NULL,
    lexical_weights jsonb NOT NULL -- {"token_id": weight, ...}
);

-- GIN index for fast key-existence queries on the jsonb column.
CREATE INDEX idx_sparse_weights_gin ON embeddings_sparse USING gin (lexical_weights);