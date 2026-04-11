-- ColBERT (multi-vector) embeddings table.
--
-- Schema: one row per TOKEN, not one row per chunk.
-- Each chunk's token matrix (n_tokens, 1024) is exploded into n_tokens rows,
-- each linked back to its chunk via chunk_id and ordered by token_index.
--
-- Dimension choice: vector(1024) for full BGE-M3 embeddings
-- Using full 1024-dimensional vectors from BGE-M3 model to maintain
-- maximum embedding quality and compatibility with the notebook implementation.
--
-- Why token-per-row instead of storing a matrix blob?
-- Individual vector(1024) rows can be HNSW-indexed, making ColBERT a
-- first-pass ANN retriever — not just a re-ranker over a dense shortlist.
-- At query time: embed query with ColBERT → ANN search for similar tokens
-- → group by chunk_id → compute MaxSim score in Python.

DROP TABLE IF EXISTS embeddings_colbert;

CREATE TABLE embeddings_colbert (
    id           SERIAL PRIMARY KEY,
    chunk_id     INTEGER NOT NULL,
    token_index  INTEGER NOT NULL,
    chunk_text   TEXT    NOT NULL,
    token_vector vector(1024) NOT NULL,
    UNIQUE (chunk_id, token_index)
);