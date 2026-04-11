-- Enable the pgvector extension.
-- This only needs to run once per database.
-- Safe to re-run: CREATE EXTENSION IF NOT EXISTS is idempotent.
CREATE EXTENSION IF NOT EXISTS vector;