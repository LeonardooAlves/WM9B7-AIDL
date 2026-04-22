-- Insert ColBERT token rows via psycopg2 execute_values.
INSERT INTO embeddings_colbert (chunk_id, token_index, chunk_text, token_vector)
VALUES %s
ON CONFLICT (chunk_id, token_index) DO NOTHING;