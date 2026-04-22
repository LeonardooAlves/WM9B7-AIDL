-- Insert dense embedding rows via psycopg2 execute_values.
INSERT INTO embeddings_dense (chunk_id, chunk_text, embedding)
VALUES %s
ON CONFLICT (chunk_id) DO NOTHING;