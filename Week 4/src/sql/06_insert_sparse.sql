-- Insert sparse embedding rows via psycopg2 execute_values.
INSERT INTO embeddings_sparse (chunk_id, chunk_text, lexical_weights)
VALUES %s
ON CONFLICT (chunk_id) DO NOTHING;