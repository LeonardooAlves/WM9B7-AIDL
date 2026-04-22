"""
Minimal Supabase PostgreSQL demo with pgvector for embeddings comparison.
"""

import os
import psycopg2
import numpy as np
from dotenv import load_dotenv
from pathlib import Path
from munch import Munch

# 📂 Load config
config_path = Path(__file__).parents[2] / "config" / "config.yaml"
with open(config_path, "r") as f:
    config = Munch.fromYAML(f)

# 📂 Load .env file
env_path = Path(__file__).parents[2] / ".env"
load_dotenv(env_path)

# 🔑 Get database password
SUPABASE_PASSWORD = os.getenv("SUPABASE_PASSWORD")

if not SUPABASE_PASSWORD:
    raise ValueError("SUPABASE_PASSWORD not found in .env file")

# 🔗 Build database URL from config
DATABASE_URL = (
    f"postgresql://{config.supabase.user}:{SUPABASE_PASSWORD}@"
    f"{config.supabase.host}:{config.supabase.port}/{config.supabase.database}"
)


def connect_db():
    """Create database connection"""
    return psycopg2.connect(DATABASE_URL)


def setup_pgvector():
    """Enable pgvector extension and create demo table"""
    conn = connect_db()
    cur = conn.cursor()

    try:
        # Enable pgvector extension
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        # Create demo table for embeddings
        cur.execute(
            """
        CREATE TABLE IF NOT EXISTS documents (
            id SERIAL PRIMARY KEY,
            text TEXT NOT NULL,
            embedding vector(1024)
        );
        """
        )

        conn.commit()
        print("✅ pgvector extension enabled and table created")

    except Exception as e:
        print(f"❌ Error setting up pgvector: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()


def insert_document(text, embedding):
    """Insert document with embedding"""
    conn = connect_db()
    cur = conn.cursor()

    try:
        # Convert numpy array to list for PostgreSQL
        embedding_list = embedding.tolist()

        cur.execute(
            """
        INSERT INTO documents (text, embedding) 
        VALUES (%s, %s) RETURNING id;
        """,
            (text, embedding_list),
        )

        doc_id = cur.fetchone()[0]
        conn.commit()
        print(f"✅ Document inserted with ID: {doc_id}")
        return doc_id

    except Exception as e:
        print(f"❌ Error inserting document: {e}")
        conn.rollback()
        return None
    finally:
        cur.close()
        conn.close()


def find_similar_documents(query_embedding, limit=3):
    """Find similar documents using cosine similarity"""
    conn = connect_db()
    cur = conn.cursor()

    try:
        # Convert numpy array to list
        query_list = query_embedding.tolist()

        # Use pgvector cosine distance (1 - cosine similarity)
        cur.execute(
            """
        SELECT id, text, 1 - (embedding <=> %s::vector) as similarity
        FROM documents
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
        """,
            (query_list, query_list, limit),
        )

        results = cur.fetchall()
        return results

    except Exception as e:
        print(f"❌ Error finding similar documents: {e}")
        return []
    finally:
        cur.close()
        conn.close()


def demo():
    """Run interactive demo"""
    print("🚀 Supabase PostgreSQL + pgvector Demo")
    print("=====================================")

    # Setup
    setup_pgvector()

    # Sample documents and embeddings (random for demo)
    sample_docs = [
        ("Machine learning is transforming technology", np.random.rand(1024)),
        ("Deep learning uses neural networks", np.random.rand(1024)),
        ("PostgreSQL is a powerful database", np.random.rand(1024)),
        ("Vector databases enable semantic search", np.random.rand(1024)),
    ]

    # Insert sample documents
    print("\n📝 Inserting sample documents...")
    for text, embedding in sample_docs:
        insert_document(text, embedding)

    # Interactive query
    while True:
        print("\n" + "=" * 50)
        print("Options:")
        print("1. Find similar documents (random query)")
        print("2. Insert new document")
        print("3. Exit")

        choice = input("\nEnter choice (1-3): ").strip()

        if choice == "1":
            # Generate random query embedding
            query_embedding = np.random.rand(1024)
            print("\n🔍 Finding similar documents...")

            results = find_similar_documents(query_embedding)

            print("\nTop similar documents:")
            for doc_id, text, similarity in results:
                print(f"ID {doc_id}: {text[:50]}... (similarity: {similarity:.4f})")

        elif choice == "2":
            text = input("Enter document text: ").strip()
            if text:
                # Generate random embedding for demo
                embedding = np.random.rand(1024)
                insert_document(text, embedding)

        elif choice == "3":
            print("👋 Goodbye!")
            break

        else:
            print("❌ Invalid choice")


if __name__ == "__main__":
    demo()
