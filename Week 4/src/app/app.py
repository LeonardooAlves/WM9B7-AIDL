import os
from pathlib import Path

from dotenv import load_dotenv
from FlagEmbedding import BGEM3FlagModel
from flask import Flask, Response, jsonify, render_template, request, stream_with_context
from munch import Munch

from database import build_database_url
from generation import build_rag_chain, format_context
from retrieval import ensemble_retrieve


# ── Paths ─────────────────────────────────────────────────────────────────────

APP_DIR = Path(__file__).parent
CONFIG_PATH = (APP_DIR / "../../config/config.yaml").resolve()
SQL_DIR = (APP_DIR / "../sql").resolve()
ENV_PATH = (APP_DIR / "../../.env").resolve()

# ── Config & secrets ──────────────────────────────────────────────────────────

with open(CONFIG_PATH, "r") as f:
    config = Munch.fromYAML(f)

load_dotenv(ENV_PATH)

SUPABASE_PASSWORD = os.getenv("SUPABASE_PASSWORD")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not SUPABASE_PASSWORD:
    raise ValueError("SUPABASE_PASSWORD not found in .env")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env")

DATABASE_URL = build_database_url(config=config, password=SUPABASE_PASSWORD)

# ── Model & chain (loaded once at startup) ────────────────────────────────────

print(f"Loading {config.embedding.model} …")
embed_model = BGEM3FlagModel(config.embedding.model, use_fp16=True)
print("✅ Embedding model loaded")

rag_chain = build_rag_chain(config=config, groq_api_key=GROQ_API_KEY)
print("✅ RAG chain ready")

# ── Flask app ─────────────────────────────────────────────────────────────────

app = Flask(__name__)


@app.route("/")
def index() -> str:
    book_name = Path(config.chunking.book_file).stem
    return render_template("index.html", book_name=book_name)


@app.route("/ask", methods=["POST"])
def ask() -> Response:
    """
    Accept a JSON body { "question": "..." } and stream the answer back
    as Server-Sent Events so the UI can render tokens as they arrive.
    """
    data = request.get_json(silent=True) or {}
    question: str = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "No question provided."}), 400

    def generate():
        # 1. Retrieve context
        chunks = ensemble_retrieve(
            query=question,
            model=embed_model,
            config=config,
            database_url=DATABASE_URL,
            sql_dir=SQL_DIR,
        )
        context = format_context(chunks=chunks)

        # 2. Send chunks first as a special event
        import json
        chunks_data = [
            {
                "chunk_id": chunk["chunk_id"],
                "chunk_text": chunk["chunk_text"],
                "score": chunk["score"],
                "method": chunk["method"]
            }
            for chunk in chunks
        ]
        yield f"event: chunks\ndata: {json.dumps(chunks_data)}\n\n"

        # 3. Stream the LLM response token by token
        for token in rag_chain.stream({"question": question, "context": context}):
            yield f"data: {token}\n\n"

        yield "data: [DONE]\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
