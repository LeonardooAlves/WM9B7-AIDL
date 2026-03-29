"""
Minimal demo of text chunking and document creation with LangChain.
Demonstrates different chunking strategies on Alice's Adventures in Wonderland.
"""

from pathlib import Path
from munch import Munch
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
)
from langchain_core.documents import Document


def load_book_content(config):
    """Load book text from config-specified file."""
    book_path = Path(__file__).parents[2] / "assets" / config.chunking.book_file
    with open(book_path, "r", encoding="utf-8") as f:
        return f.read()


def create_documents_with_metadata(chunks, chunk_method):
    """Create LangChain documents with metadata."""
    documents = []
    for i, chunk in enumerate(chunks):
        doc = Document(
            page_content=chunk,
            metadata={
                "source": chunk_method,
                "chunk_id": i,
                "chunk_method": chunk_method,
                "chunk_size": len(chunk),
            },
        )
        documents.append(doc)
    return documents


def demo_chunking_strategies():
    """Demonstrate different text chunking approaches."""
    # 📂 Load config
    config_path = Path(__file__).parents[2] / "config" / "config.yaml"
    with open(config_path, "r") as f:
        config = Munch.fromYAML(f)

    # 📖 Load book content
    print(f"Loading {config.chunking.book_file}...")
    text = load_book_content(config)
    print(f"Original text length: {len(text):,} characters")
    print("=" * 60)

    # 🔄 Strategy 1: Recursive Character Splitter (recommended)
    print("\n1️⃣  Recursive Character Text Splitter")
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunking.recursive_chunk_size,
        chunk_overlap=config.chunking.recursive_chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    recursive_chunks = recursive_splitter.split_text(text)
    recursive_docs = create_documents_with_metadata(recursive_chunks, "recursive")

    print(f"Number of chunks: {len(recursive_chunks)}")
    print(
        "Average chunk size: "
        f"{sum(len(chunk) for chunk in recursive_chunks) / len(recursive_chunks):.0f}"
        " chars"
    )
    print(f"First chunk preview: {recursive_chunks[0][:150]}...")

    # 📄 Strategy 2: Character Text Splitter (simple)
    print("\n2️⃣  Character Text Splitter")
    char_splitter = CharacterTextSplitter(
        chunk_size=config.chunking.character_chunk_size,
        chunk_overlap=config.chunking.character_chunk_overlap,
        separator="\n\n",
        length_function=len,
    )
    char_chunks = char_splitter.split_text(text)
    char_docs = create_documents_with_metadata(char_chunks, "character")

    print(f"Number of chunks: {len(char_chunks)}")
    print(
        "Average chunk size: "
        f"{sum(len(chunk) for chunk in char_chunks) / len(char_chunks):.0f} chars"
    )
    print(f"First chunk preview: {char_chunks[0][:150]}...")

    # 🔤 Strategy 3: Token Text Splitter (token-aware)
    print("\n3️⃣  Token Text Splitter")
    token_splitter = TokenTextSplitter(
        chunk_size=config.chunking.token_chunk_size,
        chunk_overlap=config.chunking.token_chunk_overlap,
    )
    token_chunks = token_splitter.split_text(text)
    token_docs = create_documents_with_metadata(token_chunks, "token")

    print(f"Number of chunks: {len(token_chunks)}")
    print(
        "Average chunk size: "
        f"{sum(len(chunk) for chunk in token_chunks) / len(token_chunks):.0f} chars"
    )
    print(f"First chunk preview: {token_chunks[0][:150]}...")

    # 📊 Strategy Comparison
    print("\n" + "=" * 60)
    print("📊 Chunking Strategy Comparison")
    print("=" * 60)
    strategies = [
        ("Recursive Character", recursive_docs),
        ("Character", char_docs),
        ("Token", token_docs),
    ]

    for name, docs in strategies:
        chunk_sizes = [doc.metadata["chunk_size"] for doc in docs]
        print(
            f"{name:20} | {len(docs):4} chunks | "
            f"Avg: {sum(chunk_sizes) / len(chunk_sizes):6.0f} | "
            f"Min: {min(chunk_sizes):4} | Max: {max(chunk_sizes):4}"
        )

    # 🔍 Sample Document Analysis
    print("\n📋 Sample Document Metadata:")
    sample_doc = recursive_docs[5]
    print(f"Content: {sample_doc.page_content[:100]}...")
    print(f"Metadata: {sample_doc.metadata}")

    return {
        "recursive": recursive_docs,
        "character": char_docs,
        "token": token_docs,
    }


if __name__ == "__main__":
    chunked_documents = demo_chunking_strategies()
    print(
        f"\n✅ Created {sum(len(docs) for docs in chunked_documents.values())}"
        " total documents"
    )
