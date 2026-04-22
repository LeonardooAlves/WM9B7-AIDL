from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from munch import Munch


SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using only the provided context.\n"
    "If the answer is not in the context, say \"I don't know based on the provided context.\"\n"
)


def build_rag_chain(config: Munch, groq_api_key: str):
    """
    Build and return the LangChain LCEL RAG chain.

    Chain shape:
        { context, question } | prompt | llm | StrOutputParser

    Args:
        config       : App config (Munch).
        groq_api_key : Groq API key string.

    Returns:
        A compiled LangChain Runnable.
    """
    llm = ChatGroq(
        api_key=groq_api_key,
        model=config.groq.model,
        temperature=config.groq.temperature,
        max_tokens=config.groq.max_tokens,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("user", "Context:\n{context}\n\nQuestion: {question}"),
    ])

    chain = (
        {
            "context": RunnableLambda(lambda x: x["context"]),
            "question": RunnablePassthrough() | RunnableLambda(lambda x: x["question"]),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


def format_context(chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a single context string for the prompt.

    Args:
        chunks : List of chunk dicts with chunk_text key.

    Returns:
        Formatted context string.
    """
    return "\n\n---\n\n".join(
        f"[Chunk {i + 1}]\n{chunk['chunk_text']}"
        for i, chunk in enumerate(chunks)
    )
