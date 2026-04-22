import os
from dotenv import load_dotenv
from pathlib import Path
from munch import Munch
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 📂 Load config
config_path = Path(__file__).parents[2] / "config" / "config.yaml"
with open(config_path, "r") as f:
    config = Munch.fromYAML(f)

# 📂 Load .env file
load_dotenv("Week 4/.env")

# 🔑 Get API key
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env file")

# 🚀 Create LangChain LLM
llm = ChatGroq(
    api_key=api_key,
    model=config.groq.model,
    temperature=config.groq.temperature,
    max_tokens=config.groq.max_tokens,
)

# 🎭 Step 1: Tone Adjustment Prompt
tone_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a writing style expert. Rewrite the given text in a {tone} tone. "
            "Maintain the original meaning but adjust the formality, vocabulary, and style. "
            "Output ONLY the rewritten text, no explanations.",
        ),
        ("user", "{text}"),
    ]
)

# 🌍 Step 2: Translation Prompt
translation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a professional translator. Translate the following text from "
            "{input_language} to {output_language}. Preserve the tone and style. "
            "Output ONLY the translation.",
        ),
        ("user", "{adjusted_text}"),
    ]
)

# 🔗 Build Multi-Step Chain using LCEL
# Step 1: Adjust tone
tone_chain = tone_prompt | llm | StrOutputParser()

# Step 2: Translate adjusted text
# We need to pass through input_language and output_language while adding adjusted_text
translation_chain = (
    {
        "adjusted_text": tone_chain,  # Output from step 1
        "input_language": lambda x: x["input_language"],  # Pass through
        "output_language": lambda x: x["output_language"],  # Pass through
    }
    | translation_prompt
    | llm
    | StrOutputParser()
)

# 🧠 Get user input
text = input("Enter text: ")
tone = input("Desired tone (e.g., formal, casual, royal, poetic): ")
input_lang = input("Input language: ")
output_lang = input("Output language: ")

# 📡 Invoke the chain
result = translation_chain.invoke(
    {
        "text": text,
        "tone": tone,
        "input_language": input_lang,
        "output_language": output_lang,
    }
)

# 📤 Print result
print("\n" + "=" * 60)
print(f"Tone-Adjusted Translation ({tone} style, {input_lang} → {output_lang}):")
print("=" * 60)
print(result)
