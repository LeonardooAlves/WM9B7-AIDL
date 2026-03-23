import os
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
from munch import Munch

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

# 🚀 Create client (Groq)
client = OpenAI(api_key=api_key, base_url=config.groq.base_url)

# 🧠 Prompt
prompt = input("Type you prompt: ")

# 📡 Send request
response = client.chat.completions.create(
    model=config.groq.model,
    messages=[{"role": "user", "content": prompt}],
    temperature=config.groq.temperature,
    max_tokens=config.groq.max_tokens,
)

# 📤 Print result
print(response.choices[0].message.content)
