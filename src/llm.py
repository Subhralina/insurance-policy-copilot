"""
Step 4: Generation backend.

Swappable LLM call: use local Ollama while developing (free, offline,
no rate limits), swap to Groq's free tier when you deploy publicly
(Ollama needs your machine running as a server, which doesn't work on
Streamlit Cloud). Same interface either way -- nothing else in the
pipeline needs to change when you switch.

Set LLM_BACKEND=ollama or LLM_BACKEND=groq as an environment variable.
For Groq, get a free API key at https://console.groq.com and set
GROQ_API_KEY.
"""

import os


def _call_ollama(prompt: str, model: str = "llama3.1") -> str:
    import ollama  # pip install ollama; requires `ollama serve` running locally

    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"]


def _call_groq(prompt: str, model: str = "llama-3.1-8b-instant") -> str:
    from groq import Groq

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def generate(prompt: str) -> str:
    # Read fresh on every call, not once at import time -- Streamlit Cloud
    # secrets get written into os.environ AFTER this module is first
    # imported, so caching this at import time would silently ignore them.
    backend = os.environ.get("LLM_BACKEND", "ollama")
    if backend == "groq":
        return _call_groq(prompt)
    return _call_ollama(prompt)


if __name__ == "__main__":
    print(generate("Say hello in one short sentence."))
