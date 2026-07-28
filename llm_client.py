"""
llm_client.py
-------------
Thin wrapper around the OpenRouter chat-completions API. Replaces the
notebook's Ollama-based calls with a free, hosted OpenRouter model.
"""

import requests

from config import OPENROUTER_MODEL

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def test_openrouter_connection(api_key: str, model: str = None) -> bool:
    """Sends a trivial request to confirm the API key + model work."""
    model = model or OPENROUTER_MODEL
    try:
        response = requests.post(
            OPENROUTER_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": "Say 'ok' and nothing else."}],
            },
            timeout=30,
        )
        response.raise_for_status()
        print(response.status_code, response.json()["choices"][0]["message"]["content"])
        return True
    except Exception as e:
        print("OpenRouter not reachable:", e)
        return False


def call_openrouter(prompt: str, api_key: str, model: str = None,
                     temperature: float = 0.2, timeout: int = 180,
                     max_tokens: int = 1500) -> str:
    """Sends a single-turn prompt to OpenRouter and returns the text reply."""
    model = model or OPENROUTER_MODEL
    response = requests.post(
        OPENROUTER_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=timeout,
    )
    if not response.ok:
        # OpenRouter's error body (rate limits, data-policy issues, model
        # unavailable, etc.) is much more useful than a bare status code.
        print("OpenRouter error", response.status_code, "-", response.text[:1000])
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]
