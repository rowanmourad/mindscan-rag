"""
llm_client.py
-------------
PURPOSE:
This file handles all communication with the Large Language Model (LLM)
through the OpenRouter API.

Pipeline:

Retrieved Context
        +
User Prompt
        ↓
Send Request to OpenRouter
        ↓
Large Language Model
        ↓
Generated Medical Response

Instead of running an LLM locally (such as Ollama), this project uses
OpenRouter, which provides access to hosted language models through an API.
This reduces hardware requirements while still allowing the system to
generate intelligent responses.
"""

import requests

# Load the default LLM model name from the configuration file.
from config import OPENROUTER_MODEL


# ==========================================================
# OPENROUTER API ENDPOINT
# ==========================================================
# Purpose:
# This is the web address where every request is sent.
#
# Why?
#
# Instead of executing the language model on our own computer,
# we send an HTTP request to OpenRouter's server, which runs
# the model and returns the generated response.
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# ==========================================================
# STEP 1 — TEST API CONNECTION
# ==========================================================
# Purpose:
# Verify that the API key is valid and the selected language
# model is available before running the full Graph RAG system.
#
# Why?
#
# Detecting configuration problems early prevents wasting time
# processing documents only to discover that the LLM cannot
# be reached.
#
# A very small prompt ("Say 'ok'") is used because it is fast,
# inexpensive, and confirms that communication is working.
def test_openrouter_connection(api_key: str, model: str = None) -> bool:
    """Sends a trivial request to confirm the API key + model work."""

    # Use the default model unless another is specified.
    model = model or OPENROUTER_MODEL
    try:

        # Send a simple HTTP POST request.
        response = requests.post(
            OPENROUTER_URL,
            headers={"Authorization": f"Bearer {api_key}"},             # Authenticate using the API key.
            json={
                "model": model,                                         # Language model to use.
                "messages": [{"role": "user", "content": "Say 'ok' and nothing else."}],                       # Very small test prompt.
            },
            timeout=30,
        )
        response.raise_for_status()                   # Raise an exception if an error occurred.
        print(response.status_code, response.json()["choices"][0]["message"]["content"])         # Print the returned response.
        return True
    except Exception as e:
        print("OpenRouter not reachable:", e)
        return False

# ==========================================================
# STEP 2 — SEND PROMPT TO THE LLM
# ==========================================================
# Purpose:
# Send the final prompt to the language model and receive the
# generated response.
#
# Why?
#
# After RAG retrieves the most relevant medical evidence,
# those retrieved chunks are combined with the user's question
# to create one prompt.
#
# This function sends that prompt to the LLM so it can generate
# a grounded, evidence-based answer.

def call_openrouter(prompt: str, api_key: str, model: str = None,
                     temperature: float = 0.2, timeout: int = 180,
                     max_tokens: int = 1500) -> str:
    """Sends a single-turn prompt to OpenRouter and returns the text reply."""
                         
    model = model or OPENROUTER_MODEL                    # Use the default configured model unless another,model is provided.

      # ------------------------------------------------------
    # Send the request to the language model.
    # ------------------------------------------------------                    
    response = requests.post(
        OPENROUTER_URL,
        headers={"Authorization": f"Bearer {api_key}"},             # Authenticate the request.
        json={
            "model": model,                                         # Selected LLM.
            "messages": [{"role": "user", "content": prompt}],        # User prompt containing the retrieved context.
            "temperature": temperature,                                 # Controls randomness,deterministic responses,Lower values produce more consistent
            "max_tokens": max_tokens,                                     # Maximum response length.
        },
        timeout=timeout,
    )

  # ======================================================
    # STEP 3 — HANDLE ERRORS
    # ======================================================
    # Purpose:
    # Display detailed error messages if the request fails.
    #
    # Why?
    #
    # Problems such as:
    # • Invalid API key
    # • Rate limits
    # • Model unavailable
    # • Policy restrictions
    #
    # are much easier to diagnose using OpenRouter's
    # detailed error message instead of only an HTTP
    # status code.                       
    if not response.ok:
        # OpenRouter's error body (rate limits, data-policy issues, model
        # unavailable, etc.) is much more useful than a bare status code.
        print("OpenRouter error", response.status_code, "-", response.text[:1000])
    response.raise_for_status()                                                           # Raise an exception if the request failed.

     # ======================================================
    # STEP 4 — RETURN THE LLM RESPONSE
    # ======================================================
    # Purpose:
    # Extract only the generated text from the JSON response.
    #
    # Why?
    #
    # The API returns a large JSON object containing metadata,
    # token usage, model information, and the generated answer.
    #
    # For the Graph RAG pipeline, we only need the final
    # response produced by the language model.                     
    return response.json()["choices"][0]["message"]["content"]
