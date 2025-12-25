import os
from typing import Any, Dict, List

import requests


def llm_chat(messages: List[Dict[str, str]]) -> str:
    base_url = os.getenv("LLM_BASE_URL", "http://localhost:1234/v1").rstrip("/")
    model = os.getenv("LLM_MODEL", "local-model")
    api_key = os.getenv("LLM_API_KEY", "lm-studio")

    url = f"{base_url}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
    }

    r = requests.post(url, headers=headers, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]
