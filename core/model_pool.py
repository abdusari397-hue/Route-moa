"""
OpenRouter API Wrapper
Async-compatible client for calling LLMs via OpenRouter.
"""
import asyncio
import json
import re
import time
from typing import Optional

import httpx

from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, MODELS, get_model_by_id

# Global shared client for connection pooling
_shared_client: Optional[httpx.AsyncClient] = None

def get_shared_client() -> httpx.AsyncClient:
    global _shared_client
    if _shared_client is None or _shared_client.is_closed:
        # Tighter 30s timeout for "Fast Fail" instead of default 120s
        _shared_client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=10.0, read=45.0) 
        )
    return _shared_client

class OpenRouterModel:
    """
    Async wrapper for a single OpenRouter model endpoint.
    Handles generation, JSON extraction, cost tracking, and retries.
    """

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.model_info = get_model_by_id(model_id)
        self.name = self.model_info["name"] if self.model_info else model_id

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        retries: int = 3,
        timeout: Optional[float] = None,
    ) -> dict:
        """
        Call the model via OpenRouter and return structured results.

        Returns:
            {
                "model_id": str,
                "answer": str,          # Raw text response
                "tokens_in": int,
                "tokens_out": int,
                "latency_ms": float,
                "cost": float,
                "error": str | None,
            }
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Dynamically fetch API key to support runtime UI input
        current_api_key = os.getenv("OPENROUTER_API_KEY", OPENROUTER_API_KEY)

        headers = {
            "Authorization": f"Bearer {current_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://routemoa.local",
            "X-Title": "RouteMoA",
        }

        payload = {
            "model": self.model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        last_error = None
        client = get_shared_client()

        for attempt in range(retries):
            try:
                start = time.perf_counter()
                
                req_timeout = timeout if timeout is not None else httpx.USE_CLIENT_DEFAULT
                
                resp = await client.post(
                    f"{OPENROUTER_BASE_URL}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=req_timeout
                )
                latency_ms = (time.perf_counter() - start) * 1000

                if resp.status_code != 200:
                    error_body = resp.text
                    last_error = f"HTTP {resp.status_code}: {error_body[:300]}"
                    if resp.status_code in (429, 502, 503):
                        await asyncio.sleep(2 ** attempt)
                        continue
                    break

                data = resp.json()

                # Extract answer
                answer = ""
                if "choices" in data and len(data["choices"]) > 0:
                    answer = data["choices"][0].get("message", {}).get("content", "")

                # Extract usage
                usage = data.get("usage", {})
                tokens_in = usage.get("prompt_tokens", 0)
                tokens_out = usage.get("completion_tokens", 0)

                # Calculate cost
                cost = 0.0
                if self.model_info:
                    cost = (
                        tokens_in * self.model_info["input_cost_per_mtok"] / 1_000_000
                        + tokens_out * self.model_info["output_cost_per_mtok"] / 1_000_000
                    )

                return {
                    "model_id": self.model_id,
                    "answer": answer,
                    "tokens_in": tokens_in,
                    "tokens_out": tokens_out,
                    "latency_ms": round(latency_ms, 1),
                    "cost": round(cost, 6),
                    "error": None,
                }

            except (httpx.TimeoutException, httpx.ConnectError) as e:
                last_error = f"Connection error: {e}"
                await asyncio.sleep(2 ** attempt)
            except Exception as e:
                last_error = f"Unexpected error: {e}"
                break

        # All retries exhausted
        return {
            "model_id": self.model_id,
            "answer": "",
            "tokens_in": 0,
            "tokens_out": 0,
            "latency_ms": 0,
            "cost": 0.0,
            "error": last_error,
        }


# ─────────────────────────────────────────────
# JSON Extraction Utilities
# ─────────────────────────────────────────────

def extract_json_from_text(text: str) -> Optional[dict]:
    """
    Robustly extract a JSON object from LLM output.
    Tries multiple strategies: direct parse, code-fence extraction, regex.
    """
    if not text:
        return None

    # Strategy 1: Direct JSON parse
    text_stripped = text.strip()
    if text_stripped.startswith("{"):
        try:
            return json.loads(text_stripped)
        except json.JSONDecodeError:
            pass

    # Strategy 2: Extract from ```json ... ``` code fences
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Strategy 3: Find the first { ... } block
    brace_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def get_model_instance(model_id: str) -> OpenRouterModel:
    """Factory function to create an OpenRouterModel by ID."""
    return OpenRouterModel(model_id)


def get_all_model_instances() -> list[OpenRouterModel]:
    """Create instances for all models in the pool."""
    return [OpenRouterModel(m["id"]) for m in MODELS]
