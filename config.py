"""
RouteMoA Configuration
Model pool definitions, costs, and hyperparameters.
"""
import os
import httpx
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# ─────────────────────────────────────────────
# Model Pool
# ─────────────────────────────────────────────
MODELS = [
    {
        "id": "google/gemini-3.1-flash-lite-preview",
        "name": "Gemini 3.1 Flash Lite",
        "specialty": "general",
        "input_cost_per_mtok": 0.0,
        "output_cost_per_mtok": 0.0,
        "avg_latency_ms": 800,
    },
    {
        "id": "qwen/qwen3.5-35b-a3b",
        "name": "Qwen 3.5 35B",
        "specialty": "general",
        "input_cost_per_mtok": 0.0,
        "output_cost_per_mtok": 0.0,
        "avg_latency_ms": 1200,
    },
    {
        "id": "qwen/qwen3.5-122b-a10b",
        "name": "Qwen 3.5 122B",
        "specialty": "general",
        "input_cost_per_mtok": 0.0,
        "output_cost_per_mtok": 0.0,
        "avg_latency_ms": 2000,
    },
    {
        "id": "minimax/minimax-m2.5",
        "name": "MiniMax M2.5",
        "specialty": "general",
        "input_cost_per_mtok": 0.0,
        "output_cost_per_mtok": 0.0,
        "avg_latency_ms": 1500,
    },
    {
        "id": "z-ai/glm-5",
        "name": "GLM-5",
        "specialty": "general",
        "input_cost_per_mtok": 0.0,
        "output_cost_per_mtok": 0.0,
        "avg_latency_ms": 1800,
    },
    {
        "id": "moonshotai/kimi-k2.5",
        "name": "Kimi K2.5",
        "specialty": "general",
        "input_cost_per_mtok": 0.0,
        "output_cost_per_mtok": 0.0,
        "avg_latency_ms": 2000,
    },
    {
        "id": "deepseek/deepseek-v3.2",
        "name": "DeepSeek V3.2",
        "specialty": "coding_math",
        "input_cost_per_mtok": 0.0,
        "output_cost_per_mtok": 0.0,
        "avg_latency_ms": 2500,
    },
    {
        "id": "mistralai/mistral-large-2512",
        "name": "Mistral Large",
        "specialty": "multilingual",
        "input_cost_per_mtok": 0.0,
        "output_cost_per_mtok": 0.0,
        "avg_latency_ms": 2200,
    },
]

# ─────────────────────────────────────────────
# Hyperparameters
# ─────────────────────────────────────────────
TOP_K = 3                       # Number of models selected per layer
MAX_LAYERS = 3                  # Maximum number of pipeline layers
EARLY_STOP_THRESHOLD = 0.85     # Stop early if confidence >= this
SCORER_WEIGHT = 0.4             # Weight for SLM scorer prediction
SELF_WEIGHT = 0.3               # Weight for self-assessment score
CROSS_WEIGHT = 0.3              # Weight for cross-assessment score
LAMBDA_QUALITY = 0.5            # λ for quality score computation

# ─────────────────────────────────────────────
# Fetch live pricing from OpenRouter
# ─────────────────────────────────────────────
def fetch_model_costs():
    """
    Fetches real-time pricing from OpenRouter API and updates the MODEL pool.
    Call this once at startup if you want live prices.
    """
    if not OPENROUTER_API_KEY:
        print("[Config] No API key set. Using default costs (0.0).")
        return

    try:
        resp = httpx.get(
            f"{OPENROUTER_BASE_URL}/models",
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
            timeout=15.0,
        )
        resp.raise_for_status()
        api_models = {m["id"]: m for m in resp.json().get("data", [])}

        for model in MODELS:
            api_info = api_models.get(model["id"])
            if api_info and "pricing" in api_info:
                pricing = api_info["pricing"]
                # OpenRouter returns cost per token; we store per million tokens
                prompt_cost = float(pricing.get("prompt", "0"))
                completion_cost = float(pricing.get("completion", "0"))
                model["input_cost_per_mtok"] = prompt_cost * 1_000_000
                model["output_cost_per_mtok"] = completion_cost * 1_000_000
                print(f"  [Config] {model['name']}: ${model['input_cost_per_mtok']:.4f} / ${model['output_cost_per_mtok']:.4f} per Mtok")
            else:
                print(f"  [Config] {model['name']}: pricing not found, using defaults")

    except Exception as e:
        print(f"[Config] Failed to fetch pricing: {e}")


def get_model_by_id(model_id: str) -> dict | None:
    """Look up a model definition by its OpenRouter ID."""
    for m in MODELS:
        if m["id"] == model_id:
            return m
    return None


def get_model_index(model_id: str) -> int:
    """Return the index of a model in MODELS list, or -1 if not found."""
    for i, m in enumerate(MODELS):
        if m["id"] == model_id:
            return i
    return -1
