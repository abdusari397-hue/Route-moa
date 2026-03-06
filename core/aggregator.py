"""
RouteMoA Aggregator
Handles mid-layer context concatenation and final answer synthesis.
"""


def build_context_from_layer(layer_results: list[dict]) -> str:
    """
    Concatenate answers from a layer into a context block for the next layer.

    Args:
        layer_results: list of dicts with "model_name", "answer", "self_score"

    Returns:
        Formatted context string
    """
    parts = []
    for i, result in enumerate(layer_results):
        model_name = result.get("model_name", f"Model {i+1}")
        answer = result.get("answer", "")
        self_score = result.get("self_score", "N/A")
        parts.append(
            f"--- {model_name} (confidence: {self_score}) ---\n{answer}"
        )
    return "\n\n".join(parts)


def format_final_results(
    query: str,
    final_answer: str,
    layer_trace: list[dict],
    total_cost: float,
    total_latency_ms: float,
    total_tokens_in: int,
    total_tokens_out: int,
) -> dict:
    """
    Format the final pipeline output into a structured result.
    """
    return {
        "query": query,
        "final_answer": final_answer,
        "layer_trace": layer_trace,
        "stats": {
            "total_cost": round(total_cost, 6),
            "total_latency_ms": round(total_latency_ms, 1),
            "total_tokens_in": total_tokens_in,
            "total_tokens_out": total_tokens_out,
            "num_layers": len(layer_trace),
        },
    }
