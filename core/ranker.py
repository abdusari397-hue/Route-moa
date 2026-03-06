"""
RouteMoA Ranker
Multi-criteria ranking of models based on the paper's priority chain:
1. Predicted/fused score (highest wins)
2. Output token cost (lowest wins)
3. Input token cost (lowest wins)
4. Latency (lowest wins)
"""
from config import MODELS, get_model_by_id, TOP_K, SCORER_WEIGHT, SELF_WEIGHT, CROSS_WEIGHT


def fuse_scores(
    scorer_score: float = 0.0,
    self_score: float = 0.0,
    cross_score: float = 0.0,
) -> float:
    """
    Weighted fusion of SLM scorer, self-assessment, and cross-assessment scores.
    final_score = w1*scorer + w2*self + w3*cross
    """
    return (
        SCORER_WEIGHT * scorer_score
        + SELF_WEIGHT * self_score
        + CROSS_WEIGHT * cross_score
    )


def rank_models(
    candidates: list[dict],
    top_k: int = TOP_K,
) -> list[dict]:
    """
    Rank candidate models using multi-criteria sorting (paper Section 3.3).

    Args:
        candidates: list of dicts with at least:
            - "model_id": str
            - "score": float (the fused score)
            Optional (falls back to config values):
            - "output_cost": float
            - "input_cost": float
            - "latency_ms": float
        top_k: number of models to select

    Returns:
        Top-K models sorted by the multi-criteria chain.
    """
    enriched = []
    for c in candidates:
        model_info = get_model_by_id(c["model_id"])
        enriched.append({
            **c,
            "score": c.get("score", 0.0),
            "output_cost": c.get("output_cost", model_info["output_cost_per_mtok"] if model_info else 0),
            "input_cost": c.get("input_cost", model_info["input_cost_per_mtok"] if model_info else 0),
            "latency_ms": c.get("latency_ms", model_info["avg_latency_ms"] if model_info else 9999),
        })

    # Multi-criteria sort:
    # Primary: score (descending)
    # Secondary: output_cost (ascending)
    # Tertiary: input_cost (ascending)
    # Quaternary: latency_ms (ascending)
    enriched.sort(
        key=lambda m: (-m["score"], m["output_cost"], m["input_cost"], m["latency_ms"])
    )

    return enriched[:top_k]


def build_initial_candidates(scorer_scores: dict[str, float]) -> list[dict]:
    """
    Convert SLM scorer predictions to candidate format for ranking.

    Args:
        scorer_scores: dict mapping model_id → predicted score

    Returns:
        List of candidate dicts ready for rank_models()
    """
    candidates = []
    for model_id, score in scorer_scores.items():
        candidates.append({
            "model_id": model_id,
            "score": score,
        })
    return candidates


def update_scores_with_assessment(
    current_candidates: list[dict],
    self_scores: dict[str, float],
    cross_scores: dict[str, float] = None,
    scorer_scores: dict[str, float] = None,
) -> list[dict]:
    """
    Update candidate scores by fusing SLM scorer, self-assessment,
    and cross-assessment scores.

    Args:
        current_candidates: existing candidate dicts
        self_scores: model_id → self-assessment score
        cross_scores: model_id → cross-assessment score (if available)
        scorer_scores: model_id → original SLM scorer score

    Returns:
        Updated candidates with fused scores
    """
    if cross_scores is None:
        cross_scores = {}
    if scorer_scores is None:
        scorer_scores = {}

    updated = []
    for c in current_candidates:
        mid = c["model_id"]
        fused = fuse_scores(
            scorer_score=scorer_scores.get(mid, c.get("score", 0.5)),
            self_score=self_scores.get(mid, 0.5),
            cross_score=cross_scores.get(mid, 0.5),
        )
        updated.append({
            **c,
            "score": round(fused, 4),
            "scorer_score": scorer_scores.get(mid, 0.0),
            "self_score": self_scores.get(mid, 0.0),
            "cross_score": cross_scores.get(mid, 0.0),
        })

    return updated
