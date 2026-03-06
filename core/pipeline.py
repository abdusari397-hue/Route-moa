"""
RouteMoA Pipeline Orchestrator
The full multi-layer inference pipeline implementing the paper's architecture:

Layer 1: SLM Scorer → Rank → Select Top-K → Generate (parallel) → Self-assess
Layer 2+: Cross-assess → Re-rank → Select Top-K → Generate → Self+Cross assess  
Final:    Best model synthesizes all answers

With Early Stopping: Skip to Final if any model score >= threshold
"""
import asyncio
import time
from typing import Optional
from colorama import Fore, Style

from config import (
    MODELS, TOP_K, MAX_LAYERS, EARLY_STOP_THRESHOLD,
    get_model_by_id,
)
from core.model_pool import OpenRouterModel, get_model_instance
from core.scorer import FallbackScorer
from core.evaluator import (
    LAYER1_SYSTEM_PROMPT,
    build_layer1_prompt,
    build_intermediate_prompt,
    build_cross_assessment_prompt,
    build_final_aggregation_prompt,
    parse_layer1_response,
    parse_intermediate_response,
    parse_cross_assessment,
)
from core.ranker import (
    rank_models,
    build_initial_candidates,
    update_scores_with_assessment,
    fuse_scores,
)
from core.aggregator import build_context_from_layer, format_final_results


class RouteMoAPipeline:
    """
    Full RouteMoA inference pipeline.
    
    Implements:
    - SLM-based pre-inference routing
    - Mixture of Judges (self + cross assessment)
    - Multi-criteria dynamic ranking
    - Early stopping
    - Async parallel model execution
    """

    def __init__(
        self,
        top_k: int = TOP_K,
        max_layers: int = MAX_LAYERS,
        early_stop_threshold: float = EARLY_STOP_THRESHOLD,
        use_slm_scorer: bool = False,
        active_model_ids: Optional[list[str]] = None,
        log_callback=None,
    ):
        self.top_k = top_k
        self.max_layers = max_layers
        self.early_stop_threshold = early_stop_threshold
        self.use_slm_scorer = use_slm_scorer
        self.log_callback = log_callback

        # Scorer (SLM or Fallback)
        if use_slm_scorer:
            try:
                from core.scorer import RouteMoAScorer
                self.scorer = RouteMoAScorer()
                if not self.scorer.load():
                    self._log("⚠️ SLM Scorer not trained yet, falling back to heuristic scorer")
                    self.scorer = FallbackScorer()
            except Exception as e:
                self._log(f"⚠️ SLM Scorer load failed: {e}, using fallback")
                self.scorer = FallbackScorer()
        else:
            self.scorer = FallbackScorer()

        # Filter active models
        if active_model_ids:
            self.model_pool = [m for m in MODELS if m["id"] in active_model_ids]
        else:
            self.model_pool = list(MODELS)

        if not self.model_pool:
            raise ValueError("No models available in the pool!")

    def _log(self, message: str):
        """Log a message via callback or print."""
        if self.log_callback:
            self.log_callback(message)
        else:
            print(message)

    async def run(self, query: str) -> dict:
        """
        Execute the full RouteMoA pipeline.

        Args:
            query: User's question/prompt

        Returns:
            Structured result with final_answer, layer_trace, and stats
        """
        pipeline_start = time.perf_counter()
        total_cost = 0.0
        total_tokens_in = 0
        total_tokens_out = 0
        layer_trace = []
        all_answers = []  # Accumulate answers across layers

        self._log(f"\n{'='*60}")
        self._log(f"🚀 RouteMoA Pipeline Started")
        self._log(f"📝 Query: {query[:100]}...")
        self._log(f"{'='*60}\n")

        # ──────────────────────────────────────
        # Step 1: SLM Scorer Prediction
        # ──────────────────────────────────────
        self._log(f"{Fore.CYAN}📊 Step 1: SLM Scorer Prediction{Style.RESET_ALL}")
        scorer_scores = self.scorer.predict_as_dict(query)

        # Only keep scores for models in the active pool
        pool_ids = {m["id"] for m in self.model_pool}
        scorer_scores = {k: v for k, v in scorer_scores.items() if k in pool_ids}

        for mid, score in sorted(scorer_scores.items(), key=lambda x: -x[1]):
            model_name = get_model_by_id(mid)["name"] if get_model_by_id(mid) else mid
            bar = "█" * int(score * 20)
            self._log(f"  {model_name:.<30} {score:.4f} {bar}")

        # ──────────────────────────────────────
        # Step 2: Layer 1 — Top-K Selection + Generation + Self-assessment
        # ──────────────────────────────────────
        self._log(f"\n{Fore.GREEN}🔄 Layer 1: Initial Generation + Self-Assessment{Style.RESET_ALL}")

        candidates = build_initial_candidates(scorer_scores)
        selected = rank_models(candidates, top_k=self.top_k)
        selected_ids = [c["model_id"] for c in selected]

        self._log(f"  Selected models: {[get_model_by_id(mid)['name'] for mid in selected_ids]}")

        # Generate in parallel
        layer1_results = await self._run_layer1(query, selected_ids)

        # Track stats
        layer1_trace = {
            "layer": 1,
            "type": "generation + self-assessment",
            "models": [],
        }
        for result in layer1_results:
            total_cost += result["cost"]
            total_tokens_in += result["tokens_in"]
            total_tokens_out += result["tokens_out"]
            layer1_trace["models"].append({
                "model_id": result["model_id"],
                "model_name": result["model_name"],
                "self_score": result["self_score"],
                "latency_ms": result["latency_ms"],
                "cost": result["cost"],
            })
            all_answers.append(result)

        layer_trace.append(layer1_trace)

        # Check early stopping
        max_score = max(r["self_score"] for r in layer1_results)
        if max_score >= self.early_stop_threshold:
            self._log(f"\n{Fore.YELLOW}⚡ Early stopping triggered! Max self_score = {max_score:.4f} >= {self.early_stop_threshold}{Style.RESET_ALL}")
        else:
            # ──────────────────────────────────────
            # Step 3: Layer 2+ — Cross-Assessment + Refinement
            # ──────────────────────────────────────
            prev_results = layer1_results
            for layer_num in range(2, self.max_layers + 1):
                self._log(f"\n{Fore.MAGENTA}🔄 Layer {layer_num}: Cross-Assessment + Refinement{Style.RESET_ALL}")

                # Cross-assessment: top model judges others
                cross_scores = await self._run_cross_assessment(query, prev_results)

                # Update scores with assessment
                self_scores_dict = {r["model_id"]: r["self_score"] for r in prev_results}
                updated_candidates = update_scores_with_assessment(
                    candidates,
                    self_scores=self_scores_dict,
                    cross_scores=cross_scores,
                    scorer_scores=scorer_scores,
                )

                # Re-rank and select
                selected = rank_models(updated_candidates, top_k=self.top_k)
                selected_ids = [c["model_id"] for c in selected]

                self._log(f"  Re-selected models: {[get_model_by_id(mid)['name'] for mid in selected_ids]}")

                # Generate with context from previous layer
                layer_results = await self._run_intermediate_layer(
                    query, 
                    selected_ids, 
                    prev_results, 
                    layer_num, 
                    scorer_scores, 
                    cross_scores
                )

                # Track stats
                layer_n_trace = {
                    "layer": layer_num,
                    "type": "cross-assessment + refinement",
                    "models": [],
                }
                for result in layer_results:
                    total_cost += result["cost"]
                    total_tokens_in += result["tokens_in"]
                    total_tokens_out += result["tokens_out"]
                    layer_n_trace["models"].append({
                        "model_id": result["model_id"],
                        "model_name": result["model_name"],
                        "self_score": result["self_score"],
                        "cross_score": cross_scores.get(result["model_id"], 0),
                        "fused_score": result.get("fused_score", 0),
                        "latency_ms": result["latency_ms"],
                        "cost": result["cost"],
                    })
                    all_answers.append(result)

                layer_trace.append(layer_n_trace)
                prev_results = layer_results

                # Check early stopping
                fused_max = max(r.get("fused_score", r["self_score"]) for r in layer_results)
                if fused_max >= self.early_stop_threshold:
                    self._log(f"\n{Fore.YELLOW}⚡ Early stopping triggered! Max fused score = {fused_max:.4f}{Style.RESET_ALL}")
                    break

        # ──────────────────────────────────────
        # Step 4: Final Aggregation
        # ──────────────────────────────────────
        self._log(f"\n{Fore.BLUE}🎯 Final Layer: Aggregation{Style.RESET_ALL}")

        # Pick the best model for aggregation
        best_result = max(all_answers, key=lambda r: r.get("fused_score", r["self_score"]))
        best_model_id = best_result["model_id"]
        best_model_name = best_result["model_name"]
        self._log(f"  Aggregator: {best_model_name}")

        final_answer, final_cost, final_tokens = await self._run_final_aggregation(
            query, all_answers, best_model_id
        )
        total_cost += final_cost
        total_tokens_in += final_tokens[0]
        total_tokens_out += final_tokens[1]

        layer_trace.append({
            "layer": "final",
            "type": "aggregation",
            "aggregator_model": best_model_name,
            "cost": final_cost,
        })

        total_latency = (time.perf_counter() - pipeline_start) * 1000

        self._log(f"\n{'='*60}")
        self._log(f"✅ Pipeline Complete!")
        self._log(f"   💰 Total Cost: ${total_cost:.6f}")
        self._log(f"   ⏱️  Total Time: {total_latency/1000:.2f}s")
        self._log(f"   📊 Layers Used: {len(layer_trace)}")
        self._log(f"{'='*60}\n")

        return format_final_results(
            query=query,
            final_answer=final_answer,
            layer_trace=layer_trace,
            total_cost=total_cost,
            total_latency_ms=total_latency,
            total_tokens_in=total_tokens_in,
            total_tokens_out=total_tokens_out,
        )

    # ─────────────────────────────────────
    # Internal Layer Methods
    # ─────────────────────────────────────

    async def _run_layer1(self, query: str, model_ids: list[str]) -> list[dict]:
        """Run Layer 1: parallel generation with self-assessment."""
        prompt = build_layer1_prompt(query)
        tasks = []
        for mid in model_ids:
            model = get_model_instance(mid)
            t = asyncio.create_task(self._generate_and_parse_layer1(model, prompt))
            tasks.append(t)

        results = []
        for coro in asyncio.as_completed(tasks):
            try:
                res = await coro
                results.append(res)
                # True Async Early Stopping! Cancel other tasks immediately
                if res["self_score"] >= self.early_stop_threshold:
                    self._log(f"  ⚡ Fast Early Stop triggered synchronously by {res['model_name']} ({res['self_score']:.2f})!")
                    for t in tasks:
                        if not t.done():
                            t.cancel()
                    break
            except asyncio.CancelledError:
                pass

        return results

    async def _generate_and_parse_layer1(
        self, model: OpenRouterModel, prompt: str
    ) -> dict:
        """Generate and parse a single Layer 1 response."""
        # Fast fail: 40s timeout for initial generation
        raw = await model.generate(prompt, system_prompt=LAYER1_SYSTEM_PROMPT, timeout=40.0)

        if raw["error"]:
            self._log(f"  ❌ {model.name}: {raw['error']}")
            return {
                "model_id": model.model_id,
                "model_name": model.name,
                "answer": "",
                "self_score": 0.0,
                "tokens_in": 0,
                "tokens_out": 0,
                "latency_ms": 0,
                "cost": 0.0,
            }

        parsed = parse_layer1_response(raw["answer"])
        self._log(
            f"  ✅ {model.name}: self_score={parsed['self_score']:.2f} "
            f"({raw['latency_ms']:.0f}ms, ${raw['cost']:.6f})"
        )

        return {
            "model_id": model.model_id,
            "model_name": model.name,
            "answer": parsed["answer"],
            "self_score": parsed["self_score"],
            "tokens_in": raw["tokens_in"],
            "tokens_out": raw["tokens_out"],
            "latency_ms": raw["latency_ms"],
            "cost": raw["cost"],
        }

    async def _run_cross_assessment(
        self, query: str, prev_results: list[dict]
    ) -> dict[str, float]:
        """
        Run cross-assessment: the top model judges all others.
        Returns dict mapping model_id → cross-assessment score.
        """
        # Pick the top-scoring model as judge
        judge = max(prev_results, key=lambda r: r["self_score"])
        judge_model = get_model_instance(judge["model_id"])
        self._log(f"  👨‍⚖️ Judge: {judge['model_name']}")

        # Build list of answers to judge (excluding the judge itself)
        answers_to_judge = [r for r in prev_results if r["model_id"] != judge["model_id"]]

        if not answers_to_judge:
            return {r["model_id"]: r["self_score"] for r in prev_results}

        prompt = build_cross_assessment_prompt(query, answers_to_judge)
        # Judging is fast, max_tokens=512, strict timeout
        raw = await judge_model.generate(prompt, max_tokens=512, timeout=20.0)

        cross_scores_dict = {}
        # Judge gives itself a high score
        cross_scores_dict[judge["model_id"]] = min(judge["self_score"] + 0.05, 1.0)

        if not raw["error"]:
            scores = parse_cross_assessment(raw["answer"], len(answers_to_judge))
            for i, ans in enumerate(answers_to_judge):
                cross_scores_dict[ans["model_id"]] = scores[i] if i < len(scores) else 0.5
                self._log(f"    {ans['model_name']}: cross_score={scores[i]:.2f}" if i < len(scores) else "")
        else:
            self._log(f"  ⚠️ Cross-assessment failed: {raw['error']}")
            for ans in answers_to_judge:
                cross_scores_dict[ans["model_id"]] = 0.5

        return cross_scores_dict

    async def _run_intermediate_layer(
        self,
        query: str,
        model_ids: list[str],
        prev_results: list[dict],
        layer_num: int,
        scorer_scores: dict[str, float],
        cross_scores: dict[str, float],
    ) -> list[dict]:
        """Run intermediate layer: parallel generation using previous context."""
        tasks = []
        for mid in model_ids:
            model = get_model_instance(mid)
            t = asyncio.create_task(
                self._generate_and_parse_layer_n(
                    model=model,
                    query=query,
                    prev_results=prev_results,
                    scorer_score=scorer_scores.get(mid, 0.0),
                    cross_score=cross_scores.get(mid, 0.0),
                )
            )
            tasks.append(t)

        results = []
        for coro in asyncio.as_completed(tasks):
            try:
                res = await coro
                results.append(res)
                # Assuming fused_score or self_score. We early stop if we hit the threshold
                if res.get("fused_score", res["self_score"]) >= self.early_stop_threshold:
                    self._log(f"  ⚡ Fast Early Stop triggered by {res['model_name']} in Layer {layer_num}!")
                    for t in tasks:
                        if not t.done():
                            t.cancel()
                    break
            except asyncio.CancelledError:
                pass

        return results

    async def _generate_and_parse_layer_n(
        self,
        model: OpenRouterModel,
        query: str,
        prev_results: list[dict],
        scorer_score: float,
        cross_score: float,
    ) -> dict:
        """Generate and parse a Layer 2+ response."""
        prompt = build_intermediate_prompt(query, prev_results, len(prev_results)) # layer_num is not needed here
        # Strict timeout for intermediate layers
        raw = await model.generate(prompt, system_prompt=LAYER_N_SYSTEM_PROMPT, timeout=40.0)

        if raw["error"]:
            self._log(f"  ❌ {model.name}: {raw['error']}")
            return {
                "model_id": model.model_id,
                "model_name": model.name,
                "answer": "",
                "self_score": 0.0,
                "fused_score": 0.0,
                "tokens_in": 0,
                "tokens_out": 0,
                "latency_ms": 0,
                "cost": 0.0,
            }

        parsed = parse_intermediate_response(raw["answer"], len(prev_results))
        fused = fuse_scores(scorer_score=scorer_score, self_score=parsed["self_score"], cross_score=cross_score)

        self._log(
            f"  ✅ {model.name}: self={parsed['self_score']:.2f} fused={fused:.2f} "
            f"({raw['latency_ms']:.0f}ms, ${raw['cost']:.6f})"
        )

        return {
            "model_id": model.model_id,
            "model_name": model.name,
            "answer": parsed["answer"],
            "self_score": parsed["self_score"],
            "peer_scores": parsed.get("peer_scores", []),
            "fused_score": fused,
            "tokens_in": raw["tokens_in"],
            "tokens_out": raw["tokens_out"],
            "latency_ms": raw["latency_ms"],
            "cost": raw["cost"],
        }

    async def _run_final_aggregation(
        self, query: str, all_answers: list[dict], aggregator_model_id: str
    ) -> tuple[str, float, tuple[int, int]]:
        """
        Run the final aggregation step.
        Returns (final_answer, cost, (tokens_in, tokens_out))
        """
        model = get_model_instance(aggregator_model_id)
        prompt = build_final_aggregation_prompt(query, all_answers)

        # Give aggregator more time and max tokens since it produces the final answer
        raw = await model.generate(prompt, max_tokens=4096, timeout=60.0)

        if raw["error"]:
            self._log(f"  ❌ Aggregation failed: {raw['error']}")
            # Fallback: use the best answer directly
            best = max(all_answers, key=lambda r: r.get("fused_score", r["self_score"]))
            return best["answer"], 0.0, (0, 0)

        self._log(f"  ✅ Aggregation complete ({raw['latency_ms']:.0f}ms, ${raw['cost']:.6f})")
        return raw["answer"], raw["cost"], (raw["tokens_in"], raw["tokens_out"])
