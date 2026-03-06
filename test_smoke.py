"""Quick smoke test for all RouteMoA modules."""
import sys
sys.path.insert(0, ".")

print("=" * 50)
print("RouteMoA Smoke Test")
print("=" * 50)

# 1. Config
print("\n[1] Testing config.py...")
from config import MODELS, TOP_K, MAX_LAYERS, EARLY_STOP_THRESHOLD
print(f"    Models: {len(MODELS)}")
print(f"    TOP_K={TOP_K}, MAX_LAYERS={MAX_LAYERS}, THRESHOLD={EARLY_STOP_THRESHOLD}")
for m in MODELS:
    print(f"    - {m['name']} ({m['id']})")

# 2. Model pool
print("\n[2] Testing core/model_pool.py...")
from core.model_pool import OpenRouterModel, extract_json_from_text, get_all_model_instances
model = OpenRouterModel("google/gemini-3.1-flash-lite-preview")
print(f"    Created model: {model.name}")

# 3. JSON extraction
print("\n[3] Testing JSON extraction...")
test_cases = [
    '{"answer": "42", "self_score": 0.9}',
    'Here is my answer: ```json\n{"answer": "hello", "self_score": 0.8}\n```',
    'blah blah {"answer": "test", "self_score": 0.7} more text',
]
for tc in test_cases:
    result = extract_json_from_text(tc)
    print(f"    Input: {tc[:50]}... -> {result}")

# 4. Evaluator
print("\n[4] Testing core/evaluator.py...")
from core.evaluator import build_layer1_prompt, parse_layer1_response, extract_json
prompt = build_layer1_prompt("What is 2+2?")
print(f"    Layer1 prompt length: {len(prompt)} chars")
parsed = parse_layer1_response('{"answer": "4", "self_score": 0.95}')
print(f"    Parsed: {parsed}")

# 5. Ranker
print("\n[5] Testing core/ranker.py...")
from core.ranker import rank_models, build_initial_candidates, fuse_scores
candidates = build_initial_candidates({"google/gemini-3.1-flash-lite-preview": 0.8, "deepseek/deepseek-v3.2": 0.9, "mistralai/mistral-large-2512": 0.7})
ranked = rank_models(candidates, top_k=2)
print(f"    Top 2: {[r['model_id'].split('/')[-1] for r in ranked]}")
fused = fuse_scores(scorer_score=0.8, self_score=0.7, cross_score=0.9)
print(f"    Fuse(0.8, 0.7, 0.9) = {fused:.4f}")

# 6. Aggregator
print("\n[6] Testing core/aggregator.py...")
from core.aggregator import format_final_results
result = format_final_results("test?", "answer", [], 0.001, 500, 100, 50)
print(f"    Result keys: {list(result.keys())}")
print(f"    Stats: {result['stats']}")

# 7. Scorer (fallback)
print("\n[7] Testing FallbackScorer...")
from core.scorer import FallbackScorer
scorer = FallbackScorer()
scores = scorer.predict_as_dict("Write a Python function to sort a list")
print(f"    Coding query scores:")
for mid, score in sorted(scores.items(), key=lambda x: -x[1])[:3]:
    print(f"      {mid.split('/')[-1]}: {score}")

# 8. Pipeline (instantiation only)
print("\n[8] Testing pipeline instantiation...")
from core.pipeline import RouteMoAPipeline
pipeline = RouteMoAPipeline(top_k=2, max_layers=2)
print(f"    Pipeline created with {len(pipeline.model_pool)} models")

print("\n" + "=" * 50)
print("ALL TESTS PASSED!")
print("=" * 50)
