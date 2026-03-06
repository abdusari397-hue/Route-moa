"""
RouteMoA Training — Generate Quality Scores

Runs all 9 models against the collected questions to produce training labels
for the SLM Scorer.

Quality formula from the paper:
    s_j(k) = λ * accuracy + (1-λ) * reward_score

Evaluation strategies per answer type:
  - numeric:         Extract final number, compare with ground truth
  - multiple_choice: Extract chosen letter (A/B/C/D), compare
  - code:            Heuristic + LLM-as-judge
  - open_ended:      LLM-as-judge

Features:
  - Checkpoint/resume: saves progress after each question batch
  - Parallel calls:    all 9 models called concurrently per question
  - Cost tracking:     total API cost reported
"""
import asyncio
import json
import os
import re
import sys
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import MODELS, LAMBDA_QUALITY
from core.model_pool import get_model_instance


# ─────────────────────────────────────────────
# Answer Evaluation Functions
# ─────────────────────────────────────────────

def evaluate_numeric(answer: str, ground_truth: str) -> float:
    """
    Extract the final numeric answer and compare.
    Handles: "The answer is 42", "#### 42", "42.0", etc.
    """
    if not answer:
        return 0.0
    def extract_number(text: str) -> str | None:
        # Try to find #### delimiter first (GSM8K format)
        if "####" in text:
            return text.split("####")[-1].strip().replace(",", "")
        # Look for common patterns
        patterns = [
            r"(?:answer|result|solution|=)\s*(?:is|:)?\s*\$?\\?boxed\{([^}]+)\}",
            r"(?:answer|result|solution|=)\s*(?:is|:)?\s*([+-]?\d[\d,]*\.?\d*)",
            r"\*\*([+-]?\d[\d,]*\.?\d*)\*\*",
            r"([+-]?\d[\d,]*\.?\d*)\s*$",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).replace(",", "").strip()
        # Last resort: find any number
        numbers = re.findall(r"[+-]?\d+\.?\d*", text)
        return numbers[-1] if numbers else None

    predicted = extract_number(answer)
    expected = ground_truth.strip().replace(",", "")

    if predicted is None:
        return 0.0

    try:
        # Compare as floats for numeric equality
        return 1.0 if abs(float(predicted) - float(expected)) < 1e-6 else 0.0
    except ValueError:
        return 1.0 if predicted.strip() == expected.strip() else 0.0


def evaluate_multiple_choice(answer: str, ground_truth: str) -> float:
    """
    Extract chosen letter (A/B/C/D) and compare with correct answer.
    """
    if not answer:
        return 0.0
    gt = ground_truth.strip().upper()

    # Strategy 1: "The answer is X"
    match = re.search(r"(?:answer|correct|choice)\s*(?:is|:)\s*\(?([A-D])\)?", answer, re.IGNORECASE)
    if match:
        return 1.0 if match.group(1).upper() == gt else 0.0

    # Strategy 2: Bolded answer "**A**"
    match = re.search(r"\*\*([A-D])\*\*", answer)
    if match:
        return 1.0 if match.group(1).upper() == gt else 0.0

    # Strategy 3: First standalone letter
    match = re.search(r"\b([A-D])\)", answer)
    if match:
        return 1.0 if match.group(1).upper() == gt else 0.0

    # Strategy 4: Just the letter
    clean = answer.strip()
    if len(clean) == 1 and clean.upper() in "ABCD":
        return 1.0 if clean.upper() == gt else 0.0

    # Strategy 5: Check if the answer text contains the correct choice text
    if gt.lower() in answer.lower():
        return 0.5  # Partial credit

    return 0.0


def evaluate_code(answer: str, ground_truth: str) -> float:
    """
    Heuristic code evaluation:
    - Check if code block is present
    - Check key function/structure overlap
    """
    if not answer:
        return 0.0
    # Extract code from markdown fences
    code_match = re.search(r"```(?:python)?\s*\n(.*?)```", answer, re.DOTALL)
    code = code_match.group(1).strip() if code_match else answer

    if not code or len(code) < 10:
        return 0.0

    # Check structural overlap with ground truth
    gt_keywords = set(re.findall(r'\b(def|return|for|while|if|class|import|print)\b', ground_truth))
    answer_keywords = set(re.findall(r'\b(def|return|for|while|if|class|import|print)\b', code))

    if not gt_keywords:
        return 0.5 if len(code) > 20 else 0.2

    keyword_overlap = len(gt_keywords & answer_keywords) / len(gt_keywords)

    # Check if function names match
    gt_funcs = set(re.findall(r'def\s+(\w+)', ground_truth))
    answer_funcs = set(re.findall(r'def\s+(\w+)', code))
    func_match = 1.0 if gt_funcs and gt_funcs & answer_funcs else 0.5

    return min(keyword_overlap * 0.6 + func_match * 0.4, 1.0)


def compute_accuracy(answer: str, ground_truth: str, answer_type: str) -> float:
    """Route to the correct evaluation function based on answer type."""
    if answer_type == "numeric":
        return evaluate_numeric(answer, ground_truth)
    elif answer_type == "multiple_choice":
        return evaluate_multiple_choice(answer, ground_truth)
    elif answer_type == "code":
        return evaluate_code(answer, ground_truth)
    else:
        # Open-ended: simple containment
        return 1.0 if ground_truth.strip().lower() in answer.lower() else 0.0


async def llm_judge_score(
    question: str,
    answer: str,
    ground_truth: str,
    judge_model_id: str = "google/gemini-3.1-flash-lite-preview",
) -> float:
    """
    Use an LLM as a judge to evaluate answer quality.
    This acts as the 'reward model' component in the paper's formula.
    """
    judge = get_model_instance(judge_model_id)
    prompt = f"""Rate the quality of this answer on a scale of 0.0 to 1.0.

Question: {question[:500]}

Reference Answer: {ground_truth[:500]}

Model's Answer: {answer[:500]}

Rate ONLY the factual accuracy, completeness, and correctness.
Respond with a single number between 0.0 and 1.0, nothing else."""

    result = await judge.generate(prompt, max_tokens=10, temperature=0.0)

    if result["error"]:
        return 0.5  # Default if judge fails

    # Extract score
    text = result["answer"].strip()
    try:
        score = float(re.search(r"([\d.]+)", text).group(1))
        return max(0.0, min(1.0, score))
    except (ValueError, AttributeError):
        return 0.5


# ─────────────────────────────────────────────
# Main Score Generation
# ─────────────────────────────────────────────

async def evaluate_model_on_question(
    model_id: str,
    question: str,
    ground_truth: str,
    answer_type: str = "numeric",
    use_llm_judge: bool = False,
) -> dict:
    """
    Run a single model on a single question and compute quality score.
    
    s_j(k) = λ * accuracy + (1-λ) * reward_score
    """
    model = get_model_instance(model_id)
    result = await model.generate(
        prompt=question,
        system_prompt="Answer the question accurately and concisely. Show your reasoning.",
        max_tokens=1024,
    )

    if result["error"]:
        return {
            "model_id": model_id,
            "answer": "",
            "quality_score": 0.0,
            "accuracy": 0.0,
            "reward_score": 0.0,
            "cost": 0.0,
            "latency_ms": 0,
            "error": result["error"],
        }

    answer = result["answer"] or ""

    # Compute accuracy (exact match / structural match)
    accuracy = compute_accuracy(answer, ground_truth, answer_type)

    # Compute reward score
    if use_llm_judge:
        reward_score = await llm_judge_score(question, answer, ground_truth)
    else:
        # Heuristic reward: word overlap + length appropriateness
        reward_score = compute_heuristic_reward(answer, ground_truth)

    # Combined quality score (paper formula)
    quality = LAMBDA_QUALITY * accuracy + (1 - LAMBDA_QUALITY) * reward_score

    return {
        "model_id": model_id,
        "answer": answer[:500],
        "quality_score": round(quality, 4),
        "accuracy": round(accuracy, 4),
        "reward_score": round(reward_score, 4),
        "cost": result["cost"],
        "latency_ms": result["latency_ms"],
        "error": None,
    }


def compute_heuristic_reward(answer: str, ground_truth: str) -> float:
    """Heuristic reward score based on word overlap and length."""
    if not answer:
        return 0.0

    answer_words = set(answer.lower().split())
    truth_words = set(ground_truth.lower().split())

    if not truth_words:
        return 0.5

    overlap = len(answer_words & truth_words) / max(len(truth_words), 1)

    length_ratio = len(answer) / max(len(ground_truth), 1)
    if length_ratio < 0.1:
        length_penalty = 0.2
    elif length_ratio > 10:
        length_penalty = 0.7
    else:
        length_penalty = 1.0

    return min(overlap * length_penalty, 1.0)


async def generate_all_scores(
    questions_path: str = "training_data/questions.json",
    output_path: str = "training_data/scores.json",
    checkpoint_path: str = "training_data/scores_checkpoint.json",
    max_questions: int = 0,
    use_llm_judge: bool = False,
    batch_size: int = 5,
):
    """
    Generate quality scores for all models on all questions.
    
    Features:
    - Checkpoint/resume: saves after each batch
    - Parallel execution: all models run concurrently per question
    - Progress tracking with cost totals
    """
    with open(questions_path, "r", encoding="utf-8") as f:
        questions = json.load(f)
    if max_questions > 0:
        questions = questions[:max_questions]

    # Load checkpoint if exists
    completed_questions = set()
    all_scores = []
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            all_scores = json.load(f)
        completed_questions = {s["question"][:100] for s in all_scores}
        print(f"📂 Resuming from checkpoint: {len(completed_questions)} questions already done")

    remaining = [q for q in questions if q["question"][:100] not in completed_questions]

    print(f"\n{'='*55}")
    print(f"🔄 RouteMoA Score Generation")
    print(f"{'='*55}")
    print(f"  Models:    {len(MODELS)}")
    print(f"  Questions: {len(remaining)} remaining / {len(questions)} total")
    print(f"  LLM Judge: {'ON' if use_llm_judge else 'OFF (heuristic)'}")
    print(f"{'='*55}\n")

    total_cost = sum(s.get("cost", 0) for s in all_scores)
    total_api_calls = len(all_scores)

    for batch_start in range(0, len(remaining), batch_size):
        batch = remaining[batch_start : batch_start + batch_size]
        batch_num = batch_start // batch_size + 1
        total_batches = (len(remaining) + batch_size - 1) // batch_size

        print(f"\n📦 Batch {batch_num}/{total_batches} ({len(batch)} questions)")

        for q_idx, q in enumerate(batch):
            q_num = len(completed_questions) + q_idx + 1
            print(f"\n  📝 Q{q_num}: [{q['domain']}] {q['question'][:70]}...")

            # Run all models in parallel
            tasks = [
                evaluate_model_on_question(
                    m["id"], q["question"], q["ground_truth"],
                    answer_type=q.get("answer_type", "numeric"),
                    use_llm_judge=use_llm_judge,
                )
                for m in MODELS
            ]
            results = await asyncio.gather(*tasks)

            # Collect results
            for result in results:
                total_cost += result.get("cost", 0)
                total_api_calls += 1
                all_scores.append({
                    "question": q["question"],
                    "domain": q["domain"],
                    "dataset": q["dataset"],
                    "answer_type": q.get("answer_type", "numeric"),
                    **result,
                })

            # Show scores
            scored = [(r["model_id"].split("/")[-1][:15], r["quality_score"], r["accuracy"]) for r in results]
            scored.sort(key=lambda x: -x[1])
            errors = [r for r in results if r.get("error")]
            print(f"     🏆 Best: {scored[0][0]} (q={scored[0][1]:.2f}, acc={scored[0][2]:.0f})")
            print(f"     📉 Worst: {scored[-1][0]} (q={scored[-1][1]:.2f}, acc={scored[-1][2]:.0f})")
            if errors:
                print(f"     ⚠️ Errors: {len(errors)} models failed")

        # Checkpoint after each batch
        completed_questions.update(q["question"][:100] for q in batch)
        os.makedirs(os.path.dirname(checkpoint_path) if os.path.dirname(checkpoint_path) else ".", exist_ok=True)
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(all_scores, f, ensure_ascii=False, indent=2)
        print(f"\n  💾 Checkpoint saved ({len(completed_questions)} questions done)")
        print(f"  💰 Running cost: ${total_cost:.4f} ({total_api_calls} API calls)")

    # Save final output
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_scores, f, ensure_ascii=False, indent=2)

    # Summary
    print(f"\n{'='*55}")
    print(f"✅ Score Generation Complete!")
    print(f"{'='*55}")
    print(f"  📊 Total scores: {len(all_scores)}")
    print(f"  💰 Total cost:   ${total_cost:.4f}")
    print(f"  📡 API calls:    {total_api_calls}")

    # Per-model quality summary
    model_scores = {}
    for s in all_scores:
        mid = s["model_id"]
        if mid not in model_scores:
            model_scores[mid] = []
        model_scores[mid].append(s["quality_score"])

    print(f"\n  📈 Average Quality per Model:")
    for mid in sorted(model_scores, key=lambda k: -sum(model_scores[k])/len(model_scores[k])):
        scores = model_scores[mid]
        avg = sum(scores) / len(scores)
        name = mid.split("/")[-1]
        bar = "█" * int(avg * 20)
        print(f"     {name:.<25} {avg:.3f} {bar}")

    print(f"\n📁 Saved to {output_path}")
    return all_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate quality scores for RouteMoA training")
    parser.add_argument("--questions", type=str, default="training_data/questions.json")
    parser.add_argument("--output", type=str, default="training_data/scores.json")
    parser.add_argument("--max-questions", type=int, default=0, help="0 = all questions")
    parser.add_argument("--llm-judge", action="store_true", help="Use LLM-as-judge for reward scoring")
    parser.add_argument("--batch-size", type=int, default=5, help="Questions per checkpoint batch")
    args = parser.parse_args()

    asyncio.run(generate_all_scores(
        questions_path=args.questions,
        output_path=args.output,
        max_questions=args.max_questions,
        use_llm_judge=args.llm_judge,
        batch_size=args.batch_size,
    ))
