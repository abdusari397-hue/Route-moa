"""
RouteMoA Evaluator
Prompt templates for Self-Assessment, Cross-Assessment, and Final Aggregation.
Implements the Mixture of Judges (MoJ) from the paper.
"""
import json
import re
from typing import Optional


# ─────────────────────────────────────────────
# Prompt Templates
# ─────────────────────────────────────────────

LAYER1_SYSTEM_PROMPT = """You are an expert AI assistant. Answer the user's question thoroughly and accurately.

After providing your answer, evaluate your own confidence in the answer on a scale from 0.0 to 1.0.

You MUST respond in valid JSON format:
{
    "answer": "Your detailed answer here",
    "self_score": 0.85
}

Rules for self_score:
- 1.0 = Absolutely certain, verifiable fact
- 0.8-0.9 = Very confident, well-understood topic
- 0.6-0.7 = Moderately confident, some uncertainty
- 0.4-0.5 = Uncertain, might need verification
- 0.0-0.3 = Low confidence, speculative"""


def build_layer1_prompt(query: str) -> str:
    """Build the Layer 1 prompt (self-assessment only)."""
    return f"""Question: {query}

Provide your best answer and self-confidence score in JSON format:
{{"answer": "...", "self_score": <0.0-1.0>}}"""


def build_intermediate_prompt(
    query: str,
    previous_answers: list[dict],
    layer_num: int,
) -> str:
    """
    Build intermediate layer prompt (self-assessment + cross-assessment).
    Models see previous answers and must evaluate them as well.
    """
    answers_text = ""
    for i, ans in enumerate(previous_answers):
        model_name = ans.get("model_name", f"Model {i+1}")
        answer = ans.get("answer", "N/A")
        answers_text += f"\n--- Response from {model_name} ---\n{answer}\n"

    return f"""Question: {query}

Here are responses from other AI models in the previous layer:
{answers_text}

Your task (Layer {layer_num}):
1. Read the question and all previous responses carefully
2. Provide your own improved answer, incorporating the best insights from others
3. Rate your own confidence (self_score)
4. Rate each previous response's quality (peer_scores, in order)

Respond in JSON:
{{
    "answer": "Your improved, comprehensive answer",
    "self_score": <0.0-1.0>,
    "peer_scores": [<score for model 1>, <score for model 2>, ...]
}}

Be critical but fair when scoring peers. A score of 0.9+ should only be given for truly excellent, complete, and accurate answers."""


def build_cross_assessment_prompt(
    query: str,
    answers_to_judge: list[dict],
) -> str:
    """
    Build a cross-assessment prompt for the top-scoring model to judge others.
    This is the "judge" role in Mixture of Judges.
    """
    answers_text = ""
    for i, ans in enumerate(answers_to_judge):
        model_name = ans.get("model_name", f"Model {i+1}")
        answer = ans.get("answer", "N/A")
        answers_text += f"\n--- Response {i+1} from {model_name} ---\n{answer}\n"

    return f"""You are acting as an expert judge. Evaluate the following responses to this question:

Question: {query}

Responses to evaluate:
{answers_text}

For each response, provide:
1. A quality score (0.0 to 1.0)
2. A brief justification

Respond in JSON:
{{
    "scores": [<score1>, <score2>, ...],
    "justifications": ["reason1", "reason2", ...]
}}"""


def build_final_aggregation_prompt(
    query: str,
    all_answers: list[dict],
) -> str:
    """
    Build the final aggregation prompt.
    The best model synthesizes all answers into one high-quality response.
    """
    answers_text = ""
    for i, ans in enumerate(all_answers):
        model_name = ans.get("model_name", f"Model {i+1}")
        answer = ans.get("answer", "N/A")
        score = ans.get("final_score", "N/A")
        answers_text += f"\n--- Response from {model_name} (Score: {score}) ---\n{answer}\n"

    return f"""You are the final aggregator. Your job is to synthesize the best parts of all responses into a single, comprehensive, high-quality answer.

Question: {query}

All responses with their confidence scores:
{answers_text}

Instructions:
- Combine the strongest insights from all responses
- Resolve any contradictions by favoring higher-scored responses
- Ensure completeness and accuracy
- Provide a clear, well-structured final answer
- Do NOT mention the individual models or scores in your answer

Provide your synthesized answer directly (no JSON wrapper needed)."""


# ─────────────────────────────────────────────
# Response Parsing
# ─────────────────────────────────────────────

def extract_json(text: str) -> Optional[dict]:
    """
    Robustly extract JSON from LLM output.
    Handles: direct JSON, code fences, embedded JSON, malformed output.
    """
    if not text:
        return None

    # Remove thinking tags if present (some models use <think>...</think>)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = text.strip()

    # Strategy 1: Direct parse
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

    # Strategy 2: Code fence extraction
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Strategy 3: Find outermost { ... }
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    start = -1

    # Strategy 4: Fallback — try to extract answer and score with regex
    answer_match = re.search(r'"answer"\s*:\s*"(.*?)"', text, re.DOTALL)
    score_match = re.search(r'"self_score"\s*:\s*([\d.]+)', text)
    if answer_match:
        result = {"answer": answer_match.group(1)}
        if score_match:
            result["self_score"] = float(score_match.group(1))
        return result

    return None


def parse_layer1_response(raw_answer: str) -> dict:
    """
    Parse a Layer 1 response, extracting answer and self_score.
    Falls back to using the raw text as the answer if JSON fails.
    """
    parsed = extract_json(raw_answer)
    if parsed and "answer" in parsed:
        return {
            "answer": parsed["answer"],
            "self_score": float(parsed.get("self_score", 0.5)),
        }
    # Fallback: use raw text as answer
    return {
        "answer": raw_answer,
        "self_score": 0.5,
    }


def parse_intermediate_response(raw_answer: str, num_peers: int = 0) -> dict:
    """
    Parse an intermediate layer response with self and peer scores.
    """
    parsed = extract_json(raw_answer)
    if parsed and "answer" in parsed:
        peer_scores = parsed.get("peer_scores", [0.5] * num_peers)
        # Ensure peer_scores is the right length
        if len(peer_scores) < num_peers:
            peer_scores.extend([0.5] * (num_peers - len(peer_scores)))
        return {
            "answer": parsed["answer"],
            "self_score": float(parsed.get("self_score", 0.5)),
            "peer_scores": [float(s) for s in peer_scores[:num_peers]],
        }
    return {
        "answer": raw_answer,
        "self_score": 0.5,
        "peer_scores": [0.5] * num_peers,
    }


def parse_cross_assessment(raw_answer: str, num_models: int = 0) -> list[float]:
    """
    Parse cross-assessment scores from the judge model.
    """
    parsed = extract_json(raw_answer)
    if parsed and "scores" in parsed:
        scores = [float(s) for s in parsed["scores"]]
        if len(scores) < num_models:
            scores.extend([0.5] * (num_models - len(scores)))
        return scores[:num_models]
    return [0.5] * num_models
