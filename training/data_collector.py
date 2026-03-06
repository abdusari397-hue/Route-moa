"""
RouteMoA Training — Data Collector
Fetches QA datasets from HuggingFace for training the SLM Scorer.

Datasets aligned with the paper:
  - GSM8K    (math reasoning)
  - ARC-C    (science/reasoning - Challenge set)
  - MMLU     (multi-domain knowledge: biology, CS, law, etc.)
  - MBPP     (Python coding)
"""
import json
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def collect_gsm8k(max_samples: int) -> list[dict]:
    """GSM8K — Grade School Math (chain-of-thought reasoning)."""
    from datasets import load_dataset
    samples = []
    try:
        print("📚 Loading GSM8K...")
        ds = load_dataset("openai/gsm8k", "main", split="test")
        for i, row in enumerate(ds):
            if i >= max_samples:
                break
            # Ground truth is the final numeric answer after ####
            answer_text = row["answer"]
            # Extract the final number
            final_answer = answer_text.split("####")[-1].strip() if "####" in answer_text else answer_text
            samples.append({
                "question": row["question"],
                "ground_truth": final_answer,
                "ground_truth_full": answer_text,
                "domain": "math",
                "dataset": "gsm8k",
                "answer_type": "numeric",
            })
        print(f"  ✅ Loaded {len(samples)} GSM8K questions")
    except Exception as e:
        print(f"  ⚠️ Failed to load GSM8K: {e}")
    return samples


def collect_arc_challenge(max_samples: int) -> list[dict]:
    """ARC-Challenge — Science reasoning (multiple choice)."""
    from datasets import load_dataset
    samples = []
    try:
        print("📚 Loading ARC-Challenge...")
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
        for i, row in enumerate(ds):
            if i >= max_samples:
                break
            choices = row["choices"]
            labels = choices["label"]
            texts = choices["text"]
            options_str = "\n".join(f"  {l}) {t}" for l, t in zip(labels, texts))
            question = f"{row['question']}\n\nOptions:\n{options_str}"

            # Ground truth is the label (A, B, C, D)
            answer_key = row["answerKey"]
            # Also get the full answer text
            answer_idx = labels.index(answer_key) if answer_key in labels else 0
            answer_text = texts[answer_idx]

            samples.append({
                "question": question,
                "ground_truth": answer_key,
                "ground_truth_full": f"{answer_key}) {answer_text}",
                "domain": "science",
                "dataset": "arc_challenge",
                "answer_type": "multiple_choice",
            })
        print(f"  ✅ Loaded {len(samples)} ARC-Challenge questions")
    except Exception as e:
        print(f"  ⚠️ Failed to load ARC-Challenge: {e}")
    return samples


def collect_mmlu(max_samples: int) -> list[dict]:
    """
    MMLU — Massive Multitask Language Understanding.
    Sample from diverse subjects: biology, computer science, law, history.
    """
    from datasets import load_dataset
    samples = []

    # Target subjects for diversity
    subjects = [
        ("anatomy", "medical"),
        ("college_biology", "medical"),
        ("college_computer_science", "coding"),
        ("computer_security", "coding"),
        ("high_school_mathematics", "math"),
        ("abstract_algebra", "math"),
        ("international_law", "reasoning"),
        ("world_religions", "reasoning"),
    ]

    per_subject = max(max_samples // len(subjects), 5)

    try:
        print("📚 Loading MMLU...")
        for subject, domain in subjects:
            try:
                ds = load_dataset("cais/mmlu", subject, split="test")
                count = 0
                for row in ds:
                    if count >= per_subject:
                        break
                    choices = row["choices"]
                    options_str = "\n".join(
                        f"  {chr(65+j)}) {c}" for j, c in enumerate(choices)
                    )
                    question = f"{row['question']}\n\nOptions:\n{options_str}"
                    answer_idx = row["answer"]  # integer 0-3
                    answer_key = chr(65 + answer_idx)
                    answer_text = choices[answer_idx]

                    samples.append({
                        "question": question,
                        "ground_truth": answer_key,
                        "ground_truth_full": f"{answer_key}) {answer_text}",
                        "domain": domain,
                        "dataset": f"mmlu_{subject}",
                        "answer_type": "multiple_choice",
                    })
                    count += 1
                print(f"  ✅ {subject}: {count} questions")
            except Exception as e:
                print(f"  ⚠️ {subject}: {e}")

        print(f"  📊 Total MMLU: {len(samples)} questions")
    except Exception as e:
        print(f"  ⚠️ Failed to load MMLU: {e}")
    return samples


def collect_mbpp(max_samples: int) -> list[dict]:
    """MBPP — Mostly Basic Python Problems."""
    from datasets import load_dataset
    samples = []
    try:
        print("📚 Loading MBPP...")
        ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
        for i, row in enumerate(ds):
            if i >= max_samples:
                break
            # Include test cases in the question for evaluation
            test_list = row.get("test_list", [])
            tests_str = "\n".join(test_list[:3]) if test_list else ""
            question = row["prompt"]
            if tests_str:
                question += f"\n\nTest cases:\n{tests_str}"

            samples.append({
                "question": question,
                "ground_truth": row["code"],
                "ground_truth_full": row["code"],
                "domain": "coding",
                "dataset": "mbpp",
                "answer_type": "code",
            })
        print(f"  ✅ Loaded {len(samples)} MBPP questions")
    except Exception as e:
        print(f"  ⚠️ Failed to load MBPP: {e}")
    return samples


def collect_training_data(
    output_path: str = "training_data/questions.json",
    max_per_dataset: int = 50,
) -> list[dict]:
    """
    Collect questions from all target datasets.
    
    Returns list of:
        {
            "question": str,
            "ground_truth": str,         # Short answer (number, letter, code)
            "ground_truth_full": str,     # Full answer with explanation
            "domain": str,               # math, science, coding, medical, reasoning
            "dataset": str,              # Dataset source name
            "answer_type": str,          # numeric, multiple_choice, code
        }
    """
    all_questions = []

    print("=" * 55)
    print("📚 RouteMoA Data Collection Pipeline")
    print("=" * 55)

    # Collect from each dataset
    all_questions.extend(collect_gsm8k(max_per_dataset))
    all_questions.extend(collect_arc_challenge(max_per_dataset))
    all_questions.extend(collect_mmlu(max_per_dataset))
    all_questions.extend(collect_mbpp(max_per_dataset))

    # Summary
    print(f"\n{'='*55}")
    domains = {}
    for q in all_questions:
        d = q["domain"]
        domains[d] = domains.get(d, 0) + 1
    print(f"📊 Total: {len(all_questions)} questions")
    for domain, count in sorted(domains.items()):
        print(f"   {domain}: {count}")

    # Save
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_questions, f, ensure_ascii=False, indent=2)
    print(f"\n📁 Saved to {output_path}")

    return all_questions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect training data for RouteMoA SLM Scorer")
    parser.add_argument("--max-per-dataset", type=int, default=50, help="Max samples per dataset")
    parser.add_argument("--output", type=str, default="training_data/questions.json", help="Output path")
    args = parser.parse_args()
    collect_training_data(output_path=args.output, max_per_dataset=args.max_per_dataset)
