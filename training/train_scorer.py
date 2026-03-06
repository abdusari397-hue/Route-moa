"""
RouteMoA Training — SLM Scorer Training Loop

Uses the dual contrastive loss to train the mDeBERTaV3-based scorer
to predict which models will perform best on a given query.

Supports CUDA GPU acceleration (RTX 2060+).
"""
import json
import os
import sys
import argparse
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import MODELS, get_model_index
from core.scorer import RouteMoAScorer
from training.contrastive_loss import DualContrastiveLoss


# ─────────────────────────────────────────────
# Device selection
# ─────────────────────────────────────────────
def get_device() -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"🚀 Using GPU: {name} ({mem:.1f} GB)")
    else:
        dev = torch.device("cpu")
        print("💻 Using CPU (no CUDA GPU detected)")
    return dev


class ScorerDataset(Dataset):
    """
    Dataset for training the SLM scorer.
    Each sample: (question_text, quality_scores_per_model)
    """

    def __init__(self, scores_path: str = "training_data/scores.json"):
        with open(scores_path, "r", encoding="utf-8") as f:
            raw_scores = json.load(f)

        # Group scores by question
        question_map = {}
        for entry in raw_scores:
            q = entry["question"]
            if q not in question_map:
                question_map[q] = {
                    "question": q,
                    "domain": entry.get("domain", "general"),
                    "scores": [0.0] * len(MODELS),
                }
            idx = get_model_index(entry["model_id"])
            if idx >= 0:
                question_map[q]["scores"][idx] = entry["quality_score"]

        self.samples = list(question_map.values())

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return sample["question"], torch.tensor(sample["scores"], dtype=torch.float32)


def train_scorer(
    scores_path: str = "training_data/scores.json",
    output_path: str = "scorer_model",
    epochs: int = 20,
    batch_size: int = 8,
    lr: float = 2e-5,
    n_clusters: int = 10,
    alpha: float = 0.5,
    temperature: float = 0.07,
):
    """
    Full training loop for the RouteMoA SLM scorer with GPU support.
    """
    print("=" * 55)
    print("🏋️ RouteMoA Scorer Training")
    print("=" * 55)

    device = get_device()
    start_time = time.time()

    # Load data
    dataset = ScorerDataset(scores_path)
    print(f"📊 Loaded {len(dataset)} training samples")

    if len(dataset) == 0:
        print("❌ No training data found! Run generate_scores.py first.")
        return

    # Initialize model and move to GPU
    scorer = RouteMoAScorer(num_models=len(MODELS))
    scorer.encoder = scorer.encoder.to(device)
    scorer.model_embeddings = nn.Parameter(scorer.model_embeddings.data.to(device))
    tokenizer = scorer.tokenizer

    # Loss function
    loss_fn = DualContrastiveLoss(temperature=temperature, alpha=alpha)

    # Optimizer (only train model_embeddings, not the frozen encoder)
    optimizer = torch.optim.Adam([scorer.model_embeddings], lr=lr)

    # Pre-compute query embeddings for clustering
    print("📐 Computing query embeddings on GPU...")
    all_embeddings = []
    all_scores = []

    with torch.no_grad():
        for question, scores in dataset:
            tokens = tokenizer(
                question,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True,
            ).to(device)
            emb = scorer.encoder(
                input_ids=tokens["input_ids"],
                attention_mask=tokens["attention_mask"],
            ).last_hidden_state[:, 0, :]  # [CLS] token
            all_embeddings.append(emb.squeeze(0).float())
            all_scores.append(scores.to(device).float())

    all_embeddings_tensor = torch.stack(all_embeddings)       # [N, hidden_dim] on GPU
    all_scores_tensor = torch.stack(all_scores)               # [N, num_models] on GPU

    embed_time = time.time() - start_time
    print(f"   Done in {embed_time:.1f}s — shape: {all_embeddings_tensor.shape}")

    # K-means clustering on query embeddings (CPU — sklearn)
    k = min(n_clusters, len(dataset))
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(all_embeddings_tensor.cpu().numpy())
    cluster_labels_tensor = torch.tensor(cluster_labels, dtype=torch.long, device=device)
    print(f"📊 Created {k} query clusters")

    # Training loop
    print(f"\n🔄 Training for {epochs} epochs (batch_size={batch_size})...\n")
    best_loss = float("inf")
    patience = 5
    patience_counter = 0

    for epoch in range(epochs):
        epoch_start = time.time()
        total_loss = 0.0
        loss_details = {"l_query_model": 0.0, "l_query_query": 0.0}
        n_batches = 0

        # Process in batches
        indices = torch.randperm(len(dataset), device=device)
        for start in range(0, len(dataset), batch_size):
            end = min(start + batch_size, len(dataset))
            batch_idx = indices[start:end]

            batch_emb = all_embeddings_tensor[batch_idx]
            batch_scores = all_scores_tensor[batch_idx]
            batch_clusters = cluster_labels_tensor[batch_idx]

            optimizer.zero_grad()
            loss, details = loss_fn(
                query_embeddings=batch_emb,
                model_embeddings=scorer.model_embeddings,
                quality_scores=batch_scores,
                cluster_labels=batch_clusters,
            )
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            for key in loss_details:
                loss_details[key] += details.get(key, 0)

        avg_loss = total_loss / max(1, n_batches)
        epoch_time = time.time() - epoch_start

        # Log
        qm = loss_details["l_query_model"] / max(1, n_batches)
        qq = loss_details["l_query_query"] / max(1, n_batches)
        status = ""

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            # Save best model (move to CPU for saving)
            scorer_cpu = RouteMoAScorer(num_models=len(MODELS))
            scorer_cpu.model_embeddings = nn.Parameter(scorer.model_embeddings.data.cpu())
            scorer_cpu.save(output_path)
            status = " ✅ SAVED"
        else:
            patience_counter += 1

        print(
            f"  Epoch {epoch+1:>2}/{epochs}: "
            f"loss={avg_loss:.4f} (qm={qm:.4f}, qq={qq:.4f}) "
            f"[{epoch_time:.1f}s]{status}"
        )

        # Early stopping
        if patience_counter >= patience:
            print(f"\n  ⏹️ Early stopping: no improvement for {patience} epochs")
            break

    total_time = time.time() - start_time
    print(f"\n{'='*55}")
    print(f"🎉 Training complete!")
    print(f"{'='*55}")
    print(f"  Best loss:  {best_loss:.4f}")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Device:     {device}")
    print(f"  📁 Model saved to {output_path}/")

    # Quick validation: show predicted scores for a test query
    print(f"\n📋 Quick validation (sample predictions):")
    test_queries = [
        "Write a Python function to sort a list",
        "What is the capital of France?",
        "Solve: 2x + 5 = 15",
    ]
    scorer.eval()
    with torch.no_grad():
        for q in test_queries:
            tokens = tokenizer(q, return_tensors="pt", max_length=512, truncation=True, padding=True).to(device)
            emb = scorer.encoder(input_ids=tokens["input_ids"], attention_mask=tokens["attention_mask"]).last_hidden_state[:, 0, :].float()
            preds = torch.sigmoid(emb @ scorer.model_embeddings.T).squeeze(0)
            top3_idx = preds.argsort(descending=True)[:3]
            top3 = [(MODELS[i]["name"], preds[i].item()) for i in top3_idx]
            print(f"  Q: \"{q[:50]}\"")
            for name, score in top3:
                print(f"     {name}: {score:.3f}")
            print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RouteMoA SLM Scorer")
    parser.add_argument("--scores", default="training_data/scores.json", help="Scores file path")
    parser.add_argument("--output", default="scorer_model", help="Output directory")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--clusters", type=int, default=10, help="K-means clusters")
    args = parser.parse_args()

    train_scorer(
        scores_path=args.scores,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        n_clusters=args.clusters,
    )
