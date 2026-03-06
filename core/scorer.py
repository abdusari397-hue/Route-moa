"""
RouteMoA SLM Scorer
Lightweight mDeBERTaV3-based scorer that predicts model performance
BEFORE running inference — the core innovation of the paper.

Formula: s = σ(E(x)ᵀ · kⱼ)
Where E(x) = query embedding, kⱼ = learnable model embedding
"""
import os
import torch
import torch.nn as nn
from typing import Optional

from config import MODELS


class RouteMoAScorer(nn.Module):
    """
    Predicts how well each model in the pool will perform on a given query,
    using a small language model (mDeBERTaV3-base) encoder and learnable
    per-model embeddings.
    
    Architecture:
        1. Encode the query text → [CLS] embedding (768-dim)
        2. Multiply by per-model embedding vectors
        3. Sigmoid → probability scores in [0, 1]
    """

    def __init__(self, num_models: int = None, hidden_dim: int = 768):
        super().__init__()
        if num_models is None:
            num_models = len(MODELS)

        self.num_models = num_models
        self.hidden_dim = hidden_dim
        self._encoder = None
        self._tokenizer = None

        # Learnable embedding for each model in the pool
        self.model_embeddings = nn.Parameter(
            torch.randn(num_models, hidden_dim) * 0.02
        )

    @property
    def encoder(self):
        """Lazy load the encoder to avoid downloading at import time."""
        if self._encoder is None:
            from transformers import AutoModel
            self._encoder = AutoModel.from_pretrained("microsoft/mdeberta-v3-base")
            self._encoder.eval()
        return self._encoder

    @encoder.setter
    def encoder(self, value):
        """Allow setting the encoder (e.g., to move to GPU)."""
        self._encoder = value

    @property
    def tokenizer(self):
        """Lazy load the tokenizer."""
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained("microsoft/mdeberta-v3-base")
        return self._tokenizer

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for training.

        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)

        Returns:
            scores: (batch, num_models) — predicted performance in [0, 1]
        """
        # Get [CLS] token embedding
        with torch.no_grad() if not self.training else torch.enable_grad():
            encoder_output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            query_emb = encoder_output.last_hidden_state[:, 0, :].float()  # (batch, hidden_dim)

        # Score each model: s = σ(E(x)ᵀ · kⱼ)
        scores = torch.sigmoid(query_emb @ self.model_embeddings.T.float())  # (batch, num_models)
        return scores

    @torch.no_grad()
    def predict(self, query: str) -> torch.Tensor:
        """
        Predict model scores for a single query string.

        Args:
            query: The input question/prompt text

        Returns:
            scores: (1, num_models) tensor of predicted performance scores
        """
        self.eval()
        tokens = self.tokenizer(
            query,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True,
        )
        scores = self.forward(tokens["input_ids"], tokens["attention_mask"])
        return scores

    def predict_as_dict(self, query: str) -> dict[str, float]:
        """
        Predict and return a dict mapping model_id → predicted score.
        """
        scores = self.predict(query).squeeze(0)  # (num_models,)
        return {
            MODELS[i]["id"]: round(scores[i].item(), 4)
            for i in range(min(len(MODELS), self.num_models))
        }

    def save(self, path: str = "scorer_model"):
        """Save model embeddings (not the frozen encoder)."""
        os.makedirs(path, exist_ok=True)
        torch.save(
            {"model_embeddings": self.model_embeddings.data, "num_models": self.num_models},
            os.path.join(path, "scorer_weights.pt"),
        )
        print(f"[Scorer] Saved to {path}/scorer_weights.pt")

    def load(self, path: str = "scorer_model") -> bool:
        """Load pre-trained model embeddings. Returns True if successful."""
        weight_path = os.path.join(path, "scorer_weights.pt")
        if not os.path.exists(weight_path):
            print(f"[Scorer] No checkpoint found at {weight_path}")
            return False
        checkpoint = torch.load(weight_path, map_location="cpu", weights_only=True)
        self.model_embeddings.data = checkpoint["model_embeddings"]
        print(f"[Scorer] Loaded from {weight_path}")
        return True


class FallbackScorer:
    """
    Simple fallback scorer that uses heuristic rules instead of mDeBERTaV3.
    Used when the SLM model is not available or not trained yet.
    
    Assigns scores based on keyword matching to model specialties.
    """

    KEYWORD_MAP = {
        "coding_math": [
            "code", "python", "javascript", "program", "function", "algorithm",
            "debug", "math", "calculate", "equation", "solve", "proof",
            "compile", "syntax", "API", "database", "SQL", "regex",
        ],
        "multilingual": [
            "translate", "french", "spanish", "arabic", "chinese", "german",
            "language", "multilingual", "localize",
        ],
    }

    def predict_as_dict(self, query: str) -> dict[str, float]:
        """
        Return heuristic scores for each model based on query keywords.
        """
        query_lower = query.lower()
        scores = {}

        # Detect dominant category
        category_hits = {}
        for category, keywords in self.KEYWORD_MAP.items():
            hits = sum(1 for kw in keywords if kw.lower() in query_lower)
            category_hits[category] = hits

        best_category = max(category_hits, key=category_hits.get) if any(category_hits.values()) else None

        for model in MODELS:
            base_score = 0.5  # Default neutral score

            # Boost models that match the detected specialty
            if best_category and model["specialty"] == best_category:
                base_score = 0.8
            elif model["specialty"] == "general":
                base_score = 0.6

            # Slight variance by model size (larger → higher potential)
            size_bonus = 0.0
            model_id = model["id"]
            if "397b" in model_id or "large" in model_id:
                size_bonus = 0.1
            elif "122b" in model_id:
                size_bonus = 0.05

            scores[model["id"]] = round(min(base_score + size_bonus, 1.0), 4)

        return scores
