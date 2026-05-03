"""Inference helpers for the hosted TextThreat demo."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from .constants import LABELS, MODEL_VERSION, STRESS_LABEL
from .utils import MODELS_DIR


class TextThreatAnalyzer:
    """Load trained models when available and provide deterministic fallback scoring."""

    def __init__(
        self,
        toxicity_model_id: str | None = None,
        stress_model_id: str | None = None,
    ) -> None:
        self.toxicity_model_id = toxicity_model_id or os.getenv("TEXTTHREAT_TOXICITY_MODEL_ID")
        self.stress_model_id = stress_model_id or os.getenv("TEXTTHREAT_STRESS_MODEL_ID")
        self._toxicity = None
        self._stress = None
        self._toxicity_loaded = False
        self._stress_loaded = False

    def _default_local_model(self, subdir: str) -> str | None:
        auto_load = os.getenv("TEXTTHREAT_AUTO_LOAD_LOCAL_MODELS", "").lower() in {"1", "true", "yes"}
        if not auto_load:
            return None
        path = MODELS_DIR / subdir
        if (path / "config.json").exists():
            return str(path)
        return None

    def _load_sequence_model(self, model_id: str | None, local_subdir: str):
        if not model_id:
            model_id = self._default_local_model(local_subdir)
        if not model_id:
            return None
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError:
            return None
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForSequenceClassification.from_pretrained(model_id)
            model.eval()
            return {"tokenizer": tokenizer, "model": model, "torch": torch, "model_id": model_id}
        except Exception:
            return None

    @property
    def toxicity_model(self):
        if not self._toxicity_loaded:
            self._toxicity = self._load_sequence_model(self.toxicity_model_id, "distilbert_jigsaw")
            self._toxicity_loaded = True
        return self._toxicity

    @property
    def stress_model(self):
        if not self._stress_loaded:
            self._stress = self._load_sequence_model(self.stress_model_id, "distilbert_dreaddit")
            self._stress_loaded = True
        return self._stress

    def _model_scores(self, bundle, text: str) -> dict[str, float]:
        tokenizer = bundle["tokenizer"]
        model = bundle["model"]
        torch = bundle["torch"]
        inputs = tokenizer(text, truncation=True, padding=True, max_length=256, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits.detach().cpu().numpy()[0]
        labels = [model.config.id2label.get(idx, LABELS[idx] if idx < len(LABELS) else str(idx)) for idx in range(len(logits))]
        if len(logits) == 2 and STRESS_LABEL in labels:
            exp = np.exp(logits - np.max(logits))
            probs = exp / exp.sum()
            return {STRESS_LABEL: float(probs[labels.index(STRESS_LABEL)])}
        probs = 1.0 / (1.0 + np.exp(-logits))
        return {label: float(prob) for label, prob in zip(labels, probs) if label in LABELS}

    def _fallback_scores(self, text: str) -> dict[str, float]:
        lowered = text.lower()
        toxicity_terms = ["hostile", "harass", "attack", "bully", "threat", "insult", "abuse"]
        stress_terms = ["stressed", "panic", "anxious", "overwhelmed", "crisis", "distress", "scared"]
        toxicity_hits = sum(term in lowered for term in toxicity_terms)
        stress_hits = sum(term in lowered for term in stress_terms)
        base = min(0.35, len(text) / 1000.0)
        return {
            "toxic": min(0.98, 0.12 + base + 0.22 * toxicity_hits),
            "severe_toxic": min(0.95, 0.04 + 0.16 * max(0, toxicity_hits - 1)),
            "obscene": min(0.92, 0.05 + 0.13 * ("abuse" in lowered)),
            "threat": min(0.98, 0.06 + 0.34 * ("threat" in lowered or "scared" in lowered)),
            "insult": min(0.96, 0.08 + 0.22 * ("insult" in lowered or "bully" in lowered)),
            "identity_hate": min(0.90, 0.04 + 0.12 * ("hate" in lowered)),
            STRESS_LABEL: min(0.98, 0.10 + base + 0.24 * stress_hits),
        }

    def analyze(self, text: str) -> dict[str, Any]:
        """Return harm scores and inference metadata for one comment."""
        scores: dict[str, float] = {}
        modes = []
        toxicity = self.toxicity_model
        if toxicity:
            scores.update(self._model_scores(toxicity, text))
            modes.append(f"toxicity_model:{toxicity['model_id']}")
        stress = self.stress_model
        if stress:
            scores.update(self._model_scores(stress, text))
            modes.append(f"stress_model:{stress['model_id']}")
        if not scores:
            scores = self._fallback_scores(text)
            modes.append("demo_keyword_fallback")
        elif STRESS_LABEL not in scores:
            scores[STRESS_LABEL] = self._fallback_scores(text)[STRESS_LABEL]
            modes.append("stress_demo_fallback")
        return {"scores": scores, "model_version": MODEL_VERSION, "inference_modes": modes}
