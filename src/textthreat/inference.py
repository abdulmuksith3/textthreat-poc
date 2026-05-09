"""Inference helpers for the hosted TextThreat demo."""

from __future__ import annotations

import os
import inspect
import re
from pathlib import Path
from typing import Any

import numpy as np
import joblib

from .constants import LABELS, MODEL_VERSION, STRESS_LABEL
from .data import normalize_social_text
from .utils import MODELS_DIR


SAFETY_OVERLAY_RULES = [
    (
        re.compile(r"\b(i\s+(want|wanna|wish|need)\s+(to\s+)?die|kill\s+myself|suicide|end\s+my\s+life)\b", re.I),
        {STRESS_LABEL: 0.95},
    ),
    (
        re.compile(r"\b(kill\s+you|hurt\s+you|i\s+will\s+kill|i'?m\s+going\s+to\s+kill|death\s+threat)\b", re.I),
        {"threat": 0.95, "toxic": 0.85},
    ),
    (
        re.compile(r"\b(fuck|fucking|shit|bitch|asshole|cunt)\b", re.I),
        {"obscene": 0.80, "toxic": 0.70},
    ),
]


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
            from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
        except ImportError:
            return None
        try:
            model_path = Path(model_id)
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            has_full_weights = (model_path / "model.safetensors").exists() or (model_path / "pytorch_model.bin").exists()
            if (model_path / "adapter_config.json").exists() and not has_full_weights:
                try:
                    from peft import PeftModel
                except ImportError:
                    return None
                adapter_config = AutoConfig.from_pretrained(model_id)
                base_model_name = "distilbert-base-uncased"
                try:
                    import json

                    payload = json.loads((model_path / "adapter_config.json").read_text(encoding="utf-8"))
                    base_model_name = payload.get("base_model_name_or_path") or base_model_name
                except Exception:
                    pass
                is_stress_model = local_subdir == "distilbert_dreaddit"
                id2label = (
                    {0: "not_stress", 1: STRESS_LABEL}
                    if is_stress_model
                    else {idx: label for idx, label in enumerate(LABELS)}
                )
                label2id = (
                    {"not_stress": 0, STRESS_LABEL: 1}
                    if is_stress_model
                    else {label: idx for idx, label in enumerate(LABELS)}
                )
                base_model = AutoModelForSequenceClassification.from_pretrained(
                    base_model_name,
                    num_labels=2 if is_stress_model else len(LABELS),
                    problem_type="single_label_classification" if is_stress_model else "multi_label_classification",
                    id2label=id2label,
                    label2id=label2id,
                )
                base_model.config.update(adapter_config.to_dict())
                model = PeftModel.from_pretrained(base_model, model_id)
            else:
                model = AutoModelForSequenceClassification.from_pretrained(model_id)
            model.eval()
            calibrators = None
            model_path = Path(model_id)
            for filename in ["platt_calibrators.joblib", "platt_stress_calibrator.joblib"]:
                calibrator_path = model_path / filename
                if calibrator_path.exists():
                    calibrators = joblib.load(calibrator_path)
                    break
            return {"tokenizer": tokenizer, "model": model, "torch": torch, "model_id": model_id, "calibrators": calibrators}
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
        prepared_text = normalize_social_text(text)
        inputs = tokenizer(prepared_text, truncation=True, padding=True, max_length=231, return_tensors="pt")
        accepted_inputs = inspect.signature(model.forward).parameters
        inputs = {key: value for key, value in inputs.items() if key in accepted_inputs}
        with torch.no_grad():
            logits = model(**inputs).logits.detach().cpu().numpy()[0]
        labels = [model.config.id2label.get(idx, LABELS[idx] if idx < len(LABELS) else str(idx)) for idx in range(len(logits))]
        calibrators = bundle.get("calibrators")
        if len(logits) == 2 and STRESS_LABEL in labels:
            if calibrators is not None and hasattr(calibrators, "predict_proba"):
                stress_probability = calibrators.predict_proba(np.asarray([[logits[1] - logits[0]]]))[0, 1]
            elif isinstance(calibrators, dict) and calibrators.get("type") == "constant":
                stress_probability = calibrators["probability"]
            else:
                exp = np.exp(logits - np.max(logits))
                probs = exp / exp.sum()
                stress_probability = probs[labels.index(STRESS_LABEL)]
            return {STRESS_LABEL: float(stress_probability)}
        if isinstance(calibrators, dict):
            calibrated_scores = {}
            for index, label in enumerate(labels):
                calibrator = calibrators.get(label)
                if label not in LABELS or calibrator is None:
                    continue
                if isinstance(calibrator, dict) and calibrator.get("type") == "constant":
                    calibrated_scores[label] = float(calibrator["probability"])
                else:
                    calibrated_scores[label] = float(calibrator.predict_proba(np.asarray([[logits[index]]]))[0, 1])
            if calibrated_scores:
                return calibrated_scores
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

    def _apply_safety_overlay(self, text: str, scores: dict[str, float]) -> bool:
        """Apply transparent minimum scores for explicit safety-critical phrases."""
        matched = False
        for pattern, floors in SAFETY_OVERLAY_RULES:
            if not pattern.search(text):
                continue
            matched = True
            for label, floor in floors.items():
                scores[label] = max(float(scores.get(label, 0.0)), floor)
        return matched

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
        if self._apply_safety_overlay(text, scores):
            modes.append("safety_lexical_overlay")
        return {"scores": scores, "model_version": MODEL_VERSION, "inference_modes": modes}
