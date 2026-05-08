---
license: mit
base_model: distilbert-base-uncased
library_name: transformers
pipeline_tag: text-classification
tags:
- distilbert
- transformers
- text-classification
- multi-label-classification
- toxicity-detection
- digital-wellbeing
- cybersecurity-analytics
- siem
- splunk
- textthreat
---

# TextThreat DistilBERT Jigsaw Classifier

This model is part of **TextThreat - AI-Powered Detection of Digital Well-Being Risks with Cybersecurity Analytics**, an MSc thesis proof-of-concept by Abdul Muksith Rizvi at the University of Doha for Science and Technology.

TextThreat detects digital well-being risk signals in social-media text and exports schema-valid cybersecurity-style events for SIEM analytics, with a Splunk-first hosted demo path.

## Model Details

- **Model type:** DistilBERT sequence classifier
- **Base model:** `distilbert-base-uncased`
- **Task:** Jigsaw Toxic Comment multi-label classification
- **Labels:** `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`
- **Problem type:** multi-label classification
- **Training method:** LoRA/PEFT fine-tuning, exported as a merged full Hugging Face model
- **Repository:** https://github.com/abdulmuksith3/textthreat-poc
- **Thesis system:** TextThreat proof-of-concept for harm detection, SIEM-ready event export, Splunk/OpenSearch analytics, SOAR-lite alerting, latency, calibration, privacy perturbation, and fairness audit artifacts

## Intended Use

This model is intended for the TextThreat proof-of-concept pipeline:

```text
social media comment
-> DistilBERT harm scoring
-> TextThreat JSON event schema
-> NDJSON / Splunk HEC export
-> SIEM dashboard and optional SOAR-lite alert
```

The model can be used to generate per-label harm probabilities for research demonstrations and thesis artifact reproduction.

## Out-of-Scope Use

This model is not intended for autonomous moderation, clinical risk assessment, law-enforcement decision-making, employment screening, or other high-stakes decisions. It should not be used as the only basis for action against a person.

## Evaluation

The current uploaded artifact corresponds to the thesis proof-of-concept training run. Metrics are stored in the repository under `experiments/results/distilbert_metrics.json`.

| Metric | Value |
|---|---:|
| Micro F1 | 0.7363 |
| Macro F1 | 0.4206 |
| Macro ROC-AUC | 0.9780 |
| Macro PR-AUC | 0.5324 |
| Expected Calibration Error | 0.0032 |
| Eval loss | 0.0471 |

## Limitations

- The model is trained for thesis proof-of-concept evidence, not as a production moderation service.
- Toxicity labels come from the Jigsaw Toxic Comment task and may not cover all digital well-being risks.
- Short, adversarial, sarcastic, reclaimed, or context-dependent text can be misclassified.
- The live TextThreat demo applies a transparent safety lexical overlay for explicit threats, self-harm terms, and profanity so obvious demo-critical safety cases are not missed by the quick-trained model.
- Fairness, calibration, and privacy experiments are represented in the companion repository and should be reviewed before deployment-style use.

## Example

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_id = "abdulmuksith/textthreat-distilbert-jigsaw"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)

text = "I will kill you"
inputs = tokenizer(text, return_tensors="pt", truncation=True)
inputs = {k: v for k, v in inputs.items() if k in model.forward.__code__.co_varnames}

with torch.no_grad():
    logits = model(**inputs).logits[0]

scores = torch.sigmoid(logits)
print({model.config.id2label[i]: float(scores[i]) for i in range(len(scores))})
```

## Citation

If referencing this model, cite the thesis project:

```text
Rizvi, A. M. (2026). TextThreat: AI-Powered Detection of Digital Well-Being Risks with Cybersecurity Analytics. MSc thesis, University of Doha for Science and Technology.
```
