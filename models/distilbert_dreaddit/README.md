---
license: mit
base_model: distilbert-base-uncased
library_name: transformers
pipeline_tag: text-classification
tags:
- distilbert
- transformers
- text-classification
- binary-classification
- stress-detection
- digital-wellbeing
- cybersecurity-analytics
- siem
- splunk
- textthreat
---

# TextThreat DistilBERT Dreaddit Stress Classifier

This model is part of **TextThreat - AI-Powered Detection of Digital Well-Being Risks with Cybersecurity Analytics**, an MSc thesis proof-of-concept by Abdul Muksith Rizvi at the University of Doha for Science and Technology.

TextThreat combines AI harm detection with cybersecurity analytics by converting model outputs into SIEM-ready threat events for Splunk/OpenSearch dashboards and SOAR-lite alerting.

## Model Details

- **Model type:** DistilBERT sequence classifier
- **Base model:** `distilbert-base-uncased`
- **Task:** Dreaddit binary stress classification
- **Labels:** `non_stress`, `stress`
- **Problem type:** single-label classification
- **Training method:** LoRA/PEFT fine-tuning, exported as a merged full Hugging Face model
- **Repository:** https://github.com/abdulmuksith3/textthreat-poc
- **Thesis role:** stress signal model used alongside Jigsaw toxicity labels to support digital well-being risk scoring and synthetic co-occurrence analytics

## Intended Use

This model is intended for the TextThreat proof-of-concept pipeline:

```text
social media comment
-> stress probability
-> TextThreat digital_wellbeing event fields
-> Splunk/OpenSearch dashboard analytics
-> optional SOAR-lite alerting
```

It can be used in research demos where stress-related language is one signal in a broader harm and digital well-being analytics workflow.

## Out-of-Scope Use

This model is not a clinical mental-health diagnostic model. It must not be used for diagnosis, emergency triage, autonomous intervention, or high-stakes decisions. Human review and appropriate support pathways are required for real-world safety workflows.

## Evaluation

The current uploaded artifact corresponds to the thesis proof-of-concept training run. Metrics are stored in the repository under `experiments/results/dreaddit_metrics.json`.

| Metric | Value |
|---|---:|
| F1 | 0.7479 |
| ROC-AUC | 0.8237 |
| PR-AUC | 0.8259 |
| Eval loss | 0.6318 |

## Limitations

- Dreaddit labels are useful for stress-language research, but do not establish clinical state.
- Short or ambiguous phrases may be poorly calibrated.
- The TextThreat demo uses a configurable alert threshold and a transparent safety lexical overlay for explicit high-risk terms.
- Model outputs should be interpreted as risk signals for dashboarding and analyst review, not final judgments.

## Example

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_id = "abdulmuksith/textthreat-distilbert-dreaddit"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)

text = "I feel stressed and overwhelmed"
inputs = tokenizer(text, return_tensors="pt", truncation=True)
inputs = {k: v for k, v in inputs.items() if k in model.forward.__code__.co_varnames}

with torch.no_grad():
    logits = model(**inputs).logits[0]

scores = torch.softmax(logits, dim=0)
print({model.config.id2label[i]: float(scores[i]) for i in range(len(scores))})
```

## Citation

If referencing this model, cite the thesis project:

```text
Rizvi, A. M. (2026). TextThreat: AI-Powered Detection of Digital Well-Being Risks with Cybersecurity Analytics. MSc thesis, University of Doha for Science and Technology.
```
