# TextThreat

**TextThreat — AI-Powered Detection of Digital Well-Being Risks with Cybersecurity Analytics**

This repository contains the MSc thesis proof-of-concept for detecting digital well-being risk signals from social-media text and exporting them as SIEM-ready threat events.

The live demo path is **Splunk-first**:

```text
user submits comment
-> TextThreat inference
-> schema-valid JSON event
-> Splunk HTTP Event Collector
-> Splunk dashboard
-> optional SOAR-lite alert
```

OpenSearch compatibility is also included for portability, but the hosted dashboard demo uses Splunk Cloud.

## Current Implementation

- Jigsaw Toxic Comment multi-label harm labels:
  `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`
- SVM + TF-IDF baseline training script.
- DistilBERT training script for Jigsaw and Dreaddit stress classification.
- Optional LoRA/PEFT and MLflow logging for thesis-parity experiments.
- TextThreat JSON Schema Draft 7 threat-event format.
- NDJSON export for SIEM ingestion.
- Splunk HEC client and Gradio hosted demo app.
- Splunk SPL query library and dashboard notes.
- OpenSearch index template and query examples.
- SOAR-lite alert logging/email.
- Latency, calibration/ECE, output-level privacy perturbation, Fairlearn audit, and synthetic co-occurrence scripts.
- Fast smoke test that runs without raw Kaggle data.

## Install

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

macOS/Linux activation:

```bash
source .venv/bin/activate
```

## Dataset Setup

Raw datasets are not committed.

Place Jigsaw files here:

```text
data/jigsaw/train.csv
data/jigsaw/test.csv
data/jigsaw/test_labels.csv
data/jigsaw/sample_submission.csv
```

Place Dreaddit files here:

```text
data/dreaddit/dreaddit-train.csv
data/dreaddit/dreaddit-test.csv
```

The smoke-test and demo scripts can run in sample mode without these files.

## Fast Local Verification

```bash
python scripts/smoke_test.py
python scripts/generate_sample_events.py
python -m src.textthreat.export_events --sample
python -m src.textthreat.latency --sample
python -m src.textthreat.dp_output --sample
python -m src.textthreat.fairness --sample
python -m src.textthreat.cooccurrence --sample
```

Expected outputs:

```text
data/exports/sample_textthreat_events.ndjson
experiments/results/classification_metrics.json
experiments/results/latency_metrics.json
experiments/results/dp_results.json
experiments/results/fairness_results.json
experiments/results/cooccurrence_results.json
experiments/results/soar_alerts_log.csv
```

## Training Workflow

Training runs locally or in Colab. The hosted demo should load trained artifacts and perform inference only.

SVM baseline:

```bash
python -m src.textthreat.train_svm
```

DistilBERT Jigsaw classifier:

```bash
python -m src.textthreat.train_distilbert --task jigsaw --epochs 1
```

Dreaddit stress classifier:

```bash
python -m src.textthreat.train_distilbert --task dreaddit --epochs 1
```

For a quick training rehearsal:

```bash
python -m src.textthreat.train_svm --sample-size 2000 --calibration-cv 3
python -m src.textthreat.train_distilbert --task jigsaw --sample-size 1000 --epochs 1
```

Training outputs are written under:

```text
models/
experiments/results/
mlruns/
```

Large weights, raw datasets, and MLflow run folders are ignored by Git.

When LoRA is enabled, the training script merges the adapters into full Hugging Face model folders before saving. A valid final DistilBERT artifact contains `config.json`, `model.safetensors`, and tokenizer files. If you have adapter-only folders from an older Colab run, repair them before zipping:

```bash
python scripts/merge_lora_artifacts.py --root .
python scripts/check_model_artifacts.py --root .
```

Upload trained models to Hugging Face Hub:

```bash
set HF_TOKEN=<your-hugging-face-write-token>
set HF_NAMESPACE=<your-hf-username-or-org>
python scripts/upload_models_to_hf.py
```

The upload script publishes:

```text
<namespace>/textthreat-distilbert-jigsaw
<namespace>/textthreat-distilbert-dreaddit
```

Then configure the demo with those model IDs:

```text
TEXTTHREAT_TOXICITY_MODEL_ID=<namespace>/textthreat-distilbert-jigsaw
TEXTTHREAT_STRESS_MODEL_ID=<namespace>/textthreat-distilbert-dreaddit
```

## Hosted Splunk Demo

Recommended free-hosted setup:

- App: Hugging Face Spaces using `app.py` / `demo/app.py`.
- SIEM: Splunk Cloud free trial.
- Ingestion: Splunk HTTP Event Collector.

Configure secrets in the host:

```text
TEXTTHREAT_TOXICITY_MODEL_ID=<local path or Hugging Face model id>
TEXTTHREAT_STRESS_MODEL_ID=<optional stress model id>
TEXTTHREAT_EVENT_THRESHOLD=0.55
SPLUNK_HEC_URL=https://<your-stack>.splunkcloud.com:8088/services/collector/event
SPLUNK_HEC_TOKEN=<token>
SPLUNK_INDEX=textthreat
```

`TEXTTHREAT_EVENT_THRESHOLD` controls when a model score becomes an active `digital_wellbeing.harm_types` label in the live demo. The default demo threshold is `0.55` to avoid treating borderline, poorly calibrated scores as active harms. The raw per-label scores are still included in the exported event JSON.

The hosted proof-of-concept also applies a transparent safety lexical overlay for explicit high-risk phrases such as direct threats, self-harm language, and profanity. This keeps the live SIEM demo responsive to obvious safety cases while the raw model scores remain visible in the exported JSON.

Run locally:

```bash
python app.py
```

Demo workflow:

```text
1. Open the Gradio app.
2. Submit a comment.
3. The app scores harm/stress risk.
4. The app hashes the text and builds a TextThreat event.
5. The event is validated against schema/textthreat_event_schema.json.
6. The event is sent to Splunk HEC if credentials are configured.
7. Splunk dashboard panels update from index=textthreat.
8. SOAR-lite can log or email alerts for risk_score > 0.8.
```

SOAR-lite escalation is enabled in the demo by default. Set `SOAR_LITE_THRESHOLD`
to control the escalation threshold and `SOAR_LITE_ENABLED=0` to disable it for
dashboard-only rehearsals. The hosted proof-of-concept uses inline SOAR-lite
escalation from the submit action, which keeps the free demo responsive without a
separate always-on worker.

If no trained model is configured, the app uses a clearly marked demo fallback scorer. For the final thesis demo, configure trained model IDs.

By default the local app does not auto-load local model folders, so the submit button responds quickly during setup. To force local model loading from `models/distilbert_jigsaw/`, set:

```powershell
$env:TEXTTHREAT_AUTO_LOAD_LOCAL_MODELS="1"
```

For a hosted demo, prefer an explicit model ID:

```powershell
$env:TEXTTHREAT_TOXICITY_MODEL_ID="your-hf-username/textthreat-distilbert-jigsaw"
```

## Splunk Dashboard

See:

```text
siem/splunk/spl_queries.md
siem/splunk/dashboard_notes.md
```

High-risk SPL:

```spl
index=textthreat event.module=textthreat digital_wellbeing.risk_score>0.8
| table _time text_hash digital_wellbeing.harm_types digital_wellbeing.risk_score digital_wellbeing.confidence source_platform session_id
```

Harm type counts:

```spl
index=textthreat event.module=textthreat
| mvexpand digital_wellbeing.harm_types
| stats count by digital_wellbeing.harm_types
```

Risk score time series:

```spl
index=textthreat event.module=textthreat
| timechart span=30m avg(digital_wellbeing.risk_score) as avg_risk perc95(digital_wellbeing.risk_score) as p95_risk
```

## SOAR-lite

Run the local SOAR-lite demo:

```bash
python soar_lite/soar_lite.py --demo
```

Without SMTP settings, alerts are written to:

```text
experiments/results/soar_alerts_log.csv
```

With SMTP settings, high-risk alerts can be emailed using the variables in `config/settings.example.env`.

In the live Gradio demo, SOAR-lite runs immediately after the event is built and
sent to Splunk. If the event risk score is above `SOAR_LITE_THRESHOLD`, the demo
returns the escalation result and writes an alert row to:

```text
experiments/results/soar_alerts_log.csv
```

An optional Splunk polling daemon is also included for thesis-parity operation when
Splunk search API access is available:

```bash
python soar_lite/soar_lite.py --poll-splunk --interval 30
```

The poller queries the configured Splunk index, applies threshold escalation and
30-minute same-session toxicity/stress co-occurrence checks, and dispatches the same
CSV/email alert format. For the free hosted demo, use the inline escalation path.

## Thesis Artifact Mapping

| Thesis component | Repo artifact |
|---|---|
| A1 SVM baseline | `src/textthreat/train_svm.py`, `experiments/results/svm_metrics.json` |
| A1 DistilBERT | `src/textthreat/train_distilbert.py`, `models/distilbert_jigsaw/` |
| Jigsaw multi-label classification | `src/textthreat/constants.py`, `src/textthreat/train_distilbert.py` |
| Dreaddit stress classification | `src/textthreat/train_distilbert.py --task dreaddit`, `experiments/results/dreaddit_metrics.json` |
| A2 Schema | `schema/textthreat_event_schema.json`, `src/textthreat/schema.py` |
| NDJSON export | `src/textthreat/export_events.py`, `data/exports/sample_textthreat_events.ndjson` |
| A3 Splunk SIEM | `src/textthreat/splunk_hec.py`, `siem/splunk/`, `demo/app.py` |
| OpenSearch compatibility | `siem/opensearch/` |
| A4 SOAR-lite | `soar_lite/soar_lite.py`, `soar_lite/playbook.yml` |
| RQ1 metrics | `src/textthreat/evaluate.py`, `experiments/results/classification_metrics.json` |
| RQ2 co-occurrence | `src/textthreat/cooccurrence.py`, `experiments/results/cooccurrence_results.json` |
| RQ3 latency | `src/textthreat/latency.py`, `experiments/results/latency_metrics.json` |
| RQ4 DP/fairness | `src/textthreat/dp_output.py`, `src/textthreat/fairness.py` |
| Reproducibility | `scripts/smoke_test.py`, `scripts/run_all_local.py`, dataset READMEs |

## Privacy And Scope Notes

- Exported SIEM events use `text_hash`; raw comment text is not stored in sample exports.
- The DP experiment is an **output-level privacy-preserving perturbation experiment**, not full formal training-level DP-SGD.
- Synthetic co-occurrence evaluation is marked as `synthetic_session_windows`.
- This is a thesis proof-of-concept, not an enterprise production deployment.
