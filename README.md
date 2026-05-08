# TextThreat

**TextThreat - AI-Powered Detection of Digital Well-Being Risks with Cybersecurity Analytics**

TextThreat is an MSc thesis proof-of-concept that detects digital well-being risk signals from social-media-style text, converts model output into a SIEM-ready JSON event, sends the event to Splunk Cloud, and triggers a lightweight SOAR-style escalation path.

The hosted demo is intentionally simple and reproducible:

```text
Hugging Face Space / Gradio UI
-> submit comment
-> TextThreat model inference
-> ECS-style TextThreat JSON event
-> Splunk HEC ingestion
-> Splunk Cloud dashboard
-> inline SOAR-lite escalation
-> CSV alert log and optional Postmark email
```

## Live Links

| Component | Location |
|---|---|
| GitHub repository | https://github.com/abdulmuksith3/textthreat-poc |
| Hosted demo UI | https://huggingface.co/spaces/abdulmuksith/textthreat-demo |
| Jigsaw harm model | https://huggingface.co/abdulmuksith/textthreat-distilbert-jigsaw |
| Dreaddit stress model | https://huggingface.co/abdulmuksith/textthreat-distilbert-dreaddit |
| Splunk dashboard | Private Splunk Cloud tenant, index `textthreat` |
| SOAR-lite email provider | Postmark HTTPS API fallback, configured through Space secrets |

## Demo Screenshots

### Hosted Hugging Face Demo

This is the public Gradio app hosted on Hugging Face Spaces.

![TextThreat Hugging Face Space demo](docs/screenshots/huggingface-space-demo.png)

### Submitted High-Risk Comment

This shows a high-risk comment being scored, exported as a JSON event, sent to Splunk, and escalated by SOAR-lite.

![TextThreat demo result with Splunk and SOAR-lite status](docs/screenshots/huggingface-demo-result.png)

### Splunk Dashboard

The Splunk dashboard is private to the configured Splunk Cloud tenant. After importing [textthreat_dashboard.xml](siem/splunk/textthreat_dashboard.xml), it should show:

- `Total TextThreat Events`
- `High Risk Events`
- `Average Risk Score`
- `Risk Score Time Series`
- `Harm Type Counts`
- `Events by Source Platform`
- `Latest Submitted Events`
- `Co-occurrence Candidates`

The dashboard screenshot from the demo session corresponds to this imported dashboard and uses `index=textthreat`.

## What The Demo Proves

The live hosted demo proves the thesis pipeline end to end:

| Thesis / system component | Implemented by |
|---|---|
| Social media text input | Gradio UI in [demo/app.py](demo/app.py) |
| Jigsaw multi-label harm detection | `abdulmuksith/textthreat-distilbert-jigsaw` |
| Dreaddit stress detection | `abdulmuksith/textthreat-distilbert-dreaddit` |
| Event schema | [schema/textthreat_event_schema.json](schema/textthreat_event_schema.json), [schema.py](src/textthreat/schema.py) |
| Splunk ingestion | [splunk_hec.py](src/textthreat/splunk_hec.py) |
| Splunk dashboard | [siem/splunk/textthreat_dashboard.xml](siem/splunk/textthreat_dashboard.xml) |
| SOAR-lite escalation | [soar_lite.py](soar_lite/soar_lite.py) |
| Alert log | `experiments/results/soar_alerts_log.csv` |
| Optional email dispatch | Postmark HTTPS API fallback from [soar_lite.py](soar_lite/soar_lite.py) |
| Reproducible scripts | `scripts/`, `src/textthreat/`, `experiments/results/` |

## Repository Structure

```text
textthreat-poc/
├── app.py                              # Hugging Face Spaces entrypoint
├── README.md                           # This reproduction guide
├── requirements.txt                    # Python dependencies
├── config/
│   ├── settings.example.env            # Safe runtime config template
│   └── settings.env                    # Local secrets file, ignored by Git
├── data/
│   ├── exports/                        # NDJSON event exports
│   ├── jigsaw/                         # Put Jigsaw CSVs here locally
│   ├── dreaddit/                       # Put Dreaddit CSVs here locally
│   └── synthetic/                      # Synthetic co-occurrence sessions
├── demo/
│   └── app.py                          # Gradio UI and live demo workflow
├── docs/
│   └── screenshots/                    # README screenshots
├── experiments/
│   └── results/                        # JSON/CSV evidence artifacts
├── models/
│   ├── distilbert_jigsaw/              # Local lightweight metadata/tokenizer files
│   ├── distilbert_dreaddit/            # Local lightweight metadata/tokenizer files
│   └── svm_tfidf/                      # Classical SVM baseline artifacts
├── notebooks/
│   └── TextThreat_Colab_Training_All.ipynb
├── schema/
│   └── textthreat_event_schema.json    # JSON Schema Draft 7 event validation
├── scripts/
│   ├── smoke_test.py                   # Fast local verification
│   ├── generate_sample_events.py       # Writes sample NDJSON events
│   ├── setup_splunk_demo.py            # Sends sample events / optional setup helper
│   ├── check_model_artifacts.py        # Checks model folder completeness
│   └── upload_models_to_hf.py          # Optional model upload helper
├── siem/
│   └── splunk/                         # SPL queries and dashboard XML
├── soar_lite/
│   ├── soar_lite.py                    # Inline SOAR-lite + optional poller
│   ├── playbook.yml                    # Escalation recommendations
│   └── README.md
└── src/textthreat/
    ├── inference.py                    # Runtime model loading and scoring
    ├── schema.py                       # Event construction and validation
    ├── splunk_hec.py                   # Splunk HTTP Event Collector client
    ├── train_svm.py                    # SVM + TF-IDF baseline
    ├── train_distilbert.py             # DistilBERT Jigsaw/Dreaddit training
    ├── evaluate.py                     # Classification metrics
    ├── calibration.py                  # ECE calculations
    ├── latency.py                      # Latency profiling
    ├── dp_output.py                    # Output-level privacy perturbation
    ├── fairness.py                     # Fairlearn audit/demo mode
    └── cooccurrence.py                 # Synthetic co-occurrence evaluation
```

## Runtime Architecture

```text
User submits text in Gradio
        |
        v
TextThreatAnalyzer loads two Hugging Face models
        |
        |-- Jigsaw DistilBERT: toxic, severe_toxic, obscene, threat, insult, identity_hate
        |-- Dreaddit DistilBERT: stress
        |
        v
schema.build_event() creates validated TextThreat JSON
        |
        v
Splunk HEC receives event in index=textthreat
        |
        v
Splunk dashboard reads the event through SPL
        |
        v
SOAR-lite threshold rule checks risk_score > 0.8
        |
        |-- writes alert CSV
        |-- sends Postmark email when configured
```

## Event Schema Summary

Every live event follows the TextThreat JSON schema and includes:

```json
{
  "@timestamp": "2026-05-08T13:54:50.313824+00:00",
  "event": {
    "kind": "signal",
    "module": "textthreat",
    "category": ["digital_wellbeing", "threat_signal"]
  },
  "text_hash": "sha256 hash of raw text",
  "digital_wellbeing": {
    "harm_types": ["toxic", "obscene", "insult"],
    "risk_score": 0.982702,
    "confidence": 0.982702,
    "model_version": "distilbert-jigsaw-v1",
    "scores": {
      "toxic": 0.982702,
      "severe_toxic": 0.184931,
      "obscene": 0.811199,
      "threat": 0.097398,
      "insult": 0.887944,
      "identity_hate": 0.119561,
      "stress": 0.526596
    }
  },
  "source_platform": "demo_form",
  "session_id": "demo-session"
}
```

Raw comment text is not exported to Splunk. The event stores `text_hash` for analyst lookup and reproducibility without storing the original text.

## Quick Reproduction Path

Use this path when you only want to verify the repository locally without training models.

### 1. Clone The Repository

```bash
git clone https://github.com/abdulmuksith3/textthreat-poc.git
cd textthreat-poc
```

### 2. Create A Python Environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Run The Smoke Test

```bash
python scripts/smoke_test.py
```

Expected result:

```text
TextThreat smoke test passed.
```

Expected files:

```text
data/exports/sample_textthreat_events.ndjson
experiments/results/classification_metrics.json
experiments/results/latency_metrics.json
experiments/results/dp_results.json
experiments/results/fairness_results.json
experiments/results/cooccurrence_results.json
experiments/results/soar_alerts_log.csv
```

### 4. Run Individual Evidence Scripts

```bash
python scripts/generate_sample_events.py
python -m src.textthreat.export_events --sample
python -m src.textthreat.evaluate --sample
python -m src.textthreat.latency --sample
python -m src.textthreat.dp_output --sample
python -m src.textthreat.fairness --sample
python -m src.textthreat.cooccurrence --sample
python soar_lite/soar_lite.py --demo
```

These commands run quickly and do not require Kaggle datasets.

## Local Demo Setup

### 1. Create A Local Env File

Copy the example:

```powershell
Copy-Item config\settings.example.env config\settings.env
```

`config/settings.env` is ignored by Git. Put secrets there, never in `settings.example.env`.

### 2. Minimum Local Demo Variables

```env
TEXTTHREAT_TOXICITY_MODEL_ID=abdulmuksith/textthreat-distilbert-jigsaw
TEXTTHREAT_STRESS_MODEL_ID=abdulmuksith/textthreat-distilbert-dreaddit
TEXTTHREAT_EVENT_THRESHOLD=0.55

SPLUNK_HEC_URL=https://<your-stack>.splunkcloud.com:8088/services/collector/event
SPLUNK_HEC_TOKEN=<your-hec-token>
SPLUNK_INDEX=textthreat
SPLUNK_SOURCETYPE=_json
SPLUNK_SOURCE=textthreat-demo
SPLUNK_VERIFY_SSL=true
SPLUNK_HEC_CHANNEL=11111111-1111-4111-8111-111111111111

SOAR_LITE_ENABLED=1
SOAR_LITE_THRESHOLD=0.8
```

### 3. Optional Postmark Email Variables

Use these if you want SOAR-lite to send email alerts.

```env
POSTMARK_API_TOKEN=<postmark-server-token>
POSTMARK_MESSAGE_STREAM=outbound
ALERT_TO_EMAIL=<recipient-email>
ALERT_FROM_EMAIL=<verified-postmark-sender-email>
```

The app can still run without email. If email is not configured or fails, SOAR-lite still logs alerts and returns `email_sent=false` with `email_reason`.

### 4. Start The App Locally

```bash
python app.py
```

Open:

```text
http://127.0.0.1:7860
```

The first run can take 10-30 seconds because the two Hugging Face models are downloaded and warmed up.

## Hosted Hugging Face Space Setup

The current hosted Space is:

```text
https://huggingface.co/spaces/abdulmuksith/textthreat-demo
```

The Space runs `app.py`, which imports the Gradio app from [demo/app.py](demo/app.py).

### Required Space Variables

Set these as non-secret Space variables:

```text
TEXTTHREAT_TOXICITY_MODEL_ID=abdulmuksith/textthreat-distilbert-jigsaw
TEXTTHREAT_STRESS_MODEL_ID=abdulmuksith/textthreat-distilbert-dreaddit
TEXTTHREAT_EVENT_THRESHOLD=0.55
```

### Required Space Secrets

Set these as Space secrets:

```text
SPLUNK_HEC_URL
SPLUNK_HEC_TOKEN
SPLUNK_INDEX
SPLUNK_SOURCETYPE
SPLUNK_SOURCE
SPLUNK_VERIFY_SSL
SPLUNK_HEC_CHANNEL
SOAR_LITE_ENABLED
SOAR_LITE_THRESHOLD
```

### Optional Email Secrets

Set these if using Postmark email escalation:

```text
POSTMARK_API_TOKEN
POSTMARK_MESSAGE_STREAM
ALERT_TO_EMAIL
ALERT_FROM_EMAIL
```

SMTP variables are also supported, but many hosted containers block outbound SMTP. The recommended hosted path is Postmark HTTPS API.

### Deploy Or Update The Space

The simplest way is to upload the runtime files using `huggingface_hub`.

PowerShell:

```powershell
$env:HF_TOKEN="<your-hugging-face-write-token>"
@'
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path='.',
    repo_id='abdulmuksith/textthreat-demo',
    repo_type='space',
    allow_patterns=[
        'app.py',
        'README.md',
        'requirements.txt',
        'demo/**',
        'src/textthreat/**',
        'schema/textthreat_event_schema.json',
        'soar_lite/**',
        'siem/splunk/**',
        'config/settings.example.env',
        'docs/screenshots/**',
    ],
    commit_message='Deploy TextThreat demo app',
)
api.restart_space('abdulmuksith/textthreat-demo')
'@ | python -
```

Do not upload raw datasets, local `.env` files, `config/settings.env`, or model weights from your machine.

## Splunk Cloud Setup

### 1. Create The Index

In Splunk Cloud, create an index named:

```text
textthreat
```

### 2. Create An HEC Token

In Splunk Cloud:

```text
Settings -> Data Inputs -> HTTP Event Collector -> New Token
```

Recommended values:

```text
Source type: _json
Index: textthreat
```

Your HEC endpoint usually looks like:

```text
https://<your-stack>.splunkcloud.com:8088/services/collector/event
```

Some Splunk Cloud configurations require an HEC acknowledgement channel. This repo supports that using:

```text
SPLUNK_HEC_CHANNEL=11111111-1111-4111-8111-111111111111
```

### 3. Send A Test Event

With `config/settings.env` configured locally:

```bash
python scripts/setup_splunk_demo.py --events-only
```

Then search in Splunk:

```spl
index=textthreat event.module=textthreat
```

### 4. Import The Dashboard

Use the dashboard XML here:

```text
siem/splunk/textthreat_dashboard.xml
```

In Splunk Cloud:

```text
Dashboards -> Create New Dashboard -> Classic XML / Source -> paste XML
```

If your Splunk UI uses Dashboard Studio only, recreate the panels using [spl_queries.md](siem/splunk/spl_queries.md).

### 5. Useful SPL Searches

Latest events:

```spl
index=textthreat event.module=textthreat
| sort - _time
| table _time text_hash digital_wellbeing.risk_score digital_wellbeing.confidence source_platform session_id
```

High-risk events:

```spl
index=textthreat event.module=textthreat digital_wellbeing.risk_score>0.8
| table _time text_hash digital_wellbeing.harm_types{} digital_wellbeing.risk_score source_platform session_id
```

Harm type counts:

```spl
index=textthreat event.module=textthreat
| spath path=digital_wellbeing.harm_types{} output=harm_type
| mvexpand harm_type
| stats count by harm_type
| sort - count
```

Risk time series:

```spl
index=textthreat event.module=textthreat
| timechart span=5m avg(digital_wellbeing.risk_score) as avg_risk perc95(digital_wellbeing.risk_score) as p95_risk
```

## SOAR-lite Escalation

SOAR-lite is implemented in [soar_lite.py](soar_lite/soar_lite.py).

The hosted demo uses inline escalation:

```text
comment submitted
-> event built
-> event sent to Splunk
-> process_events([event]) runs immediately
-> high-risk events create alert rows
-> optional Postmark email is sent
```

Threshold rule:

```text
risk_score > SOAR_LITE_THRESHOLD
```

Default threshold:

```text
SOAR_LITE_THRESHOLD=0.8
```

Alert rows include:

```text
alert_timestamp
alert_type
text_hash
session_id
harm_types
risk_score
model_version
recommendation
email_sent
email_reason
```

### Email Behavior

The hosted demo uses Postmark HTTPS fallback because many hosted runtimes block SMTP.

Successful email example:

```json
{
  "email_sent": true,
  "email_reason": "postmark_api_status:200"
}
```

If email fails, the pipeline still completes and records the reason:

```json
{
  "email_sent": false,
  "email_reason": "postmark_api_http_error:422: Sender signature not found"
}
```

### Optional Polling Daemon

A Splunk polling mode is included for thesis-parity operation, but it is not required for the free hosted demo.

```bash
python soar_lite/soar_lite.py --poll-splunk --interval 30
```

One-shot mode:

```bash
python soar_lite/soar_lite.py --poll-splunk --once
```

This requires Splunk management/search API access, which is separate from HEC ingestion.

## Model Details

### Jigsaw Harm Classifier

```text
Model: abdulmuksith/textthreat-distilbert-jigsaw
Task: multi-label toxic comment classification
Base: DistilBERT
Labels: toxic, severe_toxic, obscene, threat, insult, identity_hate
```

Used for the main harm labels in `digital_wellbeing.harm_types`.

### Dreaddit Stress Classifier

```text
Model: abdulmuksith/textthreat-distilbert-dreaddit
Task: stress classification
Base: DistilBERT
Labels: non-stress / stress
```

Used to add the `stress` score and support stress/toxicity co-occurrence analysis.

### Safety Lexical Overlay

The demo also applies a transparent lexical overlay for explicit high-risk phrases such as direct threats or self-harm language. The raw model scores remain visible in the JSON output.

## Training Reproduction

Training is optional for running the hosted demo because trained models are already hosted on Hugging Face.

### Dataset Locations

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

### Train SVM Baseline

```bash
python -m src.textthreat.train_svm
```

Quick sample run:

```bash
python -m src.textthreat.train_svm --sample-size 2000 --calibration-cv 3
```

Outputs:

```text
models/svm_tfidf/
experiments/results/svm_metrics.json
```

### Train DistilBERT Jigsaw

```bash
python -m src.textthreat.train_distilbert --task jigsaw --epochs 1
```

Quick sample run:

```bash
python -m src.textthreat.train_distilbert --task jigsaw --sample-size 1000 --epochs 1
```

Outputs:

```text
models/distilbert_jigsaw/
experiments/results/distilbert_metrics.json
```

### Train DistilBERT Dreaddit

```bash
python -m src.textthreat.train_distilbert --task dreaddit --epochs 1
```

Outputs:

```text
models/distilbert_dreaddit/
experiments/results/dreaddit_metrics.json
```

### Colab Notebook

For GPU training, use:

```text
notebooks/TextThreat_Colab_Training_All.ipynb
```

Recommended Colab settings:

```text
Runtime -> Change runtime type -> GPU
Repository -> https://github.com/abdulmuksith3/textthreat-poc
QUICK_TEST=True first
QUICK_TEST=False for final evidence run
```

## Evaluation Artifacts

| Artifact | Command | Output |
|---|---|---|
| Classification metrics | `python -m src.textthreat.evaluate --sample` | `experiments/results/classification_metrics.json` |
| Latency | `python -m src.textthreat.latency --sample` | `experiments/results/latency_metrics.json` |
| Calibration / ECE | included in evaluation | `classification_metrics.json` |
| Output-level DP perturbation | `python -m src.textthreat.dp_output --sample` | `experiments/results/dp_results.json` |
| Fairness audit/demo | `python -m src.textthreat.fairness --sample` | `experiments/results/fairness_results.json` |
| Synthetic co-occurrence | `python -m src.textthreat.cooccurrence --sample` | `experiments/results/cooccurrence_results.json` |
| SOAR-lite alerts | `python soar_lite/soar_lite.py --demo` | `experiments/results/soar_alerts_log.csv` |

The DP script is an output-level privacy-preserving perturbation experiment, not full training-level DP-SGD.

The co-occurrence evaluation is synthetic and is marked as:

```text
evaluation_type = synthetic_session_windows
```

## Example Demo Comments

Use these to exercise the demo:

| Comment | Expected behavior |
|---|---|
| `I love this community, everyone here is helpful and kind.` | Low risk, no SOAR alert |
| `You are useless and disgusting, shut up.` | Toxic / insult likely, may or may not exceed SOAR threshold |
| `You idiot, go die.` | High risk, toxic/obscene/insult likely, SOAR alert likely |
| `I swear I will kill you.` | High risk, threat likely, SOAR alert likely |
| `I feel overwhelmed, exhausted, and I cannot handle this anymore.` | Stress score should rise, may not trigger SOAR unless risk exceeds threshold |

Use the same `session_id` across multiple related comments when demonstrating session-level correlation.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Gradio returns 500 on `/run/predict` | Space running stale build or endpoint mismatch | Restart Space and confirm `/gradio_api/info` exposes `/predict` |
| Splunk says skipped | HEC env vars missing | Set `SPLUNK_HEC_URL`, `SPLUNK_HEC_TOKEN`, `SPLUNK_INDEX` |
| Splunk HEC returns invalid channel | HEC indexer acknowledgement requires channel | Set `SPLUNK_HEC_CHANNEL` to a UUID-like value |
| Splunk dashboard has no harm counts | Array field needs `spath` parsing | Use the SPL in [spl_queries.md](siem/splunk/spl_queries.md) |
| `email_sent=false` with SMTP timeout | Hosted runtime blocks SMTP | Use Postmark HTTPS API variables |
| Postmark API returns 422 | Sender, stream, or token validation issue | Verify `ALERT_FROM_EMAIL`, server token, and `POSTMARK_MESSAGE_STREAM` |
| First request is slow | Models are downloading/warming | Wait for warmup logs; later requests are faster |
| HF Hub rate warning appears | No HF token configured | Optional: add `HF_TOKEN` as Space secret |

## Security Notes

Never commit these files or values:

```text
.env
.env.*
config/settings.env
config/*.local.env
SPLUNK_HEC_TOKEN
POSTMARK_API_TOKEN
SMTP_PASSWORD
HF_TOKEN
raw Kaggle datasets
large model weights
```

If a token is pasted into chat, logs, screenshots, or a public issue, rotate it.

## Thesis Artifact Mapping

| Thesis component | Repo artifact |
|---|---|
| A1 SVM baseline | [train_svm.py](src/textthreat/train_svm.py), `experiments/results/svm_metrics.json` |
| A1 DistilBERT | [train_distilbert.py](src/textthreat/train_distilbert.py), hosted Hugging Face models |
| Jigsaw multi-label classification | [constants.py](src/textthreat/constants.py), Jigsaw model |
| Dreaddit stress classification | [train_distilbert.py](src/textthreat/train_distilbert.py), Dreaddit model |
| A2 Schema | [textthreat_event_schema.json](schema/textthreat_event_schema.json), [schema.py](src/textthreat/schema.py) |
| NDJSON export | [export_events.py](src/textthreat/export_events.py), `data/exports/sample_textthreat_events.ndjson` |
| A3 Splunk SIEM | [splunk_hec.py](src/textthreat/splunk_hec.py), [siem/splunk](siem/splunk), [demo/app.py](demo/app.py) |
| A4 SOAR-lite | [soar_lite.py](soar_lite/soar_lite.py), [playbook.yml](soar_lite/playbook.yml) |
| RQ1 metrics | [evaluate.py](src/textthreat/evaluate.py), `classification_metrics.json` |
| RQ2 co-occurrence | [cooccurrence.py](src/textthreat/cooccurrence.py), `cooccurrence_results.json` |
| RQ3 latency | [latency.py](src/textthreat/latency.py), `latency_metrics.json` |
| RQ4 DP/fairness | [dp_output.py](src/textthreat/dp_output.py), [fairness.py](src/textthreat/fairness.py) |
| Reproducibility | [smoke_test.py](scripts/smoke_test.py), [run_all_local.py](scripts/run_all_local.py), dataset READMEs |

## Proof-Of-Concept Scope

TextThreat is a thesis proof-of-concept. It demonstrates feasibility of AI-assisted digital well-being risk detection with cybersecurity analytics patterns. It is not an enterprise moderation platform, and human review remains part of the intended workflow.

The hosted demo deliberately uses inline SOAR-lite escalation because it is simple to reproduce on free hosting. The optional polling daemon is included for environments that can provide always-on workers and Splunk search API access.
