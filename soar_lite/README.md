# SOAR-lite

`soar_lite.py` provides the thesis SOAR-lite artifact for high-risk TextThreat events.

It reads schema-valid NDJSON events, deduplicates by `text_hash + @timestamp`, and sends SMTP email when SMTP variables are configured. If SMTP is not configured, alerts are logged to:

```text
experiments/results/soar_alerts_log.csv
```

Run a local demo:

```bash
python soar_lite/soar_lite.py --demo
```

In the Gradio Splunk demo, SOAR-lite runs after each submitted comment. Events with
`digital_wellbeing.risk_score >= SOAR_LITE_THRESHOLD` are escalated to email when
SMTP is configured, or to the local CSV alert log when SMTP is absent.

Environment variables:

```text
SMTP_HOST
SMTP_PORT
SMTP_USERNAME
SMTP_PASSWORD
ALERT_TO_EMAIL
ALERT_FROM_EMAIL
SOAR_LITE_ENABLED
SOAR_LITE_THRESHOLD
OPENSEARCH_URL
OPENSEARCH_USERNAME
OPENSEARCH_PASSWORD
```
