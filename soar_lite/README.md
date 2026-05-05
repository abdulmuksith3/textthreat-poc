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
`digital_wellbeing.risk_score > SOAR_LITE_THRESHOLD` are escalated to email when
SMTP is configured, or to the local CSV alert log when SMTP is absent.

The optional polling daemon provides thesis-parity operation for environments where
Splunk management/search API access is available:

```bash
python soar_lite/soar_lite.py --poll-splunk --interval 30
```

For a one-shot verification run:

```bash
python soar_lite/soar_lite.py --poll-splunk --once
```

The poller searches the configured Splunk index for recent TextThreat events, applies
threshold escalation and same-session toxicity/stress co-occurrence verification inside
a 30-minute window, then logs or emails structured alerts. For the hosted thesis demo,
the recommended free setup uses inline escalation from the Gradio submit action; the
poller is included for reproducible optional operation.

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
SOAR_LITE_SPLUNK_INDEX
SPLUNK_MANAGEMENT_URL
SPLUNK_SEARCH_HOST
SPLUNK_SEARCH_PORT
SPLUNK_SEARCH_SCHEME
SPLUNK_API_TOKEN
OPENSEARCH_URL
OPENSEARCH_USERNAME
OPENSEARCH_PASSWORD
```
