# Splunk Dashboard Notes

The hosted TextThreat demo uses Splunk Cloud as the primary SIEM dashboard.

## Setup

1. Create a Splunk Cloud free trial.
2. Create an index named `textthreat`.
3. Enable HTTP Event Collector.
4. Create a HEC token with access to `index=textthreat`.
5. In the hosted app, set:

```text
SPLUNK_HEC_URL=https://<your-stack>.splunkcloud.com:8088/services/collector/event
SPLUNK_HEC_TOKEN=<token>
SPLUNK_INDEX=textthreat
```

## Demo Panels

- Latest Submitted Events: table of newly submitted comments by hash.
- High-Risk Events: table filtered on `digital_wellbeing.risk_score > 0.8`.
- Harm Type Counts: bar chart over `digital_wellbeing.harm_types`.
- Risk Score Time Series: line chart over `_time`.
- Co-occurrence Candidates: session-window query for toxicity and stress.

## Demo Story

1. Open the Hugging Face Spaces Gradio app.
2. Submit a comment.
3. Confirm the app returns a `text_hash`, risk score, harm types, and Splunk HEC status.
4. Open the Splunk dashboard.
5. Refresh or wait for auto-refresh and show the new event in the latest-events panel.
6. Submit a high-risk sample and show the high-risk panel and alert search.
