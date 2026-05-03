# OpenSearch Query Examples

OpenSearch remains compatible with the TextThreat schema, while the hosted demo uses Splunk Cloud.

## Risk Score Time Series

```json
GET textthreat-events-*/_search
{
  "size": 0,
  "aggs": {
    "over_time": {
      "date_histogram": {
        "field": "@timestamp",
        "fixed_interval": "30m"
      },
      "aggs": {
        "avg_risk": { "avg": { "field": "digital_wellbeing.risk_score" } },
        "p95_risk": {
          "percentiles": {
            "field": "digital_wellbeing.risk_score",
            "percents": [95]
          }
        }
      }
    }
  }
}
```

## High Severity Events

```json
GET textthreat-events-*/_search
{
  "query": {
    "range": {
      "digital_wellbeing.risk_score": {
        "gt": 0.8
      }
    }
  }
}
```

## Co-occurrence Candidate Query

```json
GET textthreat-events-*/_search
{
  "query": {
    "bool": {
      "must": [
        { "range": { "digital_wellbeing.risk_score": { "gt": 0.8 } } },
        { "range": { "@timestamp": { "gte": "now-30m" } } }
      ]
    }
  },
  "_source": [
    "@timestamp",
    "session_id",
    "text_hash",
    "digital_wellbeing.harm_types",
    "digital_wellbeing.risk_score"
  ]
}
```
