# Splunk SPL Query Library

Use these queries with `index=textthreat` and JSON events sent by TextThreat HEC.

## Latest Submitted Events

```spl
index=textthreat event.module=textthreat
| sort - _time
| table _time text_hash digital_wellbeing.harm_types digital_wellbeing.risk_score digital_wellbeing.confidence digital_wellbeing.model_version source_platform session_id
```

## High-Risk Events

```spl
index=textthreat event.module=textthreat digital_wellbeing.risk_score>0.8
| table _time text_hash digital_wellbeing.harm_types digital_wellbeing.risk_score digital_wellbeing.confidence source_platform session_id
```

## Harm Type Counts

```spl
index=textthreat event.module=textthreat
| mvexpand digital_wellbeing.harm_types
| stats count by digital_wellbeing.harm_types
| sort - count
```

## Risk Score Time Series

```spl
index=textthreat event.module=textthreat
| timechart span=30m avg(digital_wellbeing.risk_score) as avg_risk perc95(digital_wellbeing.risk_score) as p95_risk
```

## Co-occurrence Candidates

```spl
index=textthreat event.module=textthreat digital_wellbeing.risk_score>0.8
| mvexpand digital_wellbeing.harm_types
| eval is_toxicity=if(digital_wellbeing.harm_types IN ("toxic","severe_toxic","obscene","threat","insult","identity_hate"),1,0)
| eval is_stress=if(digital_wellbeing.harm_types="stress",1,0)
| bin _time span=30m
| stats max(is_toxicity) as toxicity max(is_stress) as stress values(text_hash) as text_hashes by session_id _time
| where toxicity=1 AND stress=1
```

## Dashboard Alert Search

```spl
index=textthreat event.module=textthreat digital_wellbeing.risk_score>0.8
| stats count as high_risk_count latest(_time) as latest_event by session_id
| where high_risk_count > 0
```
