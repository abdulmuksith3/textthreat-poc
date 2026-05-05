# Splunk SPL Query Library

Use these queries with `index=textthreat` and JSON events sent by TextThreat HEC.

## Latest Submitted Events

```spl
index=textthreat event.module=textthreat
| spath path=digital_wellbeing.harm_types{} output=harm_types
| eval harm_types=coalesce(harm_types, 'digital_wellbeing.harm_types{}', 'digital_wellbeing.harm_types')
| sort - _time
| table _time text_hash harm_types digital_wellbeing.risk_score digital_wellbeing.confidence digital_wellbeing.model_version source_platform session_id
```

## High-Risk Events

```spl
index=textthreat event.module=textthreat digital_wellbeing.risk_score>0.8
| spath path=digital_wellbeing.harm_types{} output=harm_types
| eval harm_types=coalesce(harm_types, 'digital_wellbeing.harm_types{}', 'digital_wellbeing.harm_types')
| table _time text_hash harm_types digital_wellbeing.risk_score digital_wellbeing.confidence source_platform session_id
```

## Harm Type Counts

```spl
index=textthreat event.module=textthreat
| spath path=digital_wellbeing.harm_types{} output=harm_type
| eval harm_type=coalesce(harm_type, 'digital_wellbeing.harm_types{}', 'digital_wellbeing.harm_types')
| mvexpand harm_type
| where isnotnull(harm_type) AND harm_type!="none_above_threshold" AND harm_type!=""
| stats count by harm_type
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
| spath path=digital_wellbeing.harm_types{} output=harm_type
| eval harm_type=coalesce(harm_type, 'digital_wellbeing.harm_types{}', 'digital_wellbeing.harm_types')
| mvexpand harm_type
| eval is_toxicity=if(harm_type IN ("toxic","severe_toxic","obscene","threat","insult","identity_hate"),1,0)
| eval is_stress=if(harm_type="stress",1,0)
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
