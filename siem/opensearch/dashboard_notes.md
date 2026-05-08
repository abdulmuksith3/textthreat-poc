# OpenSearch Dashboard Notes

OpenSearch is included as a compatible SIEM target for thesis portability. The live hosted demo should use Splunk Cloud.

Suggested panels:

- Risk score time series over `digital_wellbeing.risk_score`.
- High-risk event table where risk score exceeds `0.8`.
- Harm type counts from `digital_wellbeing.harm_types`.
- Co-occurrence candidates grouped by `session_id` and 30-minute windows.
