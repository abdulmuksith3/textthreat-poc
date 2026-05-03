# Experiments

This directory stores reproducible thesis evidence artifacts.

Expected result files:

```text
experiments/results/classification_metrics.json
experiments/results/svm_metrics.json
experiments/results/distilbert_metrics.json
experiments/results/dreaddit_metrics.json
experiments/results/latency_metrics.json
experiments/results/dp_results.json
experiments/results/fairness_results.json
experiments/results/cooccurrence_results.json
experiments/results/soar_alerts_log.csv
```

Run fast demo outputs:

```bash
python scripts/smoke_test.py
```

Heavy training outputs are generated locally or in Colab and are not required for the hosted Splunk demo.
