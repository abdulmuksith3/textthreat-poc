"""Fast smoke test for TextThreat thesis artifacts."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from soar_lite.soar_lite import main as soar_main
from src.textthreat.cooccurrence import main as cooccurrence_main
from src.textthreat.dp_output import main as dp_main
from src.textthreat.evaluate import main as evaluate_main
from src.textthreat.export_events import SAMPLE_OUTPUT, sample_prediction_rows, write_events
from src.textthreat.fairness import main as fairness_main
from src.textthreat.latency import main as latency_main
from src.textthreat.schema import validate_event
from src.textthreat.utils import read_ndjson


def main() -> None:
    """Generate sample artifacts and validate the fast thesis workflow."""
    write_events(sample_prediction_rows(5), SAMPLE_OUTPUT)
    events = read_ndjson(SAMPLE_OUTPUT)
    assert len(events) == 5, "Expected five sample events."
    for event in events:
        validate_event(event)

    evaluate_main(["--sample"])
    latency_main(["--sample", "--iterations", "10"])
    dp_main(["--sample"])
    fairness_main(["--sample"])
    cooccurrence_main(["--sample", "--count", "20"])
    soar_main(["--demo", "--input", str(SAMPLE_OUTPUT)])

    expected = [
        SAMPLE_OUTPUT,
        Path("experiments/results/classification_metrics.json"),
        Path("experiments/results/latency_metrics.json"),
        Path("experiments/results/dp_results.json"),
        Path("experiments/results/fairness_results.json"),
        Path("experiments/results/cooccurrence_results.json"),
        Path("experiments/results/soar_alerts_log.csv"),
    ]
    missing = [str(path) for path in expected if not path.exists()]
    if missing:
        raise SystemExit(f"Smoke test did not create expected files: {missing}")
    print("TextThreat smoke test passed.")


if __name__ == "__main__":
    main()
