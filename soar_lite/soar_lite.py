"""SOAR-lite alert dispatcher for TextThreat events."""

from __future__ import annotations

import argparse
import csv
import os
import smtplib
import sys
from email.message import EmailMessage
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.textthreat.constants import HIGH_RISK_THRESHOLD
from src.textthreat.export_events import SAMPLE_OUTPUT, sample_prediction_rows, write_events
from src.textthreat.utils import RESULTS_DIR, ensure_dir, now_utc_iso, read_ndjson


ALERT_LOG_PATH = RESULTS_DIR / "soar_alerts_log.csv"
PLAYBOOK_PATH = Path(__file__).resolve().parent / "playbook.yml"


def load_playbook(path: Path = PLAYBOOK_PATH) -> dict[str, Any]:
    """Load SOAR-lite playbook recommendations."""
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def alert_key(event: dict[str, Any]) -> str:
    """Build a deduplication key from text hash and timestamp."""
    return f"{event.get('text_hash')}|{event.get('@timestamp')}"


def is_high_risk(event: dict[str, Any], threshold: float = HIGH_RISK_THRESHOLD) -> bool:
    """Return true when an event meets high-risk escalation criteria."""
    return float(event.get("digital_wellbeing", {}).get("risk_score", 0.0)) >= threshold


def recommendation_for(event: dict[str, Any], playbook: dict[str, Any]) -> str:
    """Select a short recommendation from the playbook."""
    score = float(event.get("digital_wellbeing", {}).get("risk_score", 0.0))
    if score >= 0.9:
        key = "threshold_critical"
    else:
        key = "threshold_high"
    return str(playbook.get("recommendations", {}).get(key, "Review in Splunk dashboard and escalate if needed."))


def smtp_configured() -> bool:
    """Check whether SMTP environment variables are available."""
    required = ["SMTP_HOST", "SMTP_PORT", "ALERT_TO_EMAIL", "ALERT_FROM_EMAIL"]
    return all(os.getenv(name) for name in required)


def send_email_alert(event: dict[str, Any], recommendation: str) -> dict[str, Any]:
    """Send an SMTP email alert when SMTP settings exist."""
    if not smtp_configured():
        return {"sent": False, "reason": "SMTP settings are not configured."}
    message = EmailMessage()
    message["Subject"] = "TextThreat high-risk alert"
    message["From"] = os.environ["ALERT_FROM_EMAIL"]
    message["To"] = os.environ["ALERT_TO_EMAIL"]
    body = "\n".join(
        [
            f"Alert timestamp: {now_utc_iso()}",
            "Alert type: threshold",
            f"Harm types: {event['digital_wellbeing'].get('harm_types', [])}",
            f"Risk score: {event['digital_wellbeing'].get('risk_score')}",
            f"Text hash: {event.get('text_hash')}",
            f"Session ID: {event.get('session_id', '')}",
            f"Model version: {event['digital_wellbeing'].get('model_version')}",
            f"Recommendation: {recommendation}",
        ]
    )
    message.set_content(body)

    host = os.environ["SMTP_HOST"]
    port = int(os.environ.get("SMTP_PORT", "587"))
    username = os.getenv("SMTP_USERNAME")
    password = os.getenv("SMTP_PASSWORD")
    with smtplib.SMTP(host, port, timeout=20) as client:
        client.starttls()
        if username and password:
            client.login(username, password)
        client.send_message(message)
    return {"sent": True}


def append_alert_log(rows: list[dict[str, Any]], path: Path = ALERT_LOG_PATH) -> Path:
    """Append alert rows to the local CSV log."""
    ensure_dir(path.parent)
    fieldnames = [
        "alert_timestamp",
        "alert_type",
        "text_hash",
        "session_id",
        "harm_types",
        "risk_score",
        "model_version",
        "recommendation",
        "email_sent",
    ]
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def process_events(events: list[dict[str, Any]], threshold: float = HIGH_RISK_THRESHOLD) -> list[dict[str, Any]]:
    """Process high-risk events and write/email SOAR-lite alerts."""
    seen: set[str] = set()
    playbook = load_playbook()
    alert_rows: list[dict[str, Any]] = []
    for event in events:
        key = alert_key(event)
        if key in seen or not is_high_risk(event, threshold):
            continue
        seen.add(key)
        recommendation = recommendation_for(event, playbook)
        email_result = send_email_alert(event, recommendation)
        alert_rows.append(
            {
                "alert_timestamp": now_utc_iso(),
                "alert_type": "threshold",
                "text_hash": event.get("text_hash"),
                "session_id": event.get("session_id", ""),
                "harm_types": "|".join(event.get("digital_wellbeing", {}).get("harm_types", [])),
                "risk_score": event.get("digital_wellbeing", {}).get("risk_score"),
                "model_version": event.get("digital_wellbeing", {}).get("model_version"),
                "recommendation": recommendation,
                "email_sent": bool(email_result.get("sent")),
            }
        )
    if alert_rows:
        append_alert_log(alert_rows)
    return alert_rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run TextThreat SOAR-lite alert dispatcher.")
    parser.add_argument("--input", type=Path, default=SAMPLE_OUTPUT, help="NDJSON event file.")
    parser.add_argument("--demo", action="store_true", help="Generate sample events first.")
    parser.add_argument("--threshold", type=float, default=HIGH_RISK_THRESHOLD)
    return parser


def main(argv: list[str] | None = None) -> list[dict[str, Any]]:
    args = build_arg_parser().parse_args(argv)
    if args.demo:
        write_events(sample_prediction_rows(5), args.input)
        if ALERT_LOG_PATH.exists():
            ALERT_LOG_PATH.unlink()
    events = read_ndjson(args.input)
    alerts = process_events(events, threshold=args.threshold)
    if alerts:
        print(f"Wrote {len(alerts)} SOAR-lite alert(s) to {ALERT_LOG_PATH}")
    else:
        print("No SOAR-lite alerts generated.")
    return alerts


if __name__ == "__main__":
    main()
