"""SOAR-lite alert dispatcher for TextThreat events."""

from __future__ import annotations

import argparse
import csv
import json
import os
import smtplib
import sys
import time
import urllib.error
import urllib.request
from email.message import EmailMessage
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / "config" / "settings.env")
load_dotenv(ROOT / ".env")

from src.textthreat.constants import HIGH_RISK_THRESHOLD
from src.textthreat.export_events import SAMPLE_OUTPUT, sample_prediction_rows, write_events
from src.textthreat.utils import RESULTS_DIR, ensure_dir, now_utc_iso, read_ndjson


ALERT_LOG_PATH = RESULTS_DIR / "soar_alerts_log.csv"
PLAYBOOK_PATH = Path(__file__).resolve().parent / "playbook.yml"
TOXICITY_TYPES = {"toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"}
seen_hashes: set[str] = set()


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
    return float(event.get("digital_wellbeing", {}).get("risk_score", 0.0)) > threshold


def recommendation_for(event: dict[str, Any], playbook: dict[str, Any], alert_type: str = "threshold") -> str:
    """Select a short recommendation from the playbook."""
    if alert_type == "co_occurrence":
        return str(
            playbook.get("recommendations", {}).get(
                "co_occurrence", "Escalate to platform moderator for cross-signal review."
            )
        )
    score = float(event.get("digital_wellbeing", {}).get("risk_score", 0.0))
    if score > 0.9:
        key = "threshold_critical"
    else:
        key = "threshold_high"
    return str(playbook.get("recommendations", {}).get(key, "Review in Splunk dashboard and escalate if needed."))


def smtp_configured() -> bool:
    """Check whether SMTP environment variables are available."""
    return all([os.getenv("SMTP_HOST"), os.getenv("SMTP_PORT"), alert_to_email(), alert_from_email()])


def smtp_username() -> str | None:
    """Return SMTP username using either supported environment naming style."""
    return os.getenv("SMTP_USERNAME") or os.getenv("SMTP_USER")


def smtp_password() -> str | None:
    """Return SMTP password using either supported environment naming style."""
    return os.getenv("SMTP_PASSWORD") or os.getenv("SMTP_PASS")


def alert_from_email() -> str | None:
    """Return alert sender address, defaulting to the SMTP user for local demos."""
    return os.getenv("ALERT_FROM_EMAIL") or smtp_username()


def alert_to_email() -> str | None:
    """Return alert recipient address, defaulting to the SMTP user for local demos."""
    return os.getenv("ALERT_TO_EMAIL") or alert_from_email()


def send_email_alert(event: dict[str, Any], recommendation: str, alert_type: str = "threshold") -> dict[str, Any]:
    """Send an SMTP email alert when SMTP settings exist."""
    if not smtp_configured():
        api_result = send_postmark_api_alert(event, recommendation, alert_type)
        if api_result.get("sent"):
            return api_result
        return {"sent": False, "reason": "SMTP settings are not configured."}
    message = EmailMessage()
    message["Subject"] = f"TextThreat {alert_type} alert"
    message["From"] = alert_from_email() or ""
    message["To"] = alert_to_email() or ""
    body = "\n".join(
        [
            f"Alert timestamp: {now_utc_iso()}",
            f"Alert type: {alert_type}",
            f"Harm types: {event['digital_wellbeing'].get('harm_types', [])}",
            f"Risk score: {event['digital_wellbeing'].get('risk_score')}",
            f"Text hash: {event.get('text_hash')}",
            f"Session ID: {event.get('session_id', '')}",
            f"Model version: {event['digital_wellbeing'].get('model_version')}",
            f"Recommendation: {recommendation}",
        ]
    )
    message.set_content(body)
    postmark_stream = os.getenv("POSTMARK_MESSAGE_STREAM")
    if postmark_stream:
        message["X-PM-Message-Stream"] = postmark_stream

    host = os.environ["SMTP_HOST"]
    port = int(os.environ.get("SMTP_PORT", "587"))
    username = smtp_username()
    password = smtp_password()
    try:
        with smtplib.SMTP(host, port, timeout=20) as client:
            client.starttls()
            if username and password:
                client.login(username, password)
            client.send_message(message)
        return {"sent": True}
    except (OSError, TimeoutError, smtplib.SMTPException) as exc:
        # Hosted containers may block outbound SMTP; try Postmark HTTPS fallback.
        api_result = send_postmark_api_alert(event, recommendation, alert_type)
        if api_result.get("sent"):
            return api_result
        return {"sent": False, "reason": f"smtp_error: {exc}; {api_result.get('reason', 'postmark_api_unavailable')}"}


def postmark_api_token() -> str | None:
    """Return Postmark server token for HTTPS API fallback."""
    return os.getenv("POSTMARK_API_TOKEN") or smtp_password()


def send_postmark_api_alert(event: dict[str, Any], recommendation: str, alert_type: str) -> dict[str, Any]:
    """Send alert through Postmark HTTPS API when configured."""
    token = postmark_api_token()
    sender = alert_from_email()
    recipient = alert_to_email()
    if not token or not sender or not recipient:
        return {"sent": False, "reason": "postmark_api_not_configured"}

    body = "\n".join(
        [
            f"Alert timestamp: {now_utc_iso()}",
            f"Alert type: {alert_type}",
            f"Harm types: {event['digital_wellbeing'].get('harm_types', [])}",
            f"Risk score: {event['digital_wellbeing'].get('risk_score')}",
            f"Text hash: {event.get('text_hash')}",
            f"Session ID: {event.get('session_id', '')}",
            f"Model version: {event['digital_wellbeing'].get('model_version')}",
            f"Recommendation: {recommendation}",
        ]
    )
    payload = {
        "From": sender,
        "To": recipient,
        "Subject": f"TextThreat {alert_type} alert",
        "TextBody": body,
    }
    stream = os.getenv("POSTMARK_MESSAGE_STREAM")
    if stream:
        payload["MessageStream"] = stream

    request = urllib.request.Request(
        "https://api.postmarkapp.com/email",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "X-Postmark-Server-Token": token,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            response_body = response.read().decode("utf-8", errors="replace")
            return {"sent": 200 <= response.status < 300, "reason": f"postmark_api_status:{response.status}", "response": response_body}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return {"sent": False, "reason": f"postmark_api_http_error:{exc.code}", "response": body}
    except urllib.error.URLError as exc:
        return {"sent": False, "reason": f"postmark_api_connection_error:{exc.reason}"}


def alert_row(
    event: dict[str, Any],
    alert_type: str,
    recommendation: str,
    email_sent: bool,
    email_reason: str = "",
) -> dict[str, Any]:
    """Build a CSV alert row."""
    return {
        "alert_timestamp": now_utc_iso(),
        "alert_type": alert_type,
        "text_hash": event.get("text_hash"),
        "session_id": event.get("session_id", ""),
        "harm_types": "|".join(event.get("digital_wellbeing", {}).get("harm_types", [])),
        "risk_score": event.get("digital_wellbeing", {}).get("risk_score"),
        "model_version": event.get("digital_wellbeing", {}).get("model_version"),
        "recommendation": recommendation,
        "email_sent": email_sent,
        "email_reason": email_reason,
    }


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
        "email_reason",
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
    playbook = load_playbook()
    alert_rows: list[dict[str, Any]] = []
    for event in events:
        key = alert_key(event)
        if key in seen_hashes or not is_high_risk(event, threshold):
            continue
        seen_hashes.add(key)
        recommendation = recommendation_for(event, playbook, "threshold")
        email_result = send_email_alert(event, recommendation, "threshold")
        alert_rows.append(
            alert_row(
                event,
                "threshold",
                recommendation,
                bool(email_result.get("sent")),
                str(email_result.get("reason", "")),
            )
        )
    alert_rows.extend(process_cooccurrence_alerts(events, playbook))
    if alert_rows:
        append_alert_log(alert_rows)
    return alert_rows


def parse_timestamp(value: str | None) -> float | None:
    """Parse an ISO timestamp to Unix seconds."""
    if not value:
        return None
    from datetime import datetime

    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def has_toxicity(event: dict[str, Any]) -> bool:
    """Return true when the event contains a toxicity harm type."""
    harms = set(event.get("digital_wellbeing", {}).get("harm_types", []))
    return bool(harms & TOXICITY_TYPES)


def has_stress(event: dict[str, Any]) -> bool:
    """Return true when the event contains the stress harm type."""
    return "stress" in set(event.get("digital_wellbeing", {}).get("harm_types", []))


def cooccurrence_key(session_id: str, first: dict[str, Any], second: dict[str, Any]) -> str:
    """Build a stable deduplication key for co-occurrence alerts."""
    hashes = sorted([str(first.get("text_hash")), str(second.get("text_hash"))])
    return f"co_occurrence|{session_id}|{'|'.join(hashes)}"


def build_cooccurrence_event(session_id: str, first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    """Build an aggregate event used for co-occurrence alert dispatch."""
    first_dw = first.get("digital_wellbeing", {})
    second_dw = second.get("digital_wellbeing", {})
    harms = sorted(set(first_dw.get("harm_types", [])) | set(second_dw.get("harm_types", [])))
    risk = max(float(first_dw.get("risk_score", 0.0)), float(second_dw.get("risk_score", 0.0)))
    return {
        "@timestamp": max(str(first.get("@timestamp", "")), str(second.get("@timestamp", ""))),
        "text_hash": f"{first.get('text_hash')}+{second.get('text_hash')}",
        "session_id": session_id,
        "digital_wellbeing": {
            "harm_types": harms,
            "risk_score": round(risk, 6),
            "model_version": first_dw.get("model_version") or second_dw.get("model_version"),
        },
    }


def process_cooccurrence_alerts(
    events: list[dict[str, Any]],
    playbook: dict[str, Any],
    window_seconds: int = 30 * 60,
) -> list[dict[str, Any]]:
    """Dispatch co-occurrence alerts for toxicity and stress in the same session window."""
    alert_rows: list[dict[str, Any]] = []
    by_session: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        session_id = event.get("session_id")
        if session_id:
            by_session.setdefault(str(session_id), []).append(event)

    for session_id, session_events in by_session.items():
        toxicity_events = [event for event in session_events if has_toxicity(event)]
        stress_events = [event for event in session_events if has_stress(event)]
        for toxicity_event in toxicity_events:
            toxicity_ts = parse_timestamp(toxicity_event.get("@timestamp"))
            if toxicity_ts is None:
                continue
            for stress_event in stress_events:
                stress_ts = parse_timestamp(stress_event.get("@timestamp"))
                if stress_ts is None or abs(toxicity_ts - stress_ts) > window_seconds:
                    continue
                key = cooccurrence_key(session_id, toxicity_event, stress_event)
                if key in seen_hashes:
                    continue
                seen_hashes.add(key)
                aggregate = build_cooccurrence_event(session_id, toxicity_event, stress_event)
                recommendation = recommendation_for(aggregate, playbook, "co_occurrence")
                email_result = send_email_alert(aggregate, recommendation, "co_occurrence")
                alert_rows.append(
                    alert_row(
                        aggregate,
                        "co_occurrence",
                        recommendation,
                        bool(email_result.get("sent")),
                        str(email_result.get("reason", "")),
                    )
                )
                break
    return alert_rows


def splunk_config_from_env() -> dict[str, Any]:
    """Read Splunk polling configuration from environment variables."""
    management_url = (os.getenv("SPLUNK_MANAGEMENT_URL") or os.getenv("SPLUNK_MGMT_URL") or "").rstrip("/")
    host = os.getenv("SPLUNK_SEARCH_HOST")
    port = int(os.getenv("SPLUNK_SEARCH_PORT", "8089"))
    scheme = os.getenv("SPLUNK_SEARCH_SCHEME", "https")
    if management_url:
        scheme, remainder = management_url.split("://", 1) if "://" in management_url else ("https", management_url)
        host_port = remainder.rstrip("/").split("/", 1)[0]
        if ":" in host_port:
            host, raw_port = host_port.rsplit(":", 1)
            port = int(raw_port)
        else:
            host = host_port
    if not host:
        raise SystemExit("Set SPLUNK_MANAGEMENT_URL or SPLUNK_SEARCH_HOST to enable SOAR-lite Splunk polling.")
    return {
        "host": host,
        "port": port,
        "scheme": scheme,
        "username": os.getenv("SPLUNK_USERNAME"),
        "password": os.getenv("SPLUNK_PASSWORD"),
        "token": os.getenv("SPLUNK_API_TOKEN") or os.getenv("SPLUNK_ACCESS_TOKEN"),
        "verify": os.getenv("SPLUNK_VERIFY_SSL", "true").lower() not in {"0", "false", "no"},
        "index": os.getenv("SOAR_LITE_SPLUNK_INDEX") or os.getenv("SPLUNK_INDEX", "textthreat"),
    }


def event_from_splunk_result(result: dict[str, Any]) -> dict[str, Any] | None:
    """Convert a Splunk search result into a TextThreat event dictionary."""
    raw = result.get("_raw")
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict) and "digital_wellbeing" in parsed:
                return parsed
            if isinstance(parsed, dict) and isinstance(parsed.get("event"), dict):
                return parsed["event"]
        except json.JSONDecodeError:
            pass

    harms = result.get("digital_wellbeing.harm_types{}") or result.get("digital_wellbeing.harm_types") or []
    if isinstance(harms, str):
        harms = [harms]
    return {
        "@timestamp": result.get("@timestamp") or result.get("_time"),
        "text_hash": result.get("text_hash"),
        "session_id": result.get("session_id"),
        "digital_wellbeing": {
            "harm_types": list(harms),
            "risk_score": float(result.get("digital_wellbeing.risk_score", 0.0)),
            "model_version": result.get("digital_wellbeing.model_version"),
        },
    }


def poll_splunk_once(threshold: float = HIGH_RISK_THRESHOLD, earliest: str = "-2m") -> list[dict[str, Any]]:
    """Query Splunk once and process the returned TextThreat alert batch."""
    try:
        import splunklib.client as splunk_client
    except ImportError as exc:
        raise SystemExit("Install splunk-sdk to use --poll-splunk: pip install splunk-sdk") from exc

    config = splunk_config_from_env()
    kwargs: dict[str, Any] = {
        "host": config["host"],
        "port": config["port"],
        "scheme": config["scheme"],
    }
    if config["token"]:
        kwargs["splunkToken"] = config["token"]
    else:
        kwargs["username"] = config["username"]
        kwargs["password"] = config["password"]
    if not config["verify"]:
        kwargs["verify"] = False

    service = splunk_client.connect(**kwargs)
    search = (
        f"search index={config['index']} event.module=textthreat earliest={earliest} "
        "| sort - _time | head 200"
    )
    stream = service.jobs.oneshot(search, output_mode="json")
    payload = json.loads(stream.read().decode("utf-8"))
    events = [event for result in payload.get("results", []) if (event := event_from_splunk_result(result))]
    return process_events(events, threshold=threshold)


def poll_splunk_forever(interval: int, threshold: float, earliest: str) -> None:
    """Run the SOAR-lite Splunk polling daemon."""
    print(f"SOAR-lite Splunk poller started: interval={interval}s threshold={threshold}")
    while True:
        try:
            alerts = poll_splunk_once(threshold=threshold, earliest=earliest)
            print(f"{now_utc_iso()} processed batch; alerts={len(alerts)}")
        except Exception as exc:  # noqa: BLE001 - daemon should keep running after transient Splunk errors.
            print(f"{now_utc_iso()} SOAR-lite poll error: {exc}", file=sys.stderr)
        time.sleep(interval)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run TextThreat SOAR-lite alert dispatcher.")
    parser.add_argument("--input", type=Path, default=SAMPLE_OUTPUT, help="NDJSON event file.")
    parser.add_argument("--demo", action="store_true", help="Generate sample events first.")
    parser.add_argument("--threshold", type=float, default=HIGH_RISK_THRESHOLD)
    parser.add_argument("--poll-splunk", action="store_true", help="Run the optional Splunk polling daemon.")
    parser.add_argument("--once", action="store_true", help="Poll Splunk once instead of running forever.")
    parser.add_argument("--interval", type=int, default=30, help="Splunk polling interval in seconds.")
    parser.add_argument("--earliest", default="-2m", help="Splunk earliest time for each polling batch.")
    return parser


def main(argv: list[str] | None = None) -> list[dict[str, Any]]:
    args = build_arg_parser().parse_args(argv)
    if args.poll_splunk:
        if args.once:
            alerts = poll_splunk_once(threshold=args.threshold, earliest=args.earliest)
            print(f"Processed one Splunk batch; alerts={len(alerts)}")
            return alerts
        poll_splunk_forever(args.interval, args.threshold, args.earliest)
        return []
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
