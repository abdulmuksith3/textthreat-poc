"""Configure the TextThreat Splunk Cloud demo and seed sample events."""

from __future__ import annotations

import argparse
import base64
import os
import ssl
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / "config" / "settings.env")
load_dotenv(ROOT / ".env")

from src.textthreat.export_events import sample_prediction_rows
from src.textthreat.schema import build_event
from src.textthreat.splunk_hec import send_event


DASHBOARD_XML = ROOT / "siem" / "splunk" / "textthreat_dashboard.xml"


def env(name: str, default: str | None = None) -> str | None:
    """Read an environment variable."""
    return os.getenv(name, default)


def require_env(name: str) -> str:
    """Read a required environment variable."""
    value = env(name)
    if not value:
        raise SystemExit(f"Missing {name}. Set it before running this script.")
    return value


def management_url() -> str | None:
    """Return Splunk management URL when configured."""
    url = env("SPLUNK_MANAGEMENT_URL") or env("SPLUNK_MGMT_URL")
    if url:
        return url.rstrip("/")
    host = env("SPLUNK_HOST")
    if host:
        host = host.removeprefix("https://").removeprefix("http://").rstrip("/")
        return f"https://{host}:8089"
    return None


def splunk_request(path: str, data: dict[str, str] | None = None) -> tuple[int, str]:
    """Call the Splunk management API using bearer token or basic auth."""
    base_url = management_url()
    if not base_url:
        raise SystemExit("Missing SPLUNK_MANAGEMENT_URL or SPLUNK_HOST for dashboard/index setup.")
    url = f"{base_url}{path}"
    body = urllib.parse.urlencode(data or {}).encode("utf-8")
    api_token = env("SPLUNK_API_TOKEN") or env("SPLUNK_ACCESS_TOKEN")
    if api_token:
        auth_value = f"Bearer {api_token}"
    else:
        username = require_env("SPLUNK_USERNAME")
        password = require_env("SPLUNK_PASSWORD")
        auth_header = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode("ascii")
        auth_value = f"Basic {auth_header}"
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Authorization": auth_value, "Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    verify_ssl = os.getenv("SPLUNK_VERIFY_SSL", "true").lower() not in {"0", "false", "no"}
    context = ssl.create_default_context() if verify_ssl else ssl._create_unverified_context()
    try:
        opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=context))
        with opener.open(request, timeout=20) as response:
            return response.status, response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as exc:
        return 0, f"connection_error: {exc.reason}"


def create_index(index: str) -> None:
    """Create the Splunk index if management API access is available."""
    status, body = splunk_request("/services/data/indexes", {"name": index})
    if status in {200, 201}:
        print(f"Created Splunk index: {index}")
    elif status == 409 or "already exists" in body.lower():
        print(f"Splunk index already exists: {index}")
    else:
        print(f"Index setup returned HTTP {status}: {body[:500]}")


def create_dashboard(index: str) -> None:
    """Create/update the TextThreat dashboard in Splunk Search app."""
    owner = env("SPLUNK_DASHBOARD_OWNER", "nobody")
    app = env("SPLUNK_DASHBOARD_APP", "search")
    dashboard_name = env("SPLUNK_DASHBOARD_NAME", "textthreat_digital_wellbeing_risk")
    xml = DASHBOARD_XML.read_text(encoding="utf-8").replace("index=textthreat", f"index={index}")
    data = {
        "name": dashboard_name,
        "eai:data": xml,
        "isDashboard": "1",
    }
    status, body = splunk_request(f"/servicesNS/{owner}/{app}/data/ui/views", data)
    if status in {200, 201}:
        print(f"Created Splunk dashboard: {dashboard_name}")
    elif status == 409 or "already exists" in body.lower():
        status, body = splunk_request(f"/servicesNS/{owner}/{app}/data/ui/views/{dashboard_name}", {"eai:data": xml})
        print(f"Updated Splunk dashboard: {dashboard_name} (HTTP {status})")
    else:
        print(f"Dashboard setup returned HTTP {status}: {body[:500]}")


def demo_rows() -> list[dict[str, Any]]:
    """Return demo rows that exercise dashboard panels."""
    rows = sample_prediction_rows(5)
    for row in rows:
        row["source_platform"] = "demo_form"
    rows.extend(
        [
            {
                "text": "synthetic demo direct threat sample",
                "source_platform": "demo_form",
                "session_id": "splunk-demo-001",
                "scores": {"toxic": 0.91, "threat": 0.95, "insult": 0.68},
            },
            {
                "text": "synthetic demo stress sample",
                "source_platform": "demo_form",
                "session_id": "splunk-demo-001",
                "scores": {"stress": 0.89, "toxic": 0.12},
            },
            {
                "text": "synthetic benign sample",
                "source_platform": "demo_form",
                "session_id": "splunk-demo-002",
                "scores": {"toxic": 0.08, "stress": 0.11},
            },
        ]
    )
    return rows


def send_demo_events() -> None:
    """Send sample TextThreat events to Splunk HEC."""
    sent = 0
    for row in demo_rows():
        event = build_event(
            row["text"],
            row["scores"],
            row.get("source_platform", "demo_form"),
            session_id=row.get("session_id"),
        )
        status = send_event(event)
        print(f"HEC event {event['text_hash'][:10]} status: {status}")
        if status.get("sent"):
            sent += 1
    print(f"Sent {sent}/{len(demo_rows())} demo events to Splunk HEC.")


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Set up TextThreat Splunk Cloud demo.")
    parser.add_argument("--events-only", action="store_true", help="Only send sample HEC events.")
    parser.add_argument("--dashboard-only", action="store_true", help="Only create index/dashboard via management API.")
    parser.add_argument("--index", default=env("SPLUNK_INDEX", "textthreat"))
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run Splunk setup."""
    args = build_arg_parser().parse_args(argv)
    if not args.dashboard_only:
        require_env("SPLUNK_HEC_URL")
        require_env("SPLUNK_HEC_TOKEN")
        send_demo_events()
    if not args.events_only:
        create_index(args.index)
        create_dashboard(args.index)


if __name__ == "__main__":
    main()
