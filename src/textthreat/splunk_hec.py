"""Splunk HTTP Event Collector client for the hosted TextThreat demo."""

from __future__ import annotations

import json
import os
import ssl
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SplunkHECConfig:
    """Configuration for Splunk HTTP Event Collector."""

    url: str
    token: str
    index: str = "textthreat"
    sourcetype: str = "_json"
    source: str = "textthreat-demo"


def config_from_env() -> SplunkHECConfig | None:
    """Build Splunk HEC config from environment variables when present."""
    url = os.getenv("SPLUNK_HEC_URL")
    token = os.getenv("SPLUNK_HEC_TOKEN")
    if not url or not token:
        return None
    return SplunkHECConfig(
        url=url.rstrip("/"),
        token=token,
        index=os.getenv("SPLUNK_INDEX", "textthreat"),
        sourcetype=os.getenv("SPLUNK_SOURCETYPE", "_json"),
        source=os.getenv("SPLUNK_SOURCE", "textthreat-demo"),
    )


def send_event(event: dict[str, Any], config: SplunkHECConfig | None = None, timeout: int | None = None) -> dict[str, Any]:
    """Send one TextThreat event to Splunk HEC."""
    config = config or config_from_env()
    if config is None:
        return {"sent": False, "status": "skipped", "reason": "Splunk HEC environment variables are not configured."}
    timeout = timeout or int(os.getenv("SPLUNK_HEC_TIMEOUT_SECONDS", "5"))
    verify_ssl = os.getenv("SPLUNK_VERIFY_SSL", "true").lower() not in {"0", "false", "no"}
    context = None if verify_ssl else ssl._create_unverified_context()

    endpoint = config.url
    if not endpoint.endswith("/services/collector/event"):
        endpoint = endpoint.rstrip("/") + "/services/collector/event"
    payload = {
        "event": event,
        "index": config.index,
        "sourcetype": config.sourcetype,
        "source": config.source,
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Splunk {config.token}",
            "Content-Type": "application/json",
            "X-Splunk-Request-Channel": os.getenv("SPLUNK_HEC_CHANNEL", str(uuid.uuid4())),
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
            body = response.read().decode("utf-8")
            return {"sent": True, "status": response.status, "response": body}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return {"sent": False, "status": exc.code, "error": body}
    except urllib.error.URLError as exc:
        return {"sent": False, "status": "connection_error", "error": str(exc.reason)}
