"""Gradio app for the hosted TextThreat-to-Splunk demo."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import gradio as gr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.textthreat.inference import TextThreatAnalyzer
from src.textthreat.schema import build_event
from src.textthreat.splunk_hec import send_event


analyzer = TextThreatAnalyzer()


def analyze_comment(comment: str, source_platform: str, session_id: str) -> tuple[str, str]:
    """Analyze one comment, send the event to Splunk when configured, and return UI output."""
    comment = (comment or "").strip()
    if not comment:
        return "Enter a comment to analyze.", "{}"

    result = analyzer.analyze(comment)
    event = build_event(
        comment,
        result["scores"],
        source_platform or "demo_form",
        model_version=result["model_version"],
        session_id=session_id.strip() or None,
    )
    splunk_status = send_event(event)
    harm_types = event["digital_wellbeing"]["harm_types"] or ["none_above_threshold"]
    risk = event["digital_wellbeing"]["risk_score"]
    summary = "\n".join(
        [
            f"Risk score: {risk:.3f}",
            f"Harm types: {', '.join(harm_types)}",
            f"Text hash: {event['text_hash']}",
            f"Inference: {', '.join(result['inference_modes'])}",
            f"Splunk: {splunk_status.get('status')}",
        ]
    )
    payload: dict[str, Any] = {"event": event, "splunk_status": splunk_status}
    return summary, json.dumps(payload, indent=2)


with gr.Blocks(title="TextThreat Splunk Demo") as demo:
    gr.Markdown("# TextThreat Splunk Demo")
    gr.Markdown("Submit a comment to generate a schema-valid TextThreat event and send it to Splunk HEC when configured.")
    with gr.Row():
        comment = gr.Textbox(label="Comment", lines=6, placeholder="Paste a social media comment for analysis.")
    with gr.Row():
        source_platform = gr.Textbox(label="Source platform", value="demo_form")
        session_id = gr.Textbox(label="Session ID", value="demo-session")
    submit = gr.Button("Analyze & Send to Splunk", variant="primary")
    summary = gr.Textbox(label="Result", lines=6)
    event_json = gr.Code(label="Event and Splunk status", language="json")
    submit.click(analyze_comment, inputs=[comment, source_platform, session_id], outputs=[summary, event_json])


if __name__ == "__main__":
    demo.launch(server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"), server_port=int(os.getenv("PORT", "7860")))
