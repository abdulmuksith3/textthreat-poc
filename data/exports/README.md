# TextThreat Exports

Generated TextThreat events are newline-delimited JSON files.

The committed sample export is:

```text
data/exports/sample_textthreat_events.ndjson
```

Generate it with:

```bash
python -m src.textthreat.export_events --sample
```

Production-style exports contain hashes and model metadata, not raw comment text.
