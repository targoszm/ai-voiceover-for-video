# AI Voiceover for Video (MVP)

A lightweight FastAPI web app that:
- accepts MP4 uploads (30s to 3m),
- detects timeline hotspots,
- generates narration commentary text,
- lets you configure a CTA link,
- captures leads via an embeddable form.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
uvicorn app.main:app --reload
```

Then open http://localhost:8000

## Notes

- The current MVP uses deterministic hotspot/commentary generation.
- `ffprobe` is required on PATH for duration checks.
