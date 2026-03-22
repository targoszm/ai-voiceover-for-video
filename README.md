# AI Storyteller Voiceovers

Create narrated videos from short clips using OpenAI vision + text-to-speech models.

This project is inspired by the workflow in:
https://henrywithu.com/the-ai-storyteller-creating-video-voiceovers-with-gpt-4-vision/

## What it does

1. Upload a short video clip from a web UI.
2. Sample representative video frames.
3. Ask a vision-capable model to generate a concise voiceover script.
4. Convert that script to speech.
5. Return generated audio by default (serverless-friendly), with optional video merge for short clips.

## Requirements

- Python 3.10+
- FFmpeg installed and available on PATH (required by MoviePy)
- OpenAI API key

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Then edit `.env` and set:

```bash
OPENAI_API_KEY=your_real_key
```

## Run locally (web UI + API)

```bash
python3 -m flask --app api/generate.py run --host 0.0.0.0 --port 5000
```

Then open `index.html` with any static file server (or run through Vercel dev):

```bash
npx vercel dev
```

Open the shown local URL, upload a video, adjust the prompt, and generate the voiceover.

## Deploy on Vercel

1. Install Vercel CLI and login:

```bash
npm i -g vercel
vercel login
```

2. Set required environment variable in Vercel project:

- `OPENAI_API_KEY`

3. Optional tuning variables (recommended for serverless reliability):

- `MAX_UPLOAD_BYTES` (default: `26214400` = 25 MB)
- `MAX_VIDEO_DURATION_SECONDS` (default: `75`)
- `MAX_VIDEO_DURATION_FOR_MERGE_SECONDS` (default: `20`)
- `MAX_FRAMES_FOR_VISION` (default: `10`)
- `SAMPLE_EVERY_N_FRAMES` (default: `30`)
- `MAX_IMAGE_EDGE_PX` (default: `768`)

4. Deploy:

```bash
vercel
```

## Vercel runtime behavior

- The UI defaults to **Audio only** mode because it is much more reliable in serverless environments.
- **Merge narration into video** is supported for short clips only.
- If a clip is too long for merge mode, the API automatically returns audio and includes a notice in the UI.

## Notes

- The app limits image/frame size and frame count to control latency and token usage.
- Temporary files are cleaned up automatically after each generation.
- This repository still contains the original `app.py` Streamlit implementation for local experiments.
