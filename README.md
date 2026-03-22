# AI Storyteller Voiceovers

Create narrated videos from short clips using OpenAI vision + text-to-speech models.

This project is inspired by the workflow in:
https://henrywithu.com/the-ai-storyteller-creating-video-voiceovers-with-gpt-4-vision/

## What it does

1. Upload a short video clip from a web UI.
2. Sample representative video frames.
3. Ask a vision-capable model to generate a concise voiceover script.
4. Convert that script to speech.
5. Merge the generated audio back into the original video.

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

2. Set environment variable in Vercel project:

- `OPENAI_API_KEY`

3. Deploy:

```bash
vercel
```

## Notes

- The app limits the number of frames sent to the model to control latency and token usage.
- Temporary files are cleaned up automatically after each generation.
- This repository still contains the original `app.py` Streamlit implementation for local experiments.
