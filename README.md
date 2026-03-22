# AI Storyteller Voiceovers

Create narrated videos from short clips using OpenAI vision + text-to-speech models.

This project is inspired by the workflow in:
https://henrywithu.com/the-ai-storyteller-creating-video-voiceovers-with-gpt-4-vision/

## What it does

1. Upload a short video clip in Streamlit.
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

## Run

```bash
streamlit run app.py
```

Open the local URL shown by Streamlit, upload a video, adjust the prompt, and generate the voiceover.

## Notes

- The app limits the number of frames sent to the model to control latency and token usage.
- Temporary files are cleaned up automatically after each generation.
