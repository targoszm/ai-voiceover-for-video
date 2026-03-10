from __future__ import annotations

import json
import os
import subprocess
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, HttpUrl

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
GENERATED_DIR = BASE_DIR / "generated"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
GENERATED_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="AI Voiceover for Video MVP")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


class CTA(BaseModel):
    text: str
    url: HttpUrl


class LeadForm(BaseModel):
    email: str
    name: str
    company: str | None = None
    company_size: str | None = None


def probe_video_duration_seconds(video_path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    try:
        output = subprocess.check_output(cmd, text=True).strip()
        return float(output)
    except (subprocess.CalledProcessError, ValueError, FileNotFoundError) as exc:
        raise HTTPException(
            status_code=400,
            detail="Unable to read MP4 duration via ffprobe",
        ) from exc


def detect_hotspots(duration_seconds: float) -> list[dict[str, Any]]:
    checkpoints = [0.15, 0.35, 0.55, 0.75]
    hotspots: list[dict[str, Any]] = []
    for idx, ratio in enumerate(checkpoints, start=1):
        ts = round(duration_seconds * ratio, 2)
        hotspots.append(
            {
                "timestamp": ts,
                "screen": f"Screen {idx}",
                "hotspot": "Primary action area highlighted",
                "commentary_hint": "Explain what the viewer can do in this section.",
            }
        )
    return hotspots


def build_commentary(hotspots: list[dict[str, Any]], ctas: list[CTA]) -> str:
    lines = [
        "Welcome! In this walkthrough, we will quickly explore the product highlights on screen.",
    ]
    for hotspot in hotspots:
        lines.append(
            f"At {hotspot['timestamp']} seconds, we reach {hotspot['screen']}. "
            "This is a key moment where users can take action confidently."
        )
    if ctas:
        lines.append("Before we finish, here are your next steps:")
    for cta in ctas:
        lines.append(f"{cta.text}: {cta.url}")
    lines.append("Thanks for watching—let's get started.")
    return "\n".join(lines)


@app.get("/", response_class=HTMLResponse)
def home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("index.html", {"request": request, "result": None})


@app.post("/generate", response_class=HTMLResponse)
async def generate_voiceover(
    request: Request,
    video: UploadFile = File(...),
    cta_text: str = Form("Start your free trial now"),
    cta_url: str = Form("https://training.skillstudio.ai/signup"),
) -> HTMLResponse:
    if video.content_type not in {"video/mp4", "application/mp4"}:
        raise HTTPException(status_code=400, detail="Only MP4 files are supported")

    project_id = str(uuid.uuid4())
    source_path = UPLOAD_DIR / f"{project_id}.mp4"
    content = await video.read()
    source_path.write_bytes(content)

    duration = probe_video_duration_seconds(source_path)
    if duration < 30 or duration > 180:
        source_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail="Video duration must be between 30 and 180 seconds",
        )

    ctas = [CTA(text=cta_text, url=cta_url)]
    hotspots = detect_hotspots(duration)
    script = build_commentary(hotspots, ctas)

    metadata = {
        "project_id": project_id,
        "duration_seconds": duration,
        "hotspots": hotspots,
        "script": script,
        "ctas": [c.model_dump() for c in ctas],
    }
    (GENERATED_DIR / f"{project_id}.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": metadata,
        },
    )


@app.post("/lead-capture")
async def capture_lead(form: LeadForm) -> dict[str, str]:
    file_path = GENERATED_DIR / "leads.jsonl"
    with file_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(form.model_dump()) + os.linesep)
    return {"status": "ok"}
