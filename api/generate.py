import base64
import os
import tempfile
import urllib.parse
from pathlib import Path

import cv2
from dotenv import load_dotenv
from flask import Flask, after_this_request, jsonify, make_response, request, send_file
from openai import OpenAI


load_dotenv()
app = Flask(__name__)

DEFAULT_PROMPT = (
    "Create a concise and engaging voiceover script for this clip. "
    "Describe what is happening and keep the tone natural."
)
DEFAULT_MODE = "audio_only"
VALID_MODES = {"audio_only", "merge_video"}

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(25 * 1024 * 1024)))
MAX_VIDEO_DURATION_SECONDS = float(os.getenv("MAX_VIDEO_DURATION_SECONDS", "75"))
MAX_VIDEO_DURATION_FOR_MERGE_SECONDS = float(os.getenv("MAX_VIDEO_DURATION_FOR_MERGE_SECONDS", "20"))
MAX_FRAMES_FOR_VISION = int(os.getenv("MAX_FRAMES_FOR_VISION", "10"))
SAMPLE_EVERY_N_FRAMES = int(os.getenv("SAMPLE_EVERY_N_FRAMES", "30"))
MAX_IMAGE_EDGE_PX = int(os.getenv("MAX_IMAGE_EDGE_PX", "768"))
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
TTS_MODEL = os.getenv("OPENAI_TTS_MODEL", "tts-1")
TTS_VOICE = os.getenv("OPENAI_TTS_VOICE", "onyx")


def _ensure_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    return api_key


def _validate_mode() -> str:
    mode = request.form.get("mode", DEFAULT_MODE).strip().lower()
    if mode not in VALID_MODES:
        raise RuntimeError("Invalid mode. Use 'audio_only' or 'merge_video'.")
    return mode


def _read_video_duration_and_frames(
    video_file_path: str, sample_every_n_frames: int = SAMPLE_EVERY_N_FRAMES
) -> tuple[list[str], float]:
    capture = cv2.VideoCapture(video_file_path)
    if not capture.isOpened():
        raise RuntimeError("Could not open uploaded video file.")

    fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps if fps > 0 else 0.0

    if duration <= 0:
        capture.release()
        raise RuntimeError("Could not determine video duration.")

    if duration > MAX_VIDEO_DURATION_SECONDS:
        capture.release()
        raise RuntimeError(
            f"Video is too long ({duration:.1f}s). Max allowed is {MAX_VIDEO_DURATION_SECONDS:.0f}s."
        )

    base64_frames: list[str] = []
    index = 0
    while capture.isOpened():
        success, frame = capture.read()
        if not success:
            break
        if index % sample_every_n_frames == 0:
            frame = _resize_frame_if_needed(frame)
            ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if ok:
                base64_frames.append(base64.b64encode(buffer).decode("utf-8"))
        index += 1

    capture.release()
    if not base64_frames:
        raise RuntimeError("No video frames were extracted from this file.")
    return base64_frames, duration


def _resize_frame_if_needed(frame):
    height, width = frame.shape[:2]
    longest = max(height, width)
    if longest <= MAX_IMAGE_EDGE_PX:
        return frame
    scale = MAX_IMAGE_EDGE_PX / float(longest)
    target = (max(1, int(width * scale)), max(1, int(height * scale)))
    return cv2.resize(frame, target, interpolation=cv2.INTER_AREA)


def _pick_representative_frames(base64_frames: list[str], max_frames: int = MAX_FRAMES_FOR_VISION) -> list[str]:
    if len(base64_frames) <= max_frames:
        return base64_frames
    step = max(1, len(base64_frames) // max_frames)
    return base64_frames[::step][:max_frames]


def _frames_to_story(client: OpenAI, base64_frames: list[str], prompt: str) -> str:
    content = [{"type": "text", "text": prompt}]
    for frame in base64_frames:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{frame}"},
            }
        )

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": content}],
        max_tokens=320,
    )
    return response.choices[0].message.content or ""


def _text_to_audio(client: OpenAI, text: str) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        speech = client.audio.speech.create(
            model=TTS_MODEL,
            voice=TTS_VOICE,
            input=text,
        )
        speech.stream_to_file(tmp_file.name)
        return tmp_file.name


def _merge_audio_video(video_filename: str, audio_filename: str, output_filename: str) -> str:
    # Lazy import keeps cold-start overhead lower for audio-only requests.
    from moviepy.editor import AudioFileClip, VideoFileClip

    video_clip = VideoFileClip(video_filename)
    audio_clip = AudioFileClip(audio_filename)
    merged_audio = audio_clip.subclip(0, min(audio_clip.duration, video_clip.duration))
    final_clip = video_clip.set_audio(merged_audio)
    final_clip.write_videofile(
        output_filename,
        codec="libx264",
        audio_codec="aac",
        preset="ultrafast",
        threads=1,
        logger=None,
    )
    final_clip.close()
    merged_audio.close()
    video_clip.close()
    audio_clip.close()
    return output_filename


def _save_uploaded_video() -> tuple[str, int]:
    if "video" not in request.files:
        raise RuntimeError("Missing video upload.")
    upload = request.files["video"]
    if not upload or upload.filename is None or upload.filename.strip() == "":
        raise RuntimeError("Video file is empty.")
    if request.content_length and request.content_length > MAX_UPLOAD_BYTES * 2:
        raise RuntimeError(f"Uploaded payload is too large. Max size is {MAX_UPLOAD_BYTES // (1024 * 1024)}MB.")

    suffix = Path(upload.filename).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        upload.save(tmp_file.name)
        size_bytes = os.path.getsize(tmp_file.name)
        if size_bytes <= 0:
            raise RuntimeError("Video file is empty.")
        if size_bytes > MAX_UPLOAD_BYTES:
            raise RuntimeError(f"Video is too large ({size_bytes // (1024 * 1024)}MB).")
        return tmp_file.name, size_bytes


@app.get("/api/health")
def health():
    return jsonify({"ok": True})


@app.post("/api/generate")
def generate():
    temp_paths: list[str] = []
    try:
        mode = _validate_mode()
        prompt = request.form.get("prompt", DEFAULT_PROMPT).strip() or DEFAULT_PROMPT
        client = OpenAI(api_key=_ensure_api_key())
        video_filename, _ = _save_uploaded_video()
        temp_paths.append(video_filename)

        frames, video_duration = _read_video_duration_and_frames(video_filename)
        selected_frames = _pick_representative_frames(frames)
        est_word_count = max(20, int(video_duration * 2.25))
        final_prompt = (
            f"{prompt}\n\n"
            f"The video is approximately {video_duration:.1f} seconds long. "
            f"Write the script in at most {est_word_count} words."
        )
        generated_text = _frames_to_story(client, selected_frames, final_prompt).strip()
        if not generated_text:
            raise RuntimeError("The model returned an empty script.")

        audio_filename = _text_to_audio(client, generated_text)
        temp_paths.append(audio_filename)

        output_path = audio_filename
        output_mime = "audio/mpeg"
        output_name = "voiceover.mp3"
        output_type = "audio"
        serverless_notice = ""

        if mode == "merge_video":
            if video_duration <= MAX_VIDEO_DURATION_FOR_MERGE_SECONDS:
                output_video = f"{Path(video_filename).with_suffix('')}_voiceover.mp4"
                final_video = _merge_audio_video(video_filename, audio_filename, output_video)
                temp_paths.append(final_video)
                output_path = final_video
                output_mime = "video/mp4"
                output_name = "voiceover_video.mp4"
                output_type = "video"
            else:
                serverless_notice = (
                    f"Returned audio-only because merge mode is limited to {MAX_VIDEO_DURATION_FOR_MERGE_SECONDS:.0f}s."
                )

        @after_this_request
        def cleanup(response):  # type: ignore[no-redef]
            for path in temp_paths:
                try:
                    os.unlink(path)
                except OSError:
                    pass
            return response

        response = make_response(
            send_file(
                output_path,
                mimetype=output_mime,
                as_attachment=True,
                download_name=output_name,
            )
        )
        response.headers["X-Voiceover-Script"] = urllib.parse.quote(generated_text[:7000], safe="")
        response.headers["X-Output-Type"] = output_type
        response.headers["X-Video-Duration"] = f"{video_duration:.2f}"
        if serverless_notice:
            response.headers["X-Serverless-Notice"] = serverless_notice
        return response
    except Exception as exc:  # noqa: BLE001
        for path in temp_paths:
            try:
                os.unlink(path)
            except OSError:
                pass
        return jsonify({"error": str(exc)}), 400
