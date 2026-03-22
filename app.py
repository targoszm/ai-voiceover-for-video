import base64
import os
import tempfile
from pathlib import Path
from typing import Sequence

import cv2
import streamlit as st
from dotenv import load_dotenv
from moviepy.editor import AudioFileClip, VideoFileClip
from openai import OpenAI


load_dotenv()


def _ensure_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    return api_key


def video_to_frames(video_file_path: str, sample_every_n_frames: int = 25) -> tuple[list[str], float]:
    capture = cv2.VideoCapture(video_file_path)
    if not capture.isOpened():
        raise RuntimeError("Could not open uploaded video file.")

    fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps if fps > 0 else 0.0

    base64_frames: list[str] = []
    index = 0
    while capture.isOpened():
        success, frame = capture.read()
        if not success:
            break
        if index % sample_every_n_frames == 0:
            ok, buffer = cv2.imencode(".jpg", frame)
            if ok:
                base64_frames.append(base64.b64encode(buffer).decode("utf-8"))
        index += 1

    capture.release()
    if not base64_frames:
        raise RuntimeError("No video frames were extracted from this file.")
    return base64_frames, duration


def pick_representative_frames(base64_frames: Sequence[str], max_frames: int = 20) -> list[str]:
    if len(base64_frames) <= max_frames:
        return list(base64_frames)
    step = max(1, len(base64_frames) // max_frames)
    sampled = list(base64_frames[::step])
    return sampled[:max_frames]


def frames_to_story(client: OpenAI, base64_frames: list[str], prompt: str) -> str:
    content = [{"type": "text", "text": prompt}]
    for frame in base64_frames:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{frame}"},
            }
        )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": content}],
        max_tokens=500,
    )
    return response.choices[0].message.content or ""


def text_to_audio(client: OpenAI, text: str) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        speech = client.audio.speech.create(
            model="tts-1",
            voice="onyx",
            input=text,
        )
        speech.stream_to_file(tmp_file.name)
        return tmp_file.name


def merge_audio_video(video_filename: str, audio_filename: str, output_filename: str) -> str:
    video_clip = VideoFileClip(video_filename)
    audio_clip = AudioFileClip(audio_filename)
    final_clip = video_clip.set_audio(audio_clip)
    final_clip.write_videofile(output_filename, codec="libx264", audio_codec="aac", logger=None)
    final_clip.close()
    video_clip.close()
    audio_clip.close()
    return output_filename


def save_uploaded_video(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        return tmp_file.name


def main() -> None:
    st.set_page_config(page_title="AI Storyteller Voiceovers", page_icon="🎙️")
    st.title("AI Storyteller Voiceovers")
    st.caption("Upload a short video and generate a matching narration using vision + TTS.")

    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "webm", "mkv"])
    prompt = st.text_area(
        "Voiceover prompt",
        value=(
            "Create a concise and engaging voiceover script for this clip. "
            "Describe what is happening and keep the tone natural."
        ),
        height=120,
    )

    if uploaded_file is not None:
        st.video(uploaded_file)

    if st.button("Generate voiceover", type="primary", disabled=uploaded_file is None):
        if uploaded_file is None:
            st.warning("Please upload a video first.")
            return

        temp_paths: list[str] = []
        try:
            client = OpenAI(api_key=_ensure_api_key())
            with st.spinner("Extracting frames..."):
                video_filename = save_uploaded_video(uploaded_file)
                temp_paths.append(video_filename)
                frames, video_duration = video_to_frames(video_filename)
                selected_frames = pick_representative_frames(frames, max_frames=20)

            est_word_count = max(20, int(video_duration * 2.25))
            final_prompt = (
                f"{prompt}\n\n"
                f"The video is approximately {video_duration:.1f} seconds long. "
                f"Write the script in at most {est_word_count} words."
            )

            with st.spinner("Generating story from frames..."):
                generated_text = frames_to_story(client, selected_frames, final_prompt)

            if not generated_text.strip():
                raise RuntimeError("The model returned an empty script.")

            st.subheader("Generated script")
            st.write(generated_text)

            with st.spinner("Generating speech..."):
                audio_filename = text_to_audio(client, generated_text)
                temp_paths.append(audio_filename)
                with open(audio_filename, "rb") as audio_file:
                    audio_bytes = audio_file.read()

            output_video = f"{Path(video_filename).with_suffix('')}_voiceover.mp4"
            with st.spinner("Merging audio with video..."):
                final_video = merge_audio_video(video_filename, audio_filename, output_video)
                temp_paths.append(final_video)
                with open(final_video, "rb") as video_file:
                    video_bytes = video_file.read()

            st.subheader("Final video")
            st.video(video_bytes)
            st.download_button(
                "Download generated audio",
                data=audio_bytes,
                file_name="voiceover.mp3",
                mime="audio/mpeg",
            )
            st.download_button(
                "Download final video",
                data=video_bytes,
                file_name="voiceover_video.mp4",
                mime="video/mp4",
            )

        except Exception as exc:  # noqa: BLE001
            st.error(f"Failed to generate voiceover: {exc}")
        finally:
            for path in temp_paths:
                try:
                    os.unlink(path)
                except OSError:
                    pass


if __name__ == "__main__":
    main()
