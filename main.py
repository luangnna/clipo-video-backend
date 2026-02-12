"""
CLIPO - Backend Externo de Processamento de Vídeo
Deploy: Railway ou Render (Python 3.11+)

Pipeline: Download (yt-dlp) → Whisper → AI Analysis → FFmpeg (9:16) → Upload → Webhook

Endpoints:
  POST /process  — Recebe job do Edge Function process-video
  GET  /health   — Health check
"""

import os
import json
import uuid
import subprocess
import tempfile
import shutil
import traceback
from pathlib import Path
from typing import Optional

import requests
import whisper
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="CLIPO Video Processor")

# ---------------------------------------------------------------------------
# Whisper (lazy singleton)
# ---------------------------------------------------------------------------

_whisper_model = None


def get_whisper_model(model_size: str = "base"):
    global _whisper_model
    if _whisper_model is None:
        print(f"[whisper] Loading model: {model_size}")
        _whisper_model = whisper.load_model(model_size)
    return _whisper_model


# ---------------------------------------------------------------------------
# Request schema — matches the payload sent by process-video Edge Function
# ---------------------------------------------------------------------------

class ProcessRequest(BaseModel):
    url: str
    project_id: str
    callback_url: str          # → video-webhook Edge Function URL
    webhook_secret: str
    supabase_url: str
    supabase_key: str          # service-role key (for Storage uploads)
    storage_bucket: str = "videos"
    ai_analyze_url: Optional[str] = None   # → ai-analyze-content Edge Function URL
    ai_webhook_secret: Optional[str] = None
    whisper_config: Optional[dict] = None  # { "language": "pt", "model_size": "base" }


# ---------------------------------------------------------------------------
# Webhook helpers — communicate with video-webhook Edge Function
# ---------------------------------------------------------------------------

def send_progress(callback_url: str, secret: str, project_id: str, progress: int):
    """Send a progress-only update (no clips). Webhook stores progress."""
    try:
        requests.post(callback_url, json={
            "project_id": project_id,
            "secret": secret,
            "progress": progress,
        }, timeout=10)
    except Exception as e:
        print(f"[progress] Failed: {e}")


def send_error(callback_url: str, secret: str, project_id: str, error_msg: str):
    """Report a processing failure. Webhook sets project status='error'."""
    try:
        requests.post(callback_url, json={
            "project_id": project_id,
            "secret": secret,
            "error": error_msg,
        }, timeout=10)
    except Exception as e:
        print(f"[error-callback] Failed: {e}")


# ---------------------------------------------------------------------------
# Step 1 — Download video with yt-dlp
# ---------------------------------------------------------------------------

def download_video(url: str, output_dir: str) -> str:
    output_path = os.path.join(output_dir, "source.mp4")
    cmd = [
        "yt-dlp",
        "-f", "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080][ext=mp4]/best",
        "--merge-output-format", "mp4",
        "-o", output_path,
        "--no-playlist",
        url,
    ]
    print(f"[download] {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed: {result.stderr[:500]}")
    print(f"[download] OK → {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Step 2 — Transcribe with Whisper
# ---------------------------------------------------------------------------

def transcribe_audio(video_path: str, config: dict | None = None) -> dict:
    config = config or {}
    language = config.get("language", "pt")
    model_size = config.get("model_size", "base")

    model = get_whisper_model(model_size)
    print(f"[whisper] Transcribing (lang={language}, model={model_size})")

    result = model.transcribe(video_path, language=language, verbose=False)

    segments = [
        {
            "start": round(seg["start"], 2),
            "end": round(seg["end"], 2),
            "text": seg["text"].strip(),
        }
        for seg in result.get("segments", [])
        if seg["text"].strip()
    ]

    full_text = result.get("text", "").strip()
    print(f"[whisper] Done: {len(segments)} segments, {len(full_text)} chars")
    return {"transcription": full_text, "segments": segments}


# ---------------------------------------------------------------------------
# Step 3 — AI analysis via ai-analyze-content Edge Function
# ---------------------------------------------------------------------------

def analyze_content(
    ai_url: str,
    secret: str,
    transcription: str,
    segments: list,
    duration: float,
    title: str,
) -> list:
    """
    Call the ai-analyze-content Edge Function.
    It uses the Lovable AI Gateway (Gemini) to detect viral moments.
    Returns a list of moment dicts with start_time, end_time, title, etc.
    """
    if not ai_url:
        print("[analyze] No AI URL configured, skipping")
        return []

    print(f"[analyze] Calling: {ai_url}")
    resp = requests.post(ai_url, json={
        "secret": secret,
        "transcription": transcription,
        "segments": segments,
        "duration": duration,
        "title": title,
    }, timeout=120)

    if resp.status_code != 200:
        print(f"[analyze] Error {resp.status_code}: {resp.text[:300]}")
        return []

    moments = resp.json().get("moments", [])
    print(f"[analyze] Detected {len(moments)} viral moments")
    return moments


# ---------------------------------------------------------------------------
# Step 4 — Cut clip with FFmpeg (9:16 vertical, 1080×1920)
# ---------------------------------------------------------------------------

def cut_clip_vertical(source_path: str, start: float, end: float, output_path: str):
    duration = end - start
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", source_path,
        "-t", str(duration),
        "-vf", "crop=ih*9/16:ih,scale=1080:1920",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        output_path,
    ]
    print(f"[ffmpeg] {start:.1f}s → {end:.1f}s")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed: {result.stderr[:500]}")


# ---------------------------------------------------------------------------
# Step 5 — Upload to Supabase Storage
# ---------------------------------------------------------------------------

def upload_to_supabase(
    file_path: str,
    bucket: str,
    remote_path: str,
    supabase_url: str,
    supabase_key: str,
) -> str:
    """Upload a file and return its public URL."""
    with open(file_path, "rb") as f:
        file_data = f.read()

    upload_url = f"{supabase_url}/storage/v1/object/{bucket}/{remote_path}"
    headers = {
        "Authorization": f"Bearer {supabase_key}",
        "Content-Type": "video/mp4",
        "x-upsert": "true",
    }

    resp = requests.post(upload_url, headers=headers, data=file_data, timeout=120)
    if resp.status_code not in (200, 201):
        raise RuntimeError(f"Upload failed ({resp.status_code}): {resp.text[:300]}")

    public_url = f"{supabase_url}/storage/v1/object/public/{bucket}/{remote_path}"
    print(f"[upload] OK → {public_url}")
    return public_url


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds via ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    return float(json.loads(result.stdout)["format"]["duration"])


def get_segment_text(segments: list, start: float, end: float) -> str:
    """Extract transcription text for a specific time range."""
    return " ".join(
        seg["text"] for seg in segments
        if seg["end"] > start and seg["start"] < end
    ).strip()


# ---------------------------------------------------------------------------
# Main pipeline (runs in background)
# ---------------------------------------------------------------------------

def process_video_pipeline(req: ProcessRequest):
    tmp_dir = tempfile.mkdtemp(prefix="clipo_")

    try:
        # 1. Download
        send_progress(req.callback_url, req.webhook_secret, req.project_id, 10)
        video_path = download_video(req.url, tmp_dir)

        # 2. Transcribe
        send_progress(req.callback_url, req.webhook_secret, req.project_id, 30)
        whisper_result = transcribe_audio(video_path, req.whisper_config)
        transcription = whisper_result["transcription"]
        segments = whisper_result["segments"]

        # Get video duration
        duration = get_video_duration(video_path)

        # 3. AI analysis (calls ai-analyze-content Edge Function)
        send_progress(req.callback_url, req.webhook_secret, req.project_id, 50)
        moments = analyze_content(
            ai_url=req.ai_analyze_url,
            secret=req.ai_webhook_secret or req.webhook_secret,
            transcription=transcription,
            segments=segments,
            duration=duration,
            title=req.url,
        )

        if not moments:
            send_error(req.callback_url, req.webhook_secret, req.project_id,
                       "IA não detectou momentos virais neste vídeo.")
            return

        # 4. Cut clips + Upload to Supabase Storage
        send_progress(req.callback_url, req.webhook_secret, req.project_id, 65)
        clips: list[dict] = []

        for i, moment in enumerate(moments):
            clip_id = uuid.uuid4().hex[:8]
            clip_filename = f"clip_{clip_id}.mp4"
            clip_path = os.path.join(tmp_dir, clip_filename)

            try:
                cut_clip_vertical(
                    video_path,
                    moment["start_time"],
                    moment["end_time"],
                    clip_path,
                )

                remote_path = f"{req.project_id}/{clip_filename}"
                video_url = upload_to_supabase(
                    clip_path,
                    req.storage_bucket,
                    remote_path,
                    req.supabase_url,
                    req.supabase_key,
                )

                clip_transcription = get_segment_text(
                    segments,
                    moment["start_time"],
                    moment["end_time"],
                )

                # Build clip object matching video-webhook expected schema
                clips.append({
                    "title": moment.get("title", f"Corte {i + 1}"),
                    "description": moment.get("description", ""),
                    "start_time": moment["start_time"],
                    "end_time": moment["end_time"],
                    "duration": moment["end_time"] - moment["start_time"],
                    "video_url": video_url,
                    "transcription": clip_transcription,
                    "classification": moment.get("classification", "educational"),
                    "hashtags": moment.get("hashtags", []),
                    "hook_text": moment.get("hook_text", ""),
                    "cta": moment.get("cta", ""),
                })

                progress = 65 + int((i + 1) / len(moments) * 25)
                send_progress(req.callback_url, req.webhook_secret, req.project_id, progress)

            except Exception as e:
                print(f"[clip] Error on moment {i}: {e}")
                continue

        if not clips:
            send_error(req.callback_url, req.webhook_secret, req.project_id,
                       "Falha ao gerar cortes de vídeo.")
            return

        # 5. Send final results to video-webhook Edge Function
        send_progress(req.callback_url, req.webhook_secret, req.project_id, 95)

        resp = requests.post(req.callback_url, json={
            "project_id": req.project_id,
            "secret": req.webhook_secret,
            "transcription": transcription,
            "segments": segments,
            "clips": clips,
        }, timeout=30)

        if resp.status_code != 200:
            print(f"[callback] Error: {resp.status_code} {resp.text[:300]}")
        else:
            print(f"[callback] Success: {len(clips)} clips delivered")

    except Exception as e:
        print(f"[pipeline] Error:\n{traceback.format_exc()}")
        send_error(req.callback_url, req.webhook_secret, req.project_id, str(e)[:500])

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@app.post("/process")
async def process_video(req: ProcessRequest, background_tasks: BackgroundTasks):
    """Receive job from process-video Edge Function and start background pipeline."""
    print(f"[api] Job received: project={req.project_id} url={req.url}")
    background_tasks.add_task(process_video_pipeline, req)
    return {"status": "accepted", "project_id": req.project_id}


@app.get("/health")
async def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
    
