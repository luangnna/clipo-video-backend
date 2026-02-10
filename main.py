"""
CLIPO - Backend de Processamento de Vídeo (Worker)
Pipeline: yt-dlp → Whisper → IA → FFmpeg → Supabase → Webhook
"""

import os
import json
import uuid
import subprocess
import tempfile
import shutil
import traceback
from typing import Optional
from asyncio import Semaphore

import requests
import whisper
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel

# ========================
# CONFIG
# ========================

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
WHISPER_LANG = os.getenv("WHISPER_LANG", "pt")
YT_DLP_TIMEOUT = int(os.getenv("YT_DLP_TIMEOUT", "480"))
FFMPEG_TIMEOUT = int(os.getenv("FFMPEG_TIMEOUT", "240"))

PIPELINE_SEMAPHORE = Semaphore(1)
_whisper_model = None

app = FastAPI(title="CLIPO Video Processor")

# ========================
# MODELS
# ========================

class ProcessRequest(BaseModel):
    url: str
    project_id: str
    callback_url: str
    webhook_secret: str
    supabase_url: str
    supabase_key: str
    storage_bucket: str = "videos"
    ai_analyze_url: Optional[str] = None
    ai_webhook_secret: Optional[str] = None
    whisper_config: Optional[dict] = None

# ========================
# HELPERS
# ========================

def get_whisper_model(model_size: Optional[str] = None):
    global _whisper_model
    model_size = model_size or WHISPER_MODEL
    if _whisper_model is None:
        print(f"[whisper] Loading model: {model_size}")
        _whisper_model = whisper.load_model(model_size)
    return _whisper_model

def send_callback(url, payload):
    try:
        requests.post(url, json=payload, timeout=15)
    except Exception as e:
        print(f"[callback] Failed: {e}")

def download_video(url: str, output_dir: str) -> str:
    output_path = os.path.join(output_dir, "source.mp4")
    cmd = [
        "yt-dlp",
        "-f", "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best",
        "--merge-output-format", "mp4",
        "-o", output_path,
        "--no-playlist",
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=YT_DLP_TIMEOUT)
    if result.returncode != 0:
        raise RuntimeError(result.stderr[:400])
    return output_path

def transcribe(video_path: str, config: dict | None):
    config = config or {}
    model = get_whisper_model(config.get("model_size"))
    result = model.transcribe(
        video_path,
        language=config.get("language", WHISPER_LANG),
        verbose=False,
    )
    segments = [
        {
            "start": round(s["start"], 2),
            "end": round(s["end"], 2),
            "text": s["text"].strip(),
        }
        for s in result.get("segments", [])
        if s["text"].strip()
    ]
    return result.get("text", ""), segments

def cut_clip(source, start, end, out):
    duration = end - start
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", source,
        "-t", str(duration),
        "-vf", "crop=ih*9/16:ih,scale=1080:1920",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-c:a", "aac",
        "-movflags", "+faststart",
        out,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=FFMPEG_TIMEOUT)
    if result.returncode != 0:
        raise RuntimeError(result.stderr[:400])

# ========================
# PIPELINE
# ========================

def run_pipeline(req: ProcessRequest):
    tmp = tempfile.mkdtemp(prefix="clipo_")
    try:
        send_callback(req.callback_url, {
            "project_id": req.project_id,
            "secret": req.webhook_secret,
            "progress": 10,
        })

        video = download_video(req.url, tmp)

        text, segments = transcribe(video, req.whisper_config)

        send_callback(req.callback_url, {
            "project_id": req.project_id,
            "secret": req.webhook_secret,
            "progress": 50,
            "transcription": text,
            "segments": segments,
        })

    except Exception as e:
        print(traceback.format_exc())
        send_callback(req.callback_url, {
            "project_id": req.project_id,
            "secret": req.webhook_secret,
            "error": str(e)[:300],
        })
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

# ========================
# API
# ========================

@app.post("/process")
async def process(req: ProcessRequest, bg: BackgroundTasks):
    async with PIPELINE_SEMAPHORE:
        bg.add_task(run_pipeline, req)
    return {"status": "accepted", "project_id": req.project_id}

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
