"""FastAPI application — Cymatic Seal web service.

No payments. Built for platform integration (Spotify, YouTube Music, SoundCloud).
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from api.database import (
    get_seal_history,
    hash_identifier,
    init_db,
    record_seal_job,
)

logger = logging.getLogger(__name__)

UPLOAD_MAX_MB = 100
UPLOAD_MAX_BYTES = UPLOAD_MAX_MB * 1024 * 1024

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
DOWNLOAD_DIR = BASE_DIR / "data" / "downloads"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Cymatic Seal",
    description="Adversarial audio protection for creators. Built for platform integration.",
    version="0.3.0",
)

app.mount(
    "/static",
    StaticFiles(directory=str(FRONTEND_DIR / "static")),
    name="static",
)
templates = Jinja2Templates(directory=str(FRONTEND_DIR / "templates"))

WORK_DIR = Path(tempfile.gettempdir()) / "cymatic_seal_work"
WORK_DIR.mkdir(exist_ok=True)


@app.on_event("startup")
async def startup():
    init_db()


def _save_upload(upload: UploadFile, dest: Path) -> None:
    with open(dest, "wb") as f:
        shutil.copyfileobj(upload.file, f)


def _get_identifier(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        ip = forwarded.split(",")[0].strip()
    else:
        ip = request.client.host if request.client else "unknown"
    return hash_identifier(ip)


# ── pages ─────────────────────────────────────────────────────────

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/verify")
async def verify_page(request: Request):
    return templates.TemplateResponse("verify.html", {"request": request})


@app.get("/history")
async def history_page(request: Request):
    return templates.TemplateResponse("history.html", {"request": request})


@app.get("/batch")
async def batch_page(request: Request):
    return templates.TemplateResponse("batch.html", {"request": request})


@app.get("/for-platforms")
async def for_platforms_page(request: Request):
    return templates.TemplateResponse("for-platforms.html", {"request": request})


# ── seal API ──────────────────────────────────────────────────────

@app.post("/api/seal")
async def api_seal(
    request: Request,
    file: UploadFile = File(...),
    artist: str = Form(""),
    title: str = Form(""),
    method: str = Form("fgsm"),
    steps: int = Form(3),
    epsilon: float = Form(0.008),
    margin_db: float = Form(-4.0),
    lowpass_cutoff_hz: float = Form(6000.0),
    device: str = Form("auto"),
    model: str = Form("htdemucs"),
    pro: str = Form("false"),
):
    """Seal an uploaded audio file. No credits; all options configurable."""
    is_pro = pro.lower() in ("true", "1", "on", "yes")

    if file.size is not None and file.size > UPLOAD_MAX_BYTES:
        raise HTTPException(413, f"File too large (max {UPLOAD_MAX_MB} MB).")

    if is_pro:
        steps = max(steps, 8)
        epsilon = min(epsilon, 0.01)

    job_id = uuid.uuid4().hex[:12]
    job_dir = DOWNLOAD_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    ext = Path(file.filename or "track.wav").suffix or ".wav"
    input_path = job_dir / f"input{ext}"
    output_path = job_dir / f"sealed{ext}"
    cert_path = job_dir / "certificate.json"

    try:
        _save_upload(file, input_path)

        from cymatic_seal.seal import seal_audio

        _, cert = seal_audio(
            input_path,
            output_path=output_path,
            certificate_path=cert_path,
            artist=artist,
            title=title,
            method=method,
            steps=steps,
            epsilon=epsilon,
            margin_db=margin_db,
            lowpass_cutoff_hz=lowpass_cutoff_hz,
            device=device,
            model_name=model,
        )

        identifier = _get_identifier(request)
        record_seal_job(
            job_id=job_id,
            identifier=identifier,
            original_filename=file.filename or "track",
            duration_seconds=cert.duration_seconds,
        )

        return JSONResponse(
            {
                "job_id": job_id,
                "certificate": cert.to_json(),
                "download_audio": f"/api/download/{job_id}/audio",
                "download_certificate": f"/api/download/{job_id}/certificate",
            }
        )
    except Exception:
        logger.exception("Seal failed for job %s", job_id)
        raise HTTPException(500, "Seal processing failed. Check server logs.")


# ── batch seal API ────────────────────────────────────────────────

@app.post("/api/seal/batch")
async def api_seal_batch(
    request: Request,
    files: List[UploadFile] = File(...),
    artist: str = Form(""),
    title: str = Form(""),
    method: str = Form("fgsm"),
    steps: int = Form(3),
    epsilon: float = Form(0.008),
    margin_db: float = Form(-4.0),
    lowpass_cutoff_hz: float = Form(6000.0),
    device: str = Form("auto"),
    model: str = Form("htdemucs"),
):
    """Seal multiple audio files. No credits."""
    identifier = _get_identifier(request)

    results = []
    for f in files:
        if f.size is not None and f.size > UPLOAD_MAX_BYTES:
            results.append({"filename": f.filename, "error": "File too large", "job_id": None})
            continue

        job_id = uuid.uuid4().hex[:12]
        job_dir = DOWNLOAD_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        ext = Path(f.filename or "track.wav").suffix or ".wav"
        input_path = job_dir / f"input{ext}"
        output_path = job_dir / f"sealed{ext}"
        cert_path = job_dir / "certificate.json"

        try:
            _save_upload(f, input_path)

            from cymatic_seal.seal import seal_audio

            _, cert = seal_audio(
                input_path,
                output_path=output_path,
                certificate_path=cert_path,
                artist=artist,
                title=title,
                method=method,
                steps=steps,
                epsilon=epsilon,
                margin_db=margin_db,
                lowpass_cutoff_hz=lowpass_cutoff_hz,
                device=device,
                model_name=model,
            )

            record_seal_job(
                job_id=job_id,
                identifier=identifier,
                original_filename=f.filename or "track",
                duration_seconds=cert.duration_seconds,
            )

            results.append({
                "filename": f.filename,
                "job_id": job_id,
                "download_audio": f"/api/download/{job_id}/audio",
                "download_certificate": f"/api/download/{job_id}/certificate",
                "error": None,
            })
        except Exception:
            logger.exception("Batch seal failed for file %s", f.filename)
            results.append({"filename": f.filename, "error": "Processing failed", "job_id": None})

    return JSONResponse({"results": results})


# ── download ──────────────────────────────────────────────────────

@app.get("/api/download/{job_id}/audio")
async def download_audio(job_id: str):
    for base in [DOWNLOAD_DIR, WORK_DIR]:
        job_dir = base / job_id
        candidates = list(job_dir.glob("sealed.*"))
        if candidates:
            return FileResponse(
                candidates[0],
                media_type="application/octet-stream",
                filename=candidates[0].name,
            )
    raise HTTPException(404, "Sealed audio not found.")


@app.get("/api/download/{job_id}/certificate")
async def download_certificate(job_id: str):
    for base in [DOWNLOAD_DIR, WORK_DIR]:
        cert_path = base / job_id / "certificate.json"
        if cert_path.exists():
            return FileResponse(
                cert_path,
                media_type="application/json",
                filename="cymatic_seal_certificate.json",
            )
    raise HTTPException(404, "Certificate not found.")


# ── verify ────────────────────────────────────────────────────────

@app.post("/api/verify")
async def api_verify(
    file: UploadFile = File(...),
    certificate: UploadFile = File(...),
):
    """Verify that an audio file matches its seal certificate."""
    job_id = uuid.uuid4().hex[:12]
    job_dir = WORK_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    ext = Path(file.filename or "track.wav").suffix or ".wav"
    audio_path = job_dir / f"audio{ext}"
    cert_path = job_dir / "certificate.json"

    try:
        _save_upload(file, audio_path)
        _save_upload(certificate, cert_path)

        from cymatic_seal.verify import verify_seal

        result = verify_seal(audio_path, cert_path)

        resp: dict = {
            "verified": result.verified,
            "reason": result.reason,
        }
        if result.certificate:
            resp["certificate"] = {
                "algorithm": result.certificate.algorithm,
                "timestamp": result.certificate.timestamp,
                "artist": result.certificate.artist,
                "title": result.certificate.title,
                "duration_seconds": result.certificate.duration_seconds,
            }
        return JSONResponse(resp)
    except Exception:
        logger.exception("Verify failed for job %s", job_id)
        raise HTTPException(500, "Verification failed. Check server logs.")


# ── history ───────────────────────────────────────────────────────

@app.get("/api/history")
async def api_history(request: Request):
    identifier = _get_identifier(request)
    jobs = get_seal_history(identifier)
    return JSONResponse({"jobs": jobs})
