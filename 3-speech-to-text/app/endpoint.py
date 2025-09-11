import os
import hmac
import json
import hashlib
from typing import Optional

import httpx
from fastapi import UploadFile, File, Form, HTTPException, Request
from fastapi import Depends
from dotenv import load_dotenv

from request import (
    TranscribeByUrlRequest,
    TranscribeByUrlResponse,
    TranscribeByFileResponse,
    TranscriptStatusResponse
)

API_KEY = "" # For testing

# Load environment variables
load_dotenv()
AAI_KEY = API_KEY
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET")  # optional shared secret for webhook auth

if not AAI_KEY:
    raise RuntimeError("ASSEMBLYAI_API_KEY not set")

# ---- HTTP client (reuse across requests)
async def get_client():
    async with httpx.AsyncClient(timeout=60) as client:
        yield client

# ---- Helpers
def webhook_url() -> str:
    if not PUBLIC_BASE_URL:
        # You can still test with polling if no public URL yet
        return ""
    return f"{PUBLIC_BASE_URL}/assemblyai/webhook"

def _headers_json():
    return {
        "Authorization": AAI_KEY,
        "Content-Type": "application/json"
    }

def _headers_upload():
    return {
        "Authorization": AAI_KEY
    }

# ---- Endpoints

async def transcribe_by_url(
    payload: TranscribeByUrlRequest,
    client: httpx.AsyncClient = Depends(get_client)
) -> TranscribeByUrlResponse:
    """
    Submit a transcription job by URL.
    """
    body = {
        "audio_url": str(payload.audio_url),
        # Attach metadata so the webhook can roundtrip it back
        "metadata": {
            "emails": payload.emails or [],
            "user_metadata": payload.metadata or {}
        }
    }

    if webhook_url():
        body["webhook_url"] = webhook_url()

    r = await client.post(
        "https://api.assemblyai.com/v2/transcript",
        headers=_headers_json(),
        json=body
    )
    if r.status_code >= 300:
        raise HTTPException(status_code=502, detail=f"AssemblyAI error: {r.text}")

    data = r.json()
    return TranscribeByUrlResponse(
        id=data["id"],
        status=data["status"],
        audio_url=payload.audio_url
    )


async def transcribe_by_file(
    file: UploadFile = File(...),
    emails: Optional[str] = Form(default=None),  # comma-separated emails
    client: httpx.AsyncClient = Depends(get_client)
) -> TranscribeByFileResponse:
    """
    Upload a local file to AssemblyAI, then request transcription.
    """
    # 1) Upload the file (streaming upload)
    # AssemblyAI supports chunked upload via /v2/upload
    # We'll stream the file to avoid loading into memory.
    upload_url = "https://api.assemblyai.com/v2/upload"

    # Stream file bytes
    async with client.stream("POST", upload_url, headers=_headers_upload(), data=file.file) as resp:
        if resp.status_code >= 300:
            text = await resp.aread()
            raise HTTPException(status_code=502, detail=f"Upload failed: {text.decode()}")
        data = await resp.json()

    audio_url = data["upload_url"]

    # 2) Create transcript job
    body = {
        "audio_url": audio_url,
        "metadata": {
            "emails": [e.strip() for e in (emails or "").split(",") if e.strip()]
        }
    }
    if webhook_url():
        body["webhook_url"] = webhook_url()

    r = await client.post(
        "https://api.assemblyai.com/v2/transcript",
        headers=_headers_json(),
        json=body
    )
    if r.status_code >= 300:
        raise HTTPException(status_code=502, detail=f"AssemblyAI error: {r.text}")

    j = r.json()
    return TranscribeByFileResponse(id=j["id"], status=j["status"])


async def get_transcript_status(
    transcript_id: str,
    client: httpx.AsyncClient = Depends(get_client)
) -> TranscriptStatusResponse:
    r = await client.get(
        f"https://api.assemblyai.com/v2/transcript/{transcript_id}",
        headers={"Authorization": AAI_KEY}
    )
    if r.status_code >= 300:
        raise HTTPException(status_code=502, detail=f"AssemblyAI error: {r.text}")
    data = r.json()
    return TranscriptStatusResponse(
        id=data["id"],
        status=data["status"],
        text=data.get("text")
    )

# ---- Webhook

def _verify_webhook(raw_body: bytes, signature: Optional[str]) -> bool:
    """
    Optional: if you set WEBHOOK_SECRET, verify HMAC-SHA256 signature sent in header
    X-Aai-Signature: hex(hmac_sha256(body, WEBHOOK_SECRET))
    """
    if not WEBHOOK_SECRET:
        return True  # no verification configured
    if not signature:
        return False
    expected = hmac.new(WEBHOOK_SECRET.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)

async def assemblyai_webhook(request: Request):
    """
    Receives callbacks from AssemblyAI when transcripts are updated/completed.
    """
    raw = await request.body()
    signature = request.headers.get("X-Aai-Signature")

    if not _verify_webhook(raw, signature):
        raise HTTPException(status_code=401, detail="Invalid webhook signature")

    try:
        data = json.loads(raw.decode("utf-8"))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    # Expected payload includes: id, status, text (when completed), metadata, etc.
    status = data.get("status")
    transcript_id = data.get("id")
    text = data.get("text", "")
    meta = data.get("metadata") or {}

    # For now, just return. In the next step we'll send to Gmail here.
    # (You could also persist to DB.)
    # Example: emails = meta.get("emails", [])

    return {"ok": True, "id": transcript_id, "status": status, "length": len(text)}
