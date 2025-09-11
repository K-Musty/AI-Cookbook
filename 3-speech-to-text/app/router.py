from fastapi import APIRouter
from endpoint import (
    transcribe_by_url,
    transcribe_by_file,
    get_transcript_status,
    assemblyai_webhook
)

router = APIRouter()

# Public endpoints
router.post("/transcribe/url")(transcribe_by_url)
router.post("/transcribe/file")(transcribe_by_file)
router.get("/transcripts/{transcript_id}")(get_transcript_status)

# Webhook endpoint (called by AssemblyAI)
router.post("/assemblyai/webhook")(assemblyai_webhook)
