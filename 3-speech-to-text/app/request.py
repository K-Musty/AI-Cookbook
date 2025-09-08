from pydantic import BaseModel, HttpUrl, Field
from typing import List, Optional

class TranscribeByUrlRequest(BaseModel):
    audio_url: HttpUrl
    emails: Optional[List[str]] = Field(
        default=None,
        description="Optional: who to email later with the transcript"
    )
    metadata: Optional[dict] = None  # anything you want to roundtrip back

class TranscribeByFileResponse(BaseModel):
    id: str
    status: str

class TranscribeByUrlResponse(BaseModel):
    id: str
    status: str
    audio_url: HttpUrl

class TranscriptStatusResponse(BaseModel):
    id: str
    status: str
    text: Optional[str] = None
