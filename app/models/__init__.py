"""
Data models package.

This package contains SQLAlchemy ORM models for the application.
All models follow an agnostic design pattern - they are not tied to
specific use cases (like podcasts or interviews) and can be used for
any type of audio analysis.
"""

from app.models.database import (
    Base,
    AudioFile,
    Transcription,
    TranscriptionSegment,
    Topic,
    Analysis,
    Embedding,
    AudioStatus,
    TranscriptionStatus,
    AnalysisStatus,
)

__all__ = [
    "Base",
    "AudioFile",
    "Transcription",
    "TranscriptionSegment",
    "Topic",
    "Analysis",
    "Embedding",
    "AudioStatus",
    "TranscriptionStatus",
    "AnalysisStatus",
]
