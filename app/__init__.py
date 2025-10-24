"""
AudioMind - AI-Powered Audio Knowledge Intelligence
===================================================

Transform audio content into actionable insights using AI.

Main Components:
- Audio Transcription (Whisper)
- Topic Modeling (LDA + BERTopic)
- LLM Synthesis (GPT-4)
- RAG Search (ChromaDB)

Quick Start:
    >>> from app.processors import WhisperProcessor
    >>> processor = WhisperProcessor()
    >>> result = processor.transcribe("audio.mp3")

Documentation: https://github.com/yourusername/audiomind
"""

__version__ = "1.0.0"
__author__ = "Julio César García Escoto"
__email__ = "your.email@example.com"

# Core imports for easy access
from app.config import settings

__all__ = ["settings", "__version__"]
