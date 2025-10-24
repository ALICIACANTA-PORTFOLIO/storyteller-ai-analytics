"""
Utility Functions for AudioMind.

Shared utilities across the application:
- audio.py: Audio file validation and preprocessing
- text.py: Text processing utilities
- logging.py: Logging configuration
- metrics.py: Performance metrics
- validators.py: Input validation
- helpers.py: General helper functions

Usage:
    >>> from app.utils.audio import validate_audio_file
    >>> from app.utils.text import clean_text
    >>> from app.utils.metrics import track_duration
"""

__all__ = [
    "validate_audio_file",
    "clean_text",
    "setup_logging",
    "track_duration",
    "validate_input",
]
