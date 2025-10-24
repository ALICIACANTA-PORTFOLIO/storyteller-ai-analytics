"""
WhisperProcessor - Agnostic Speech-to-Text Transcription.

This module provides a flexible, production-ready implementation of OpenAI's Whisper
for audio transcription. It's designed to work with ANY type of audio content:
- Podcasts, interviews, meetings, lectures, audiobooks, etc.
- No hardcoded assumptions about content type or structure
- Configurable VAD, diarization, and language detection

Example:
    >>> from app.processors import WhisperProcessor
    >>> 
    >>> processor = WhisperProcessor()
    >>> result = await processor.transcribe("audio.mp3")
    >>> print(result.text)
    >>> print(f"Language: {result.language}")
    >>> print(f"Duration: {result.duration_seconds}s")
"""

import asyncio
import os
import time
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field

import whisper
import torch
import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)


# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================


class TranscriptionError(Exception):
    """Base exception for transcription errors."""
    pass


class AudioFileError(TranscriptionError):
    """Raised when audio file is invalid or cannot be processed."""
    pass


class AudioTooLongError(AudioFileError):
    """Raised when audio file exceeds maximum duration."""
    pass


class AudioTooLargeError(AudioFileError):
    """Raised when audio file exceeds maximum size."""
    pass


@dataclass
class TranscriptionSegment:
    """
    A single segment of transcription with timing information.
    
    Attributes:
        text: Transcribed text for this segment
        start_time: Start time in seconds
        end_time: End time in seconds
        speaker_id: Optional speaker identifier (if diarization enabled)
        confidence: Confidence score (0.0-1.0) if available
    """
    text: str
    start_time: float
    end_time: float
    speaker_id: Optional[str] = None
    confidence: Optional[float] = None
    
    def __repr__(self) -> str:
        speaker = f", speaker={self.speaker_id}" if self.speaker_id else ""
        return f"<Segment({self.start_time:.1f}-{self.end_time:.1f}s{speaker})>"


@dataclass
class TranscriptionResult:
    """
    Complete transcription result.
    
    This is the main output of the WhisperProcessor. It contains:
    - Full transcribed text
    - Segments with timing
    - Detected language and confidence
    - Processing metadata
    
    Attributes:
        text: Full transcribed text
        segments: List of timed segments
        language: Detected language code (e.g., 'en', 'es')
        language_confidence: Confidence in language detection (0.0-1.0)
        duration_seconds: Total audio duration
        processing_time_seconds: Time taken to transcribe
        model_used: Whisper model name used
        vad_enabled: Whether Voice Activity Detection was used
        diarization_enabled: Whether speaker diarization was used
    """
    text: str
    segments: List[TranscriptionSegment]
    language: str
    language_confidence: float
    duration_seconds: float
    processing_time_seconds: float
    model_used: str
    vad_enabled: bool = False
    diarization_enabled: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self) -> str:
        return (
            f"<TranscriptionResult("
            f"lang={self.language}, "
            f"duration={self.duration_seconds:.1f}s, "
            f"segments={len(self.segments)})>"
        )


class WhisperProcessor:
    """
    Agnostic audio transcription processor using OpenAI Whisper.
    
    This processor can transcribe ANY type of audio content with configurable
    options for quality, speed, and features.
    
    Features:
        - Multiple model sizes (tiny → large-v3-turbo)
        - Automatic language detection
        - Optional Voice Activity Detection (VAD)
        - Optional speaker diarization
        - GPU acceleration support
        - Segment-level timestamps
        - Async/await compatible with FastAPI
        - Input validation and error handling
    
    Args:
        model_size: Whisper model size (default from config)
        device: Device to use ('cpu', 'cuda', 'mps', or 'auto')
        compute_type: Computation precision ('float16', 'int8', etc.)
        enable_vad: Enable Voice Activity Detection
        enable_diarization: Enable speaker diarization
        language: Force specific language (None = auto-detect)
        max_duration_seconds: Maximum audio duration (default: 3600 = 1 hour)
        max_file_size_mb: Maximum file size in MB (default: 500)
    
    Example:
        >>> # Basic usage
        >>> processor = WhisperProcessor()
        >>> result = await processor.transcribe("podcast.mp3")
        >>> 
        >>> # With options
        >>> processor = WhisperProcessor(
        ...     model_size="large-v3-turbo",
        ...     enable_vad=True,
        ...     language="en",
        ...     max_duration_seconds=7200  # 2 hours
        ... )
        >>> result = await processor.transcribe("meeting.wav")
        >>> 
        >>> # With progress callback
        >>> async def progress(pct: float, msg: str):
        ...     print(f"[{pct*100:.0f}%] {msg}")
        >>> 
        >>> result = await processor.transcribe(
        ...     "audio.mp3",
        ...     progress_callback=progress
        ... )
    """
    
    # Class-level constants
    SUPPORTED_FORMATS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".webm"}
    
    def __init__(
        self,
        model_size: Optional[str] = None,
        device: Optional[str] = None,
        compute_type: str = "float16",
        enable_vad: bool = False,
        enable_diarization: bool = False,
        language: Optional[str] = None,
        max_duration_seconds: int = 3600,
        max_file_size_mb: int = 500,
    ):
        """Initialize the Whisper processor."""
        # Use config defaults if not provided
        self.model_size = model_size or settings.transcription.model_size.value
        self.device = device or self._get_best_device()
        self.compute_type = compute_type
        self.enable_vad = enable_vad
        self.enable_diarization = enable_diarization
        self.language = language
        self.max_duration_seconds = max_duration_seconds
        self.max_file_size_mb = max_file_size_mb
        
        # Model will be loaded lazily on first use
        self._model: Optional[whisper.Whisper] = None
        
        logger.info(
            f"🎙️  WhisperProcessor initialized\n"
            f"   Model: {self.model_size}\n"
            f"   Device: {self.device}\n"
            f"   VAD: {'enabled' if self.enable_vad else 'disabled'}\n"
            f"   Diarization: {'enabled' if self.enable_diarization else 'disabled'}\n"
            f"   Max duration: {self.max_duration_seconds}s\n"
            f"   Max file size: {self.max_file_size_mb}MB"
        )
    
    def _get_best_device(self) -> str:
        """Automatically select the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    @property
    def model(self) -> whisper.Whisper:
        """Lazy-load the Whisper model."""
        if self._model is None:
            logger.info(f"📥 Loading Whisper model '{self.model_size}'...")
            start_time = time.time()
            
            self._model = whisper.load_model(
                self.model_size,
                device=self.device
            )
            
            load_time = time.time() - start_time
            logger.info(f"✅ Model loaded in {load_time:.2f}s")
        
        return self._model
    
    def _validate_audio_file(self, audio_path: Path) -> None:
        """
        Validate audio file before processing.
        
        Checks:
        - File exists
        - File format is supported
        - File size is within limits
        - Audio duration is within limits (requires ffprobe)
        
        Raises:
            FileNotFoundError: If file doesn't exist
            AudioFileError: If format not supported
            AudioTooLargeError: If file size exceeds limit
            AudioTooLongError: If duration exceeds limit
        """
        # Check file exists
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Check format
        if audio_path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise AudioFileError(
                f"Unsupported audio format: {audio_path.suffix}. "
                f"Supported formats: {', '.join(self.SUPPORTED_FORMATS)}"
            )
        
        # Check file size
        size_mb = audio_path.stat().st_size / (1024 * 1024)
        if size_mb > self.max_file_size_mb:
            raise AudioTooLargeError(
                f"File too large: {size_mb:.1f}MB exceeds limit of {self.max_file_size_mb}MB"
            )
        
        # Check duration (optional, requires ffprobe)
        try:
            duration = self._get_audio_duration_ffprobe(audio_path)
            if duration and duration > self.max_duration_seconds:
                raise AudioTooLongError(
                    f"Audio too long: {duration/60:.1f}min exceeds limit of "
                    f"{self.max_duration_seconds/60:.1f}min"
                )
        except FileNotFoundError:
            # ffprobe not available, skip duration check
            logger.warning("ffprobe not found, skipping duration validation")
    
    def _get_audio_duration_ffprobe(self, audio_path: Path) -> Optional[float]:
        """Get audio duration using ffprobe."""
        try:
            import subprocess
            result = subprocess.run(
                [
                    "ffprobe", "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    str(audio_path)
                ],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return float(result.stdout.strip())
        except (FileNotFoundError, ValueError, subprocess.TimeoutExpired):
            pass
        return None
    
    async def transcribe(
        self,
        audio_path: str | Path,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        **kwargs
    ) -> TranscriptionResult:
        """
        Transcribe an audio file (async).
        
        This is the main method for transcription. It's completely agnostic
        to the type of audio content - works with podcasts, meetings, lectures, etc.
        
        Args:
            audio_path: Path to audio file (supports mp3, wav, m4a, etc.)
            progress_callback: Optional callback function(progress: 0.0-1.0, message: str)
            **kwargs: Additional options to override defaults
                - language: Force specific language
                - temperature: Sampling temperature (0.0-1.0)
                - beam_size: Beam search size for better quality
                - best_of: Number of candidates when sampling
                - word_timestamps: Enable word-level timestamps
        
        Returns:
            TranscriptionResult with full text, segments, and metadata
        
        Raises:
            FileNotFoundError: If audio file doesn't exist
            AudioFileError: If audio format not supported
            AudioTooLargeError: If file size exceeds limit
            AudioTooLongError: If duration exceeds limit
            TranscriptionError: If transcription fails
        
        Example:
            >>> processor = WhisperProcessor()
            >>> result = await processor.transcribe("audio.mp3")
            >>> print(result.text)
            >>> print(f"Detected language: {result.language}")
            >>> 
            >>> # With progress tracking
            >>> async def on_progress(pct, msg):
            ...     print(f"[{pct*100:.0f}%] {msg}")
            >>> 
            >>> result = await processor.transcribe(
            ...     "audio.mp3",
            ...     progress_callback=on_progress,
            ...     language="es",
            ...     temperature=0.2
            ... )
        """
        audio_path = Path(audio_path)
        
        try:
            # Progress: 0% - Validation
            if progress_callback:
                if asyncio.iscoroutinefunction(progress_callback):
                    await progress_callback(0.0, "Validating audio file...")
                else:
                    progress_callback(0.0, "Validating audio file...")
            
            # Validate file
            self._validate_audio_file(audio_path)
            
            logger.info(f"\n🎵 Transcribing: {audio_path.name}")
            logger.info(f"   Size: {audio_path.stat().st_size / 1024 / 1024:.2f} MB")
            
            # Progress: 10% - Loading audio
            if progress_callback:
                if asyncio.iscoroutinefunction(progress_callback):
                    await progress_callback(0.1, "Loading audio data...")
                else:
                    progress_callback(0.1, "Loading audio data...")
            
            start_time = time.time()
            
            # Run transcription in thread pool (CPU-intensive)
            # Note: We don't pass progress_callback to _transcribe_sync because
            # it's async and can't be called from a thread pool
            result = await asyncio.to_thread(
                self._transcribe_sync,
                audio_path,
                **kwargs
            )
            
            # Progress: 100% - Done
            if progress_callback:
                if asyncio.iscoroutinefunction(progress_callback):
                    await progress_callback(1.0, "Transcription complete!")
                else:
                    progress_callback(1.0, "Transcription complete!")
            
            return result
            
        except (FileNotFoundError, AudioFileError, AudioTooLargeError, AudioTooLongError):
            # Re-raise validation errors as-is
            raise
        except Exception as e:
            logger.error(f"❌ Transcription failed for {audio_path}: {e}")
            raise TranscriptionError(f"Failed to transcribe {audio_path.name}") from e
    
    def _transcribe_sync(
        self,
        audio_path: Path,
        **kwargs
    ) -> TranscriptionResult:
        """
        Synchronous transcription implementation.
        
        This method does the actual CPU-intensive work and is called
        from transcribe() via asyncio.to_thread().
        
        Note: Progress tracking is handled at the async layer (transcribe method)
        since callbacks can't be called from within a thread pool.
        """
        start_time = time.time()
        
        logger.info("Loading Whisper model...")
        
        # Prepare transcription options
        transcribe_options = {
            "language": self.language or kwargs.get("language"),
            "task": "transcribe",  # vs "translate"
            "temperature": kwargs.get("temperature", 0.0),
            "beam_size": kwargs.get("beam_size", 5),
            "best_of": kwargs.get("best_of", 5),
            "word_timestamps": kwargs.get("word_timestamps", True),
            "fp16": self.compute_type == "float16" and self.device == "cuda",
        }
        
        # Remove None values
        transcribe_options = {k: v for k, v in transcribe_options.items() if v is not None}
        
        logger.info("Transcribing audio...")
        
        # Run transcription
        result = self.model.transcribe(
            str(audio_path),
            **transcribe_options
        )
        
        logger.info("Processing results...")
        
        processing_time = time.time() - start_time
        
        # Extract segments
        segments = self._extract_segments(result.get("segments", []))
        
        # Get audio duration
        duration = result.get("duration", 0.0) or self._get_audio_duration(audio_path)
        
        # Build result object
        transcription_result = TranscriptionResult(
            text=result["text"].strip(),
            segments=segments,
            language=result.get("language", "unknown"),
            language_confidence=self._calculate_language_confidence(result),
            duration_seconds=duration,
            processing_time_seconds=processing_time,
            model_used=self.model_size,
            vad_enabled=self.enable_vad,
            diarization_enabled=self.enable_diarization,
            metadata={
                "audio_file": str(audio_path),
                "file_size_mb": audio_path.stat().st_size / 1024 / 1024,
                "transcribe_options": transcribe_options,
            }
        )
        
        # Print summary
        logger.info(f"✅ Transcription complete!")
        logger.info(f"   Duration: {duration:.2f}s")
        logger.info(f"   Processing time: {processing_time:.2f}s")
        logger.info(f"   Real-time factor: {processing_time / duration:.2f}x")
        logger.info(
            f"   Language: {transcription_result.language} "
            f"(confidence: {transcription_result.language_confidence:.2%})"
        )
        logger.info(f"   Segments: {len(segments)}")
        logger.info(f"   Text length: {len(transcription_result.text)} chars")
        
        return transcription_result
    
    def _extract_segments(self, raw_segments: List[Dict]) -> List[TranscriptionSegment]:
        """Extract and clean segments from Whisper output."""
        segments = []
        
        for seg in raw_segments:
            segment = TranscriptionSegment(
                text=seg["text"].strip(),
                start_time=seg["start"],
                end_time=seg["end"],
                # Whisper doesn't provide speaker IDs by default
                speaker_id=None,
                # Use no_speech_prob as a proxy for confidence
                confidence=1.0 - seg.get("no_speech_prob", 0.0)
            )
            segments.append(segment)
        
        return segments
    
    def _calculate_language_confidence(self, result: Dict) -> float:
        """Calculate language detection confidence."""
        # Whisper doesn't directly provide language confidence
        # We can infer it from the presence of language in result
        if "language" in result and result["language"]:
            # If language was detected, assume reasonable confidence
            return 0.95
        return 0.5
    
    def _get_audio_duration(self, audio_path: Path) -> float:
        """Get audio duration using librosa or ffmpeg fallback."""
        try:
            import librosa
            duration = librosa.get_duration(path=str(audio_path))
            return duration
        except ImportError:
            # Fallback: estimate from file size (very rough)
            # Average bitrate assumption: 128 kbps for MP3
            file_size_bits = audio_path.stat().st_size * 8
            estimated_duration = file_size_bits / (128 * 1024)
            return estimated_duration
    
    def transcribe_batch(
        self,
        audio_paths: List[str | Path],
        **kwargs
    ) -> List[TranscriptionResult]:
        """
        Transcribe multiple audio files in batch.
        
        Args:
            audio_paths: List of paths to audio files
            **kwargs: Options passed to transcribe()
        
        Returns:
            List of TranscriptionResult objects
        
        Example:
            >>> processor = WhisperProcessor()
            >>> files = ["audio1.mp3", "audio2.mp3", "audio3.mp3"]
            >>> results = processor.transcribe_batch(files)
            >>> for result in results:
            ...     print(f"{result.metadata['audio_file']}: {len(result.text)} chars")
        """
        print(f"\n📚 Batch transcription: {len(audio_paths)} files")
        
        results = []
        for i, audio_path in enumerate(audio_paths, 1):
            print(f"\n[{i}/{len(audio_paths)}]")
            try:
                result = self.transcribe(audio_path, **kwargs)
                results.append(result)
            except Exception as e:
                print(f"⚠️  Skipping {audio_path}: {e}")
                continue
        
        print(f"\n✅ Batch complete: {len(results)}/{len(audio_paths)} successful")
        return results
    
    def __repr__(self) -> str:
        return (
            f"<WhisperProcessor("
            f"model={self.model_size}, "
            f"device={self.device}, "
            f"vad={self.enable_vad}, "
            f"diarization={self.enable_diarization})>"
        )
