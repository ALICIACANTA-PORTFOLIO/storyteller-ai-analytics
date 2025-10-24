"""
SQLAlchemy database models with agnostic design.

These models are designed to work with ANY type of audio content:
- Podcasts, interviews, meetings, lectures, audiobooks, etc.
- No hardcoded assumptions about content type
- Flexible metadata storage using JSONB
- Optimized with strategic indexes
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from uuid import uuid4

from sqlalchemy import (
    Column,
    String,
    Integer,
    Float,
    DateTime,
    Boolean,
    ForeignKey,
    Text,
    Index,
    CheckConstraint,
    Enum as SQLEnum,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.orm import DeclarativeBase, relationship, Mapped, mapped_column
from sqlalchemy.sql import func
import enum


# ============================================================================
# BASE
# ============================================================================


class Base(DeclarativeBase):
    """Base class for all models."""

    pass


# ============================================================================
# ENUMS
# ============================================================================


class AudioStatus(enum.Enum):
    """Status of audio file processing."""

    UPLOADED = "uploaded"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    ARCHIVED = "archived"
    
    def __str__(self):
        return self.value


class TranscriptionStatus(str, enum.Enum):
    """Status of transcription process."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class AnalysisStatus(str, enum.Enum):
    """Status of analysis process."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


# ============================================================================
# MODELS
# ============================================================================


class AudioFile(Base):
    """
    Generic audio file metadata.
    
    Agnostic design - works with ANY audio type:
    - Podcasts, interviews, meetings, lectures, etc.
    - No assumptions about content structure
    - Flexible metadata in JSONB column
    
    Examples:
        # Podcast episode
        metadata = {
            "show_name": "Tech Talk",
            "episode_number": 42,
            "guests": ["John Doe"]
        }
        
        # Interview
        metadata = {
            "interviewer": "Jane Smith",
            "interviewee": "Bob Johnson",
            "topic": "AI Research"
        }
        
        # Meeting
        metadata = {
            "meeting_type": "standup",
            "participants": ["Alice", "Bob", "Carol"],
            "project": "Project X"
        }
    """

    __tablename__ = "audio_files"
    __table_args__ = (
        Index("idx_audio_status", "status"),
        Index("idx_audio_created", "created_at"),
        Index("idx_audio_duration", "duration_seconds"),
        Index("idx_audio_metadata_gin", "metadata", postgresql_using="gin"),
        {"schema": "audiomind"},
    )

    # Primary Key
    id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
        comment="Unique identifier for the audio file",
    )

    # File Information
    filename: Mapped[str] = mapped_column(
        String(255), nullable=False, comment="Original filename"
    )
    file_path: Mapped[str] = mapped_column(
        String(512), nullable=False, unique=True, comment="Storage path or URL"
    )
    file_size_bytes: Mapped[int] = mapped_column(
        Integer, nullable=False, comment="File size in bytes"
    )
    mime_type: Mapped[str] = mapped_column(
        String(100), nullable=False, comment="MIME type (audio/mpeg, audio/wav, etc.)"
    )

    # Audio Properties
    duration_seconds: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, comment="Duration in seconds"
    )
    sample_rate: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True, comment="Sample rate in Hz"
    )
    channels: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True, comment="Number of audio channels"
    )
    bitrate: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True, comment="Bitrate in bps"
    )

    # Processing Status
    status: Mapped[AudioStatus] = mapped_column(
        SQLEnum(
            AudioStatus, 
            name="audio_status", 
            schema="audiomind", 
            native_enum=True, 
            create_constraint=False,
            values_callable=lambda x: [e.value for e in x]  # Use .value (lowercase)
        ),
        nullable=False,
        default=AudioStatus.UPLOADED,
        server_default="uploaded",  # Default value in database
        comment="Current processing status",
    )

    # Flexible Metadata (JSONB for any custom fields)
    custom_metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSONB,
        nullable=True,
        comment="Flexible metadata - podcast info, meeting details, etc.",
        name="metadata",  # Column name in DB is still "metadata"
    )

    # User/Source Information
    uploaded_by: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True, comment="User or system that uploaded"
    )
    source: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True, comment="Source of the audio (upload, API, etc.)"
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        comment="When record was created",
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
        comment="When record was last updated",
    )
    deleted_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Soft delete timestamp",
    )

    # Relationships
    transcription: Mapped[Optional["Transcription"]] = relationship(
        "Transcription",
        back_populates="audio_file",
        uselist=False,
        cascade="all, delete-orphan",
    )
    analyses: Mapped[List["Analysis"]] = relationship(
        "Analysis",
        back_populates="audio_file",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<AudioFile(id={self.id}, filename='{self.filename}', status={self.status})>"


class Transcription(Base):
    """
    Transcription result from speech-to-text.
    
    Agnostic design:
    - Generic text field for transcription
    - Segments stored separately for granular analysis
    - Language detected automatically
    - Speaker diarization optional (stored in segments)
    """

    __tablename__ = "transcriptions"
    __table_args__ = (
        Index("idx_transcription_audio", "audio_file_id"),
        Index("idx_transcription_status", "status"),
        Index("idx_transcription_language", "language"),
        Index("idx_transcription_text_fts", "text", postgresql_using="gin"),
        {"schema": "audiomind"},
    )

    # Primary Key
    id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
        comment="Unique identifier",
    )

    # Foreign Key
    audio_file_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("audiomind.audio_files.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        comment="Reference to audio file",
    )

    # Transcription Content
    text: Mapped[str] = mapped_column(
        Text, nullable=False, comment="Full transcription text"
    )

    # Language & Confidence
    language: Mapped[Optional[str]] = mapped_column(
        String(10), nullable=True, comment="Detected language code (en, es, fr, etc.)"
    )
    language_confidence: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Confidence score for language detection (0.0-1.0)",
    )

    # Processing Info
    status: Mapped[TranscriptionStatus] = mapped_column(
        SQLEnum(
            TranscriptionStatus,
            name="transcription_status",
            schema="audiomind",
            native_enum=True,
            create_constraint=False,
            values_callable=lambda x: [e.value for e in x]  # Use .value (lowercase)
        ),
        nullable=False,
        default=TranscriptionStatus.PENDING,
        server_default="pending",  # Default in database
        comment="Processing status",
    )
    model_used: Mapped[Optional[str]] = mapped_column(
        String(100), nullable=True, comment="Whisper model used (large-v3-turbo, etc.)"
    )
    processing_time_seconds: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, comment="Time taken to transcribe"
    )

    # Features Used
    vad_enabled: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        comment="Voice Activity Detection used",
    )
    diarization_enabled: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, comment="Speaker diarization used"
    )

    # Error Handling
    error_message: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True, comment="Error message if status=failed"
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Relationships
    audio_file: Mapped["AudioFile"] = relationship(
        "AudioFile",
        back_populates="transcription",
    )
    segments: Mapped[List["TranscriptionSegment"]] = relationship(
        "TranscriptionSegment",
        back_populates="transcription",
        cascade="all, delete-orphan",
        order_by="TranscriptionSegment.start_time",
    )
    embeddings: Mapped[List["Embedding"]] = relationship(
        "Embedding",
        back_populates="transcription",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<Transcription(id={self.id}, audio_id={self.audio_file_id}, status={self.status})>"


class TranscriptionSegment(Base):
    """
    Individual segment of transcription with timing.
    
    Used for:
    - Time-based navigation
    - Speaker diarization
    - Granular embeddings
    - Highlighting specific parts
    """

    __tablename__ = "transcription_segments"
    __table_args__ = (
        Index("idx_segment_transcription", "transcription_id"),
        Index("idx_segment_time", "start_time", "end_time"),
        Index("idx_segment_speaker", "speaker_id"),
        CheckConstraint("end_time > start_time", name="check_valid_time_range"),
        {"schema": "audiomind"},
    )

    # Primary Key
    id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )

    # Foreign Key
    transcription_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("audiomind.transcriptions.id", ondelete="CASCADE"),
        nullable=False,
    )

    # Segment Content
    text: Mapped[str] = mapped_column(Text, nullable=False)

    # Timing
    start_time: Mapped[float] = mapped_column(
        Float, nullable=False, comment="Start time in seconds"
    )
    end_time: Mapped[float] = mapped_column(
        Float, nullable=False, comment="End time in seconds"
    )

    # Speaker Information (optional)
    speaker_id: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        comment="Speaker identifier (SPEAKER_00, SPEAKER_01, etc.)",
    )

    # Confidence Score
    confidence: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, comment="Confidence score for this segment (0.0-1.0)"
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Relationships
    transcription: Mapped["Transcription"] = relationship(
        "Transcription",
        back_populates="segments",
    )

    def __repr__(self) -> str:
        speaker = f", speaker={self.speaker_id}" if self.speaker_id else ""
        return f"<Segment(id={self.id}, time={self.start_time:.1f}-{self.end_time:.1f}{speaker})>"


class Topic(Base):
    """
    Discovered topic from topic modeling.
    
    Agnostic design:
    - Works with LDA, BERTopic, or hybrid approaches
    - Generic topic representation
    - Flexible keywords storage
    """

    __tablename__ = "topics"
    __table_args__ = (
        Index("idx_topic_analysis", "analysis_id"),
        Index("idx_topic_relevance", "relevance_score"),
        Index("idx_topic_keywords_gin", "keywords", postgresql_using="gin"),
        {"schema": "audiomind"},
    )

    # Primary Key
    id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )

    # Foreign Key
    analysis_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("audiomind.analyses.id", ondelete="CASCADE"),
        nullable=False,
    )

    # Topic Information
    topic_number: Mapped[int] = mapped_column(
        Integer, nullable=False, comment="Topic number/ID within analysis"
    )
    label: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True, comment="Human-readable topic label"
    )
    description: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True, comment="Generated description of the topic"
    )

    # Keywords & Scores
    keywords: Mapped[List[str]] = mapped_column(
        ARRAY(String), nullable=False, comment="Top keywords for this topic"
    )
    keyword_weights: Mapped[Optional[Dict[str, float]]] = mapped_column(
        JSONB, nullable=True, comment="Keyword importance scores"
    )

    # Relevance
    relevance_score: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="How relevant is this topic to the content (0.0-1.0)",
    )
    coverage_percentage: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="What % of content belongs to this topic",
    )

    # Methodology
    method_used: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        comment="Method used: LDA, BERTopic, Hybrid",
    )

    # Additional Metadata
    custom_metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSONB,
        nullable=True,
        comment="Additional topic metadata",
        name="metadata",  # Column name in DB is still "metadata"
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Relationships
    analysis: Mapped["Analysis"] = relationship(
        "Analysis",
        back_populates="topics",
    )

    def __repr__(self) -> str:
        keywords_str = ", ".join(self.keywords[:3]) if self.keywords else "N/A"
        return f"<Topic(id={self.id}, #={self.topic_number}, keywords=[{keywords_str}])>"


class Analysis(Base):
    """
    Complete analysis of audio content.
    
    Agnostic design:
    - Contains topics, insights, summaries
    - Works with any LLM provider
    - Flexible results storage in JSONB
    """

    __tablename__ = "analyses"
    __table_args__ = (
        Index("idx_analysis_audio", "audio_file_id"),
        Index("idx_analysis_status", "status"),
        Index("idx_analysis_created", "created_at"),
        Index("idx_analysis_results_gin", "results", postgresql_using="gin"),
        {"schema": "audiomind"},
    )

    # Primary Key
    id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )

    # Foreign Key
    audio_file_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("audiomind.audio_files.id", ondelete="CASCADE"),
        nullable=False,
    )

    # Analysis Type
    analysis_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        comment="Type: topic_modeling, sentiment, summary, etc.",
    )

    # Status
    status: Mapped[AnalysisStatus] = mapped_column(
        SQLEnum(
            AnalysisStatus,
            name="analysis_status",
            schema="audiomind",
            native_enum=True,
            create_constraint=False,
            values_callable=lambda x: [e.value for e in x]  # Use .value (lowercase)
        ),
        nullable=False,
        default=AnalysisStatus.PENDING,
        server_default="pending",  # Default in database
    )

    # Results (flexible JSONB for different analysis types)
    results: Mapped[Dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        comment="Analysis results - structure depends on analysis_type",
    )

    # Summary & Insights
    summary: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True, comment="Human-readable summary"
    )
    insights: Mapped[Optional[List[str]]] = mapped_column(
        ARRAY(Text), nullable=True, comment="Key insights extracted"
    )

    # Model Information
    model_used: Mapped[Optional[str]] = mapped_column(
        String(100), nullable=True, comment="LLM model used (gpt-4o, claude-3, etc.)"
    )
    processing_time_seconds: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )

    # Configuration Used
    config_snapshot: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSONB,
        nullable=True,
        comment="Configuration used for this analysis",
    )

    # Error Handling
    error_message: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Relationships
    audio_file: Mapped["AudioFile"] = relationship(
        "AudioFile",
        back_populates="analyses",
    )
    topics: Mapped[List["Topic"]] = relationship(
        "Topic",
        back_populates="analysis",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<Analysis(id={self.id}, type={self.analysis_type}, status={self.status})>"


class Embedding(Base):
    """
    Vector embeddings for RAG (Retrieval Augmented Generation).
    
    Stores embeddings for:
    - Full transcription
    - Individual segments
    - Topics
    - Summaries
    
    Used for semantic search and similarity.
    """

    __tablename__ = "embeddings"
    __table_args__ = (
        Index("idx_embedding_transcription", "transcription_id"),
        Index("idx_embedding_segment", "segment_id"),
        Index("idx_embedding_type", "embedding_type"),
        {"schema": "audiomind"},
    )

    # Primary Key
    id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )

    # Foreign Keys (one of these must be set)
    transcription_id: Mapped[Optional[UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("audiomind.transcriptions.id", ondelete="CASCADE"),
        nullable=True,
    )
    segment_id: Mapped[Optional[UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("audiomind.transcription_segments.id", ondelete="CASCADE"),
        nullable=True,
    )

    # Embedding Data
    embedding_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        comment="Type: full_text, segment, topic, summary",
    )
    vector: Mapped[List[float]] = mapped_column(
        ARRAY(Float),
        nullable=False,
        comment="Embedding vector",
    )

    # Model Info
    model_used: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        comment="Embedding model (text-embedding-3-small, etc.)",
    )
    dimension: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="Vector dimension (1536, 3072, etc.)",
    )

    # Text that was embedded
    text_excerpt: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="First 500 chars of embedded text (for reference)",
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Relationships
    transcription: Mapped[Optional["Transcription"]] = relationship(
        "Transcription",
        back_populates="embeddings",
    )

    def __repr__(self) -> str:
        return f"<Embedding(id={self.id}, type={self.embedding_type}, dim={self.dimension})>"
