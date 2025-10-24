"""
Database session management and helper functions.

This module provides async database session management for FastAPI,
along with helper functions to interact with the database.

Author: Alicia Canta
Date: 2025-10-24
"""

import logging
from typing import AsyncGenerator, Optional, List, Dict, Any
from contextlib import asynccontextmanager
from uuid import UUID

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    AsyncEngine,
    create_async_engine,
    async_sessionmaker,
)
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.config import settings
from app.models import (
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
from app.processors.whisper_processor import TranscriptionResult
from app.processors.topic_modeler import TopicModelingResult

# Configure logging
logger = logging.getLogger(__name__)

# Global engine instance
_engine: Optional[AsyncEngine] = None
_session_factory: Optional[async_sessionmaker[AsyncSession]] = None


def get_engine() -> AsyncEngine:
    """
    Get or create the global database engine.
    
    Returns:
        AsyncEngine: SQLAlchemy async engine
    """
    global _engine
    
    if _engine is None:
        logger.info("🔧 Creating database engine...")
        logger.info(f"   Database: {settings.database.host}:{settings.database.port}/{settings.database.database}")
        
        _engine = create_async_engine(
            settings.database.async_connection_string,
            echo=settings.database.echo_sql,
            pool_size=settings.database.pool_size,
            max_overflow=settings.database.max_overflow,
            pool_pre_ping=True,  # Verify connections before using
            pool_recycle=3600,   # Recycle connections after 1 hour
        )
        
        logger.info("✅ Database engine created")
    
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """
    Get or create the session factory.
    
    Returns:
        async_sessionmaker: Factory for creating async sessions
    """
    global _session_factory
    
    if _session_factory is None:
        engine = get_engine()
        _session_factory = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,  # Don't expire objects after commit
            autoflush=False,         # Manual flush control
            autocommit=False,        # Manual commit control
        )
        logger.info("✅ Session factory created")
    
    return _session_factory


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency for getting database sessions.
    
    Usage:
        @app.get("/items")
        async def get_items(db: AsyncSession = Depends(get_db)):
            ...
    
    Yields:
        AsyncSession: Database session
    """
    session_factory = get_session_factory()
    
    async with session_factory() as session:
        try:
            yield session
        except Exception as e:
            logger.error(f"❌ Database session error: {e}")
            await session.rollback()
            raise
        finally:
            await session.close()


@asynccontextmanager
async def get_db_context():
    """
    Context manager for database sessions (for non-FastAPI code).
    
    Usage:
        async with get_db_context() as db:
            result = await db.execute(...)
    
    Yields:
        AsyncSession: Database session
    """
    session_factory = get_session_factory()
    
    async with session_factory() as session:
        try:
            yield session
        except Exception as e:
            logger.error(f"❌ Database context error: {e}")
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db():
    """
    Initialize database (create tables).
    
    This is typically run once during application startup.
    """
    engine = get_engine()
    
    logger.info("📊 Initializing database tables...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("✅ Database tables initialized")


async def close_db():
    """
    Close database connections.
    
    This should be called during application shutdown.
    """
    global _engine, _session_factory
    
    if _engine:
        logger.info("🔒 Closing database connections...")
        await _engine.dispose()
        _engine = None
        _session_factory = None
        logger.info("✅ Database connections closed")


# ============================================================================
# HELPER FUNCTIONS - Audio Files
# ============================================================================

async def save_audio_file(
    db: AsyncSession,
    file_path: str,
    filename: str,
    file_size_bytes: int,
    mime_type: str = "audio/mpeg",
    duration_seconds: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None,
    uploaded_by: Optional[str] = None,
    source: Optional[str] = None,
) -> AudioFile:
    """
    Save audio file metadata to database.
    
    Args:
        db: Database session
        file_path: Path to stored audio file
        filename: Original filename
        file_size_bytes: File size in bytes
        mime_type: MIME type (e.g., "audio/mpeg")
        duration_seconds: Audio duration (optional)
        metadata: Additional metadata (optional)
        uploaded_by: User who uploaded (optional)
        source: Source of upload (optional)
    
    Returns:
        AudioFile: Created audio file record
    """
    audio_file = AudioFile(
        file_path=file_path,
        filename=filename,
        file_size_bytes=file_size_bytes,
        mime_type=mime_type,
        duration_seconds=duration_seconds,
        custom_metadata=metadata or {},
        uploaded_by=uploaded_by,
        source=source,
        status=AudioStatus.UPLOADED.value,  # Use .value to get the string
    )
    
    db.add(audio_file)
    await db.commit()
    await db.refresh(audio_file)
    
    logger.info(f"✅ Audio file saved: {audio_file.id} ({filename})")
    
    return audio_file


async def get_audio_file(db: AsyncSession, audio_file_id: UUID) -> Optional[AudioFile]:
    """
    Get audio file by ID.
    
    Args:
        db: Database session
        audio_file_id: Audio file UUID
    
    Returns:
        AudioFile or None
    """
    result = await db.execute(
        select(AudioFile).where(AudioFile.id == audio_file_id)
    )
    return result.scalar_one_or_none()


# ============================================================================
# HELPER FUNCTIONS - Transcriptions
# ============================================================================

async def save_transcription(
    db: AsyncSession,
    audio_file_id: UUID,
    transcription_result: TranscriptionResult,
    model_name: str = "whisper-large-v3-turbo",
) -> Transcription:
    """
    Save transcription result to database.
    
    Args:
        db: Database session
        audio_file_id: Associated audio file UUID
        transcription_result: Result from WhisperProcessor
        model_name: Name of the model used
    
    Returns:
        Transcription: Created transcription record with segments
    """
    # Create main transcription record
    transcription = Transcription(
        audio_file_id=audio_file_id,
        text=transcription_result.text,
        language=transcription_result.language,
        language_confidence=transcription_result.language_confidence,
        processing_time_seconds=transcription_result.processing_time_seconds,
        model_used=transcription_result.model_used,
        vad_enabled=transcription_result.vad_enabled,
        diarization_enabled=transcription_result.diarization_enabled,
        status=TranscriptionStatus.COMPLETED.value,  # Use .value to get lowercase string
    )
    
    db.add(transcription)
    await db.flush()  # Get transcription.id before adding segments
    
    # Create segment records
    for segment in transcription_result.segments:
        db_segment = TranscriptionSegment(
            transcription_id=transcription.id,
            start_time=segment.start_time,
            end_time=segment.end_time,
            text=segment.text,
            speaker_id=segment.speaker_id,
            confidence=segment.confidence,
        )
        db.add(db_segment)
    
    await db.commit()
    await db.refresh(transcription)
    
    logger.info(
        f"✅ Transcription saved: {transcription.id} "
        f"({len(transcription_result.segments)} segments, "
        f"{transcription_result.language})"
    )
    
    return transcription


async def get_transcription(
    db: AsyncSession,
    transcription_id: UUID,
    include_segments: bool = True,
) -> Optional[Transcription]:
    """
    Get transcription by ID.
    
    Args:
        db: Database session
        transcription_id: Transcription UUID
        include_segments: Whether to load segments
    
    Returns:
        Transcription or None
    """
    query = select(Transcription).where(Transcription.id == transcription_id)
    
    if include_segments:
        query = query.options(selectinload(Transcription.segments))
    
    result = await db.execute(query)
    return result.scalar_one_or_none()


async def get_transcription_by_audio(
    db: AsyncSession,
    audio_file_id: UUID,
) -> Optional[Transcription]:
    """
    Get transcription for an audio file.
    
    Args:
        db: Database session
        audio_file_id: Audio file UUID
    
    Returns:
        Transcription or None
    """
    result = await db.execute(
        select(Transcription)
        .where(Transcription.audio_file_id == audio_file_id)
        .options(selectinload(Transcription.segments))
        .order_by(Transcription.created_at.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


# ============================================================================
# HELPER FUNCTIONS - Topic Analysis
# ============================================================================

async def save_topic_analysis(
    db: AsyncSession,
    audio_file_id: UUID,
    topic_result: TopicModelingResult,
    model_used: str = "hybrid",
) -> Analysis:
    """
    Save topic modeling results to database.
    
    Args:
        db: Database session
        audio_file_id: Associated audio file UUID
        topic_result: Result from HybridTopicModeler
        model_used: Method used ("lda", "bertopic", or "hybrid")
    
    Returns:
        Analysis: Created analysis record with topics
    """
    # Create main analysis record
    analysis = Analysis(
        audio_file_id=audio_file_id,
        analysis_type="topic_modeling",
        status=AnalysisStatus.COMPLETED.value,  # Use .value to get lowercase string
        model_used=model_used,
        results={
            "num_topics": topic_result.num_topics,
            "coherence_scores": topic_result.coherence_scores,
            "method_used": topic_result.method_used,
            "metadata": topic_result.metadata,
            "processing_time": topic_result.processing_time_seconds,
        },
        summary=f"Extracted {topic_result.num_topics} topics using {topic_result.method_used} method",
        config_snapshot={
            "method": model_used,
            "num_topics": topic_result.num_topics,
        },
    )
    
    db.add(analysis)
    await db.flush()  # Get analysis.id before adding topics
    
    # Create topic records
    for topic in topic_result.topics:
        db_topic = Topic(
            analysis_id=analysis.id,
            topic_number=topic.topic_number,
            label=topic.label,
            keywords=topic.keywords,
            keyword_weights=topic.keyword_weights,
            relevance_score=topic.relevance_score,
            method_used=topic.method_used,
            custom_metadata={
                "representative_docs": topic.representative_docs,
                "document_count": topic.document_count,
            },
        )
        db.add(db_topic)
    
    await db.commit()
    await db.refresh(analysis)
    
    # Calculate average coherence if available
    avg_coherence = 0.0
    if topic_result.coherence_scores:
        avg_coherence = sum(topic_result.coherence_scores.values()) / len(topic_result.coherence_scores)
    
    logger.info(
        f"✅ Topic analysis saved: {analysis.id} "
        f"({topic_result.num_topics} topics, "
        f"avg coherence: {avg_coherence:.3f})"
    )
    
    return analysis


async def get_topic_analysis(
    db: AsyncSession,
    analysis_id: UUID,
    include_topics: bool = True,
) -> Optional[Analysis]:
    """
    Get topic analysis by ID.
    
    Args:
        db: Database session
        analysis_id: Analysis UUID
        include_topics: Whether to load topics
    
    Returns:
        Analysis or None
    """
    query = select(Analysis).where(
        Analysis.id == analysis_id,
        Analysis.analysis_type == "topic_modeling"
    )
    
    if include_topics:
        query = query.options(selectinload(Analysis.topics))
    
    result = await db.execute(query)
    return result.scalar_one_or_none()


async def get_topic_analysis_by_audio(
    db: AsyncSession,
    audio_file_id: UUID,
) -> Optional[Analysis]:
    """
    Get topic analysis for an audio file.
    
    Args:
        db: Database session
        audio_file_id: Audio file UUID
    
    Returns:
        Analysis or None
    """
    result = await db.execute(
        select(Analysis)
        .where(
            Analysis.audio_file_id == audio_file_id,
            Analysis.analysis_type == "topic_modeling"
        )
        .options(selectinload(Analysis.topics))
        .order_by(Analysis.created_at.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


# ============================================================================
# HELPER FUNCTIONS - LLM Analysis
# ============================================================================

async def save_llm_analysis(
    db: AsyncSession,
    audio_file_id: UUID,
    analysis_type: str,
    summary: Optional[str] = None,
    insights: Optional[List[str]] = None,
    sentiment: Optional[Dict[str, Any]] = None,
    action_items: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Analysis:
    """
    Save LLM analysis results to database.
    
    Args:
        db: Database session
        audio_file_id: Associated audio file UUID
        analysis_type: Type of analysis (e.g., "summary", "insights")
        summary: Generated summary
        insights: List of key insights
        sentiment: Sentiment analysis results
        action_items: List of action items
        metadata: Additional metadata
    
    Returns:
        Analysis: Created analysis record
    """
    analysis = Analysis(
        audio_file_id=audio_file_id,
        analysis_type=analysis_type,
        status=AnalysisStatus.COMPLETED.value,  # Use .value to get lowercase string
        results={
            "summary": summary,
            "insights": insights or [],
            "sentiment": sentiment,
            "action_items": action_items or [],
        },
        summary=summary,
        insights=insights or [],
        config_snapshot=metadata or {},
    )
    
    db.add(analysis)
    await db.commit()
    await db.refresh(analysis)
    
    logger.info(f"✅ LLM analysis saved: {analysis.id} (type: {analysis_type})")
    
    return analysis


async def get_analysis(db: AsyncSession, analysis_id: UUID) -> Optional[Analysis]:
    """
    Get analysis by ID.
    
    Args:
        db: Database session
        analysis_id: Analysis UUID
    
    Returns:
        Analysis or None
    """
    result = await db.execute(
        select(Analysis).where(Analysis.id == analysis_id)
    )
    return result.scalar_one_or_none()


# ============================================================================
# HELPER FUNCTIONS - Embeddings
# ============================================================================

async def save_embedding(
    db: AsyncSession,
    transcription_id: UUID,
    chunk_text: str,
    chunk_index: int,
    embedding_vector: List[float],
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    metadata: Optional[dict] = None,
) -> Embedding:
    """
    Save embedding to database.
    
    Args:
        db: Database session
        transcription_id: Associated transcription UUID
        chunk_text: Text chunk
        chunk_index: Index of chunk in sequence
        embedding_vector: Embedding vector (list of floats)
        start_time: Start time in audio (optional)
        end_time: End time in audio (optional)
        model_name: Embedding model name
        metadata: Additional metadata
    
    Returns:
        Embedding: Created embedding record
    """
    embedding = Embedding(
        transcription_id=transcription_id,
        chunk_text=chunk_text,
        chunk_index=chunk_index,
        embedding_vector=embedding_vector,
        start_time=start_time,
        end_time=end_time,
        model_name=model_name,
        metadata_=metadata or {},
    )
    
    db.add(embedding)
    await db.commit()
    await db.refresh(embedding)
    
    return embedding


async def get_embeddings_by_transcription(
    db: AsyncSession,
    transcription_id: UUID,
) -> List[Embedding]:
    """
    Get all embeddings for a transcription.
    
    Args:
        db: Database session
        transcription_id: Transcription UUID
    
    Returns:
        List of embeddings
    """
    result = await db.execute(
        select(Embedding)
        .where(Embedding.transcription_id == transcription_id)
        .order_by(Embedding.chunk_index)
    )
    return list(result.scalars().all())


# ============================================================================
# HELPER FUNCTIONS - Processing Status
# ============================================================================

async def update_audio_status(
    db: AsyncSession,
    audio_file_id: UUID,
    status: AudioStatus,
    error_message: Optional[str] = None,
):
    """
    Update processing status for an audio file.
    
    Args:
        db: Database session
        audio_file_id: Audio file UUID
        status: New processing status
        error_message: Error message if failed
    """
    audio_file = await get_audio_file(db, audio_file_id)
    
    if audio_file:
        audio_file.status = status
        
        if error_message and status == AudioStatus.FAILED:
            # Store error in metadata
            if audio_file.custom_metadata is None:
                audio_file.custom_metadata = {}
            audio_file.custom_metadata["error_message"] = error_message
        
        await db.commit()
        
        logger.info(
            f"📊 Audio status updated: {audio_file_id} -> {status.value}"
        )


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

async def get_full_analysis_by_audio(
    db: AsyncSession,
    audio_file_id: UUID,
) -> Optional[dict]:
    """
    Get complete analysis data for an audio file.
    
    Returns dictionary with audio, transcription, topics, and analysis.
    
    Args:
        db: Database session
        audio_file_id: Audio file UUID
    
    Returns:
        Dictionary with all data or None
    """
    # Get audio file
    audio = await get_audio_file(db, audio_file_id)
    if not audio:
        return None
    
    # Get transcription
    transcription = await get_transcription_by_audio(db, audio_file_id)
    
    # Get topic analysis
    topic_analysis = None
    if audio:
        topic_analysis = await get_topic_analysis_by_audio(db, audio_file_id)
    
    # Get LLM analysis
    llm_analysis = None
    if audio:
        result = await db.execute(
            select(Analysis)
            .where(
                Analysis.audio_file_id == audio_file_id,
                Analysis.analysis_type != "topic_modeling"  # Get non-topic analysis
            )
            .order_by(Analysis.created_at.desc())
            .limit(1)
        )
        llm_analysis = result.scalar_one_or_none()
    
    return {
        "audio_file": audio,
        "transcription": transcription,
        "topic_analysis": topic_analysis,
        "llm_analysis": llm_analysis,
    }
