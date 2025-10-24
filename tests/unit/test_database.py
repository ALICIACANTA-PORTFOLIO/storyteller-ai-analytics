"""
Unit Tests for Database Helper Functions
========================================

Tests for database.py helper functions including:
- save_audio_file()
- save_transcription()
- save_topic_analysis()
- save_llm_analysis()
- update_audio_status()
- get_full_analysis_by_audio()
"""

from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import (
    get_full_analysis_by_audio,
    save_audio_file,
    save_llm_analysis,
    save_topic_analysis,
    save_transcription,
    update_audio_status,
)
from app.models.database import (
    Analysis,
    AudioFile,
    AudioStatus,
    Transcription,
    TranscriptionSegment,
    Topic,
)
from app.processors.whisper_processor import TranscriptionResult
from app.processors.topic_modeler import TopicModelingResult


@pytest.mark.unit
@pytest.mark.asyncio
class TestSaveAudioFile:
    """Tests for save_audio_file() function."""

    async def test_save_audio_file_basic(self, db_session: AsyncSession):
        """Test saving a basic audio file."""
        # Arrange
        audio_data = {
            "filename": "test_audio.mp3",
            "file_path": "/tmp/test_audio.mp3",
            "file_size_bytes": 1024000,
            "duration_seconds": 120.5,
            "mime_type": "audio/mpeg",
        }

        # Act
        audio_file = await save_audio_file(db_session, **audio_data)

        # Assert
        assert audio_file.id is not None
        assert audio_file.filename == "test_audio.mp3"
        assert audio_file.file_size_bytes == 1024000
        assert audio_file.duration_seconds == 120.5
        assert audio_file.status.value == AudioStatus.UPLOADED.value
        assert audio_file.custom_metadata == {}  # Default is empty dict

    async def test_save_audio_file_with_metadata(self, db_session: AsyncSession):
        """Test saving audio file with custom metadata."""
        # Arrange
        metadata = {"source": "upload", "user_id": "123"}

        # Act
        audio_file = await save_audio_file(
            db_session,
            filename="test.mp3",
            file_path="/tmp/test.mp3",
            file_size_bytes=500000,
            duration_seconds=60.0,
            mime_type="audio/mpeg",
            metadata=metadata,
        )

        # Assert
        assert audio_file.custom_metadata == metadata
        assert audio_file.custom_metadata["source"] == "upload"

    async def test_save_audio_file_persists_to_db(self, db_session: AsyncSession):
        """Test that audio file is actually persisted to database."""
        # Arrange & Act
        audio_file = await save_audio_file(
            db_session,
            filename="persist_test.mp3",
            file_path="/tmp/persist.mp3",
            file_size_bytes=1000,
            duration_seconds=30.0,
        )

        # Refresh to ensure data is from DB
        await db_session.commit()
        await db_session.refresh(audio_file)

        # Query directly from DB
        result = await db_session.execute(
            select(AudioFile).where(AudioFile.id == audio_file.id)
        )
        db_audio = result.scalar_one()

        # Assert
        assert db_audio is not None
        assert db_audio.filename == "persist_test.mp3"


@pytest.mark.unit
@pytest.mark.asyncio
class TestSaveTranscription:
    """Tests for save_transcription() function."""

    async def test_save_transcription_basic(
        self,
        db_session: AsyncSession,
        mock_transcription_result: TranscriptionResult,
    ):
        """Test saving a basic transcription."""
        # Arrange: Create audio file first
        audio_file = await save_audio_file(
            db_session,
            filename="test.mp3",
            file_path="/tmp/test.mp3",
            file_size_bytes=1000,
            duration_seconds=30.0,
        )

        # Act
        transcription = await save_transcription(
            db_session,
            audio_file.id,
            mock_transcription_result,
        )

        # Assert
        assert transcription.id is not None
        assert transcription.audio_file_id == audio_file.id
        assert transcription.text == mock_transcription_result.text
        assert transcription.language == "en"
        assert transcription.language_confidence == 0.98

    async def test_save_transcription_with_segments(
        self,
        db_session: AsyncSession,
        mock_transcription_result: TranscriptionResult,
    ):
        """Test that transcription segments are saved correctly."""
        # Arrange
        audio_file = await save_audio_file(
            db_session,
            filename="segments_test.mp3",
            file_path="/tmp/test.mp3",
            file_size_bytes=1000,
            duration_seconds=30.0,
        )

        # Act
        transcription = await save_transcription(
            db_session,
            audio_file.id,
            mock_transcription_result,
        )

        await db_session.commit()
        await db_session.refresh(transcription)

        # Query segments
        result = await db_session.execute(
            select(TranscriptionSegment).where(
                TranscriptionSegment.transcription_id == transcription.id
            )
        )
        segments = result.scalars().all()

        # Assert
        assert len(segments) == 3  # mock has 3 segments
        assert segments[0].text == "This is the first segment of transcribed audio."
        assert segments[0].start_time == 0.0
        assert segments[0].end_time == 5.5


@pytest.mark.unit
@pytest.mark.asyncio
class TestSaveTopicAnalysis:
    """Tests for save_topic_analysis() function."""

    async def test_save_topic_analysis(
        self,
        db_session: AsyncSession,
        mock_topic_analysis_result: TopicModelingResult,
    ):
        """Test saving topic analysis results."""
        # Arrange: Create audio + transcription
        audio_file = await save_audio_file(
            db_session,
            filename="topics_test.mp3",
            file_path="/tmp/test.mp3",
            file_size_bytes=1000,
            duration_seconds=30.0,
        )

        # Act
        analysis = await save_topic_analysis(
            db_session,
            audio_file.id,
            mock_topic_analysis_result,
        )

        await db_session.commit()

        # Query topics
        result = await db_session.execute(
            select(Topic).where(Topic.analysis_id == analysis.id)
        )
        topics = result.scalars().all()

        # Assert
        assert analysis.id is not None
        assert analysis.audio_file_id == audio_file.id
        assert analysis.analysis_type == "topic_modeling"
        assert len(topics) == 2  # mock has 2 topics


@pytest.mark.unit
@pytest.mark.asyncio
class TestSaveLLMAnalysis:
    """Tests for save_llm_analysis() function."""

    async def test_save_llm_analysis(
        self,
        db_session: AsyncSession,
        mock_llm_analysis: dict,
    ):
        """Test saving LLM analysis results."""
        # Arrange
        audio_file = await save_audio_file(
            db_session,
            filename="llm_test.mp3",
            file_path="/tmp/test.mp3",
            file_size_bytes=1000,
            duration_seconds=30.0,
        )

        # Act
        analysis = await save_llm_analysis(
            db_session,
            audio_file.id,
            analysis_type="llm_synthesis",
            summary=mock_llm_analysis["summary"],
            insights=mock_llm_analysis["insights"],
            sentiment=mock_llm_analysis["sentiment"],
            action_items=mock_llm_analysis["action_items"],
            metadata=mock_llm_analysis["metadata"],
        )

        await db_session.commit()
        await db_session.refresh(analysis)

        # Assert
        assert analysis.id is not None
        assert analysis.analysis_type == "llm_synthesis"
        assert analysis.results["summary"] == mock_llm_analysis["summary"]
        assert len(analysis.results["insights"]) == 3


@pytest.mark.unit
@pytest.mark.asyncio
class TestUpdateAudioStatus:
    """Tests for update_audio_status() function."""

    async def test_update_status_to_processing(self, db_session: AsyncSession):
        """Test updating audio status to processing."""
        # Arrange
        audio_file = await save_audio_file(
            db_session,
            filename="status_test.mp3",
            file_path="/tmp/test.mp3",
            file_size_bytes=1000,
            duration_seconds=30.0,
        )
        await db_session.commit()

        initial_status = audio_file.status

        # Act
        await update_audio_status(
            db_session,
            audio_file.id,
            AudioStatus.PROCESSING,  # Pass enum, not .value
        )

        await db_session.refresh(audio_file)

        # Assert
        assert initial_status.value == AudioStatus.UPLOADED.value
        assert audio_file.status.value == AudioStatus.PROCESSING.value

    async def test_update_status_to_completed(self, db_session: AsyncSession):
        """Test updating audio status to completed."""
        # Arrange
        audio_file = await save_audio_file(
            db_session,
            filename="completed_test.mp3",
            file_path="/tmp/test.mp3",
            file_size_bytes=1000,
            duration_seconds=30.0,
        )
        await db_session.commit()

        # Act
        await update_audio_status(
            db_session,
            audio_file.id,
            AudioStatus.COMPLETED,  # Pass enum, not .value
        )

        await db_session.refresh(audio_file)

        # Assert
        assert audio_file.status.value == AudioStatus.COMPLETED.value


@pytest.mark.unit
@pytest.mark.asyncio
class TestGetFullAnalysis:
    """Tests for get_full_analysis_by_audio() function."""

    async def test_get_full_analysis_complete_workflow(
        self,
        db_session: AsyncSession,
        mock_transcription_result: TranscriptionResult,
        mock_topic_analysis_result: TopicModelingResult,
        mock_llm_analysis: dict,
    ):
        """Test retrieving full analysis with all components."""
        # Arrange: Create complete workflow
        audio_file = await save_audio_file(
            db_session,
            filename="full_analysis_test.mp3",
            file_path="/tmp/test.mp3",
            file_size_bytes=1000,
            duration_seconds=30.0,
        )

        transcription = await save_transcription(
            db_session,
            audio_file.id,
            mock_transcription_result,
        )

        topic_analysis = await save_topic_analysis(
            db_session,
            audio_file.id,
            mock_topic_analysis_result,
        )

        llm_analysis = await save_llm_analysis(
            db_session,
            audio_file.id,
            analysis_type="llm_synthesis",
            summary=mock_llm_analysis["summary"],
            insights=mock_llm_analysis["insights"],
            sentiment=mock_llm_analysis["sentiment"],
            action_items=mock_llm_analysis["action_items"],
            metadata=mock_llm_analysis["metadata"],
        )

        await db_session.commit()

        # Act
        result = await get_full_analysis_by_audio(db_session, audio_file.id)

        # Assert
        assert result is not None
        assert result["audio_file"].id == audio_file.id
        assert result["transcription"].id == transcription.id
        assert result["topic_analysis"].id == topic_analysis.id
        assert result["llm_analysis"].id == llm_analysis.id

    async def test_get_full_analysis_nonexistent_audio(
        self,
        db_session: AsyncSession,
    ):
        """Test retrieving analysis for non-existent audio file."""
        # Act
        result = await get_full_analysis_by_audio(db_session, uuid4())

        # Assert
        assert result is None
