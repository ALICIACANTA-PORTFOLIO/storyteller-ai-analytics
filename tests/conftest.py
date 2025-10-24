"""
Pytest Configuration and Shared Fixtures
=======================================

This module provides shared fixtures and configuration for all tests.

Key Fixtures:
- db_engine: Async SQLAlchemy engine for tests
- db_session: Async database session (auto-cleanup)
- test_audio_file: Sample audio file for testing
- mock_transcription_result: Mock Whisper output
"""

import asyncio
import os
from pathlib import Path
from typing import AsyncGenerator, Generator
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from faker import Faker
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

from app.config import Settings, get_settings
from app.models.database import Base
from app.processors.whisper_processor import (
    TranscriptionResult,
    TranscriptionSegment,
)
from app.processors.topic_modeler import (
    TopicModelingResult,
    Topic as TopicModel,
)

# Initialize Faker for test data generation
fake = Faker()

# Test database URL - Using PostgreSQL (already configured locally)
# Uses AudioMind database for tests (we'll clean up after each test)
TEST_DB_URL = os.getenv(
    "TEST_DATABASE_URL",
    "postgresql+asyncpg://postgres:postgres@localhost:5432/AudioMind"
)


# ============================================================================
# Session-scoped Fixtures (Created once per test session)
# ============================================================================

@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_settings() -> Settings:
    """Get test configuration settings."""
    return get_settings()


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Path to test data directory."""
    path = Path(__file__).parent / "fixtures" / "data"
    path.mkdir(parents=True, exist_ok=True)
    return path


# ============================================================================
# Database Fixtures
# ============================================================================

@pytest_asyncio.fixture(scope="function")
async def db_engine() -> AsyncGenerator[AsyncEngine, None]:
    """
    Create async database engine for tests.
    
    Uses PostgreSQL (audiomind database) for all tests.
    Each test function gets isolated transactions.
    """
    # Use PostgreSQL with connection pooling disabled for tests
    engine = create_async_engine(
        TEST_DB_URL,
        echo=False,
        poolclass=NullPool,  # Disable pooling for tests
    )
    
    yield engine
    
    # Cleanup: dispose engine (tables remain in database)
    await engine.dispose()


@pytest_asyncio.fixture(scope="function")
async def db_session(db_engine: AsyncEngine) -> AsyncGenerator[AsyncSession, None]:
    """
    Create async database session for tests.
    
    Uses transactions for isolation - each test runs in its own transaction
    that gets rolled back automatically after the test completes.
    """
    # Create connection
    async with db_engine.connect() as connection:
        # Begin transaction
        transaction = await connection.begin()
        
        # Create session bound to this connection
        async_session_maker = sessionmaker(
            bind=connection,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        
        async with async_session_maker() as session:
            yield session
            
            # Rollback transaction (cleanup)
            await transaction.rollback()


# ============================================================================
# Audio File Fixtures
# ============================================================================

@pytest.fixture
def sample_audio_path(test_data_dir: Path) -> Path:
    """
    Path to sample audio file.
    
    Returns path to existing test audio or creates a placeholder.
    """
    # Check if we have a real test audio file
    audio_path = test_data_dir / "test_audio.mp3"
    
    if not audio_path.exists():
        # Create a small placeholder file (won't work for real transcription)
        audio_path.write_bytes(b"fake audio data for testing")
    
    return audio_path


@pytest.fixture
def real_audio_path() -> Path:
    """
    Path to real audio file for integration tests.
    
    Uses the audio file from Books/ directory.
    """
    audio_path = Path("Books/story-telling/gutenberg_21144_ch01.mp3")
    
    if not audio_path.exists():
        pytest.skip(f"Real audio file not found: {audio_path}")
    
    return audio_path


# ============================================================================
# Mock Data Fixtures
# ============================================================================

@pytest.fixture
def mock_transcription_result() -> TranscriptionResult:
    """
    Mock transcription result for testing.
    
    Simulates Whisper output without actually running the model.
    """
    segments = [
        TranscriptionSegment(
            text="This is the first segment of transcribed audio.",
            start_time=0.0,
            end_time=5.5,
            confidence=0.95,
        ),
        TranscriptionSegment(
            text="This is the second segment with different content.",
            start_time=5.5,
            end_time=12.3,
            confidence=0.92,
        ),
        TranscriptionSegment(
            text="And here is the third and final segment.",
            start_time=12.3,
            end_time=18.7,
            confidence=0.88,
        ),
    ]
    
    full_text = " ".join(seg.text for seg in segments)
    
    return TranscriptionResult(
        text=full_text,
        segments=segments,
        language="en",
        language_confidence=0.98,
        duration_seconds=18.7,
        processing_time_seconds=2.5,
        model_used="whisper-base",
        vad_enabled=False,
        diarization_enabled=False,
    )


@pytest.fixture
def mock_topic_analysis_result() -> TopicModelingResult:
    """Mock topic analysis result for testing."""
    topics = [
        TopicModel(
            topic_number=0,
            label="Technology & Innovation",
            keywords=["AI", "machine learning", "technology", "innovation", "data"],
            keyword_weights={"AI": 0.5, "machine learning": 0.4, "technology": 0.3, "innovation": 0.2, "data": 0.1},
            relevance_score=0.85,
            method_used="hybrid",
            representative_docs=["Sample document about AI technology..."],
        ),
        TopicModel(
            topic_number=1,
            label="Business & Strategy",
            keywords=["business", "strategy", "market", "growth", "customer"],
            keyword_weights={"business": 0.5, "strategy": 0.4, "market": 0.3, "growth": 0.2, "customer": 0.1},
            relevance_score=0.78,
            method_used="hybrid",
            representative_docs=["Sample document about business strategy..."],
        ),
    ]
    
    return TopicModelingResult(
        topics=topics,
        num_topics=2,
        coherence_scores={"c_v": 0.65, "c_uci": 0.42, "u_mass": -0.85},
        processing_time_seconds=1.8,
        method_used="hybrid",
    )


@pytest.fixture
def sample_uuid() -> UUID:
    """Generate a sample UUID for testing."""
    return uuid4()


@pytest.fixture
def mock_llm_analysis() -> dict:
    """Mock LLM analysis result for testing."""
    return {
        "summary": "This is a comprehensive summary of the audio content.",
        "insights": [
            "Key insight number one about the content",
            "Important observation number two",
            "Third critical finding from the analysis",
        ],
        "action_items": [
            "Follow up on the first point discussed",
            "Investigate the second topic further",
        ],
        "sentiment": {
            "overall": "positive",
            "score": 0.75,
        },
        "questions_raised": [
            "What are the next steps?",
            "How can this be implemented?",
        ],
        "metadata": {
            "model": "gpt-4o-mini",
            "tokens_used": 1250,
        },
    }


# ============================================================================
# Helper Functions
# ============================================================================

@pytest.fixture
def create_audio_file_data() -> dict:
    """Factory fixture to create audio file data."""
    def _create(
        filename: str = None,
        original_path: str = None,
        file_size_bytes: int = None,
    ) -> dict:
        return {
            "filename": filename or f"{fake.file_name(extension='mp3')}",
            "original_path": original_path or f"/tmp/{fake.file_name(extension='mp3')}",
            "file_size_bytes": file_size_bytes or fake.random_int(min=100000, max=10000000),
            "duration_seconds": fake.random_int(min=30, max=3600),
            "mime_type": "audio/mpeg",
            "custom_metadata": {
                "source": fake.random_element(["upload", "recording", "import"]),
                "user_id": str(uuid4()),
            },
        }
    
    return _create


# ============================================================================
# Test Markers Registration
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: Unit tests (fast, isolated)"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests (slower, requires services)"
    )
    config.addinivalue_line(
        "markers", "slow: Slow tests (may take >5 seconds)"
    )
    config.addinivalue_line(
        "markers", "requires_gpu: Tests requiring GPU"
    )
    config.addinivalue_line(
        "markers", "requires_model: Tests requiring ML models"
    )
    config.addinivalue_line(
        "markers", "smoke: Smoke tests (critical path)"
    )


# ============================================================================
# Pytest Hooks
# ============================================================================

def pytest_collection_modifyitems(config, items):
    """
    Modify test collection to add markers automatically.
    
    - Tests in tests/unit/ get @pytest.mark.unit
    - Tests in tests/integration/ get @pytest.mark.integration
    """
    for item in items:
        # Add unit marker to unit tests
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        
        # Add integration marker to integration tests
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Mark slow tests (async tests with real models)
        if "whisper" in item.name or "topic" in item.name:
            if "real" in item.name or "integration" in str(item.fspath):
                item.add_marker(pytest.mark.slow)
