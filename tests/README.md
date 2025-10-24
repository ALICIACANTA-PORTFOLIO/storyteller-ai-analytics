# 🧪 tests/ - Test Suite

## Purpose
Comprehensive test suite for AudioMind ensuring code quality, reliability, and correctness.

## Structure

```
tests/
├── __init__.py
├── conftest.py           # Pytest configuration & fixtures
├── unit/                 # Unit tests (80%+ coverage)
│   ├── test_whisper_processor.py
│   ├── test_topic_modeler.py
│   ├── test_llm_synthesizer.py
│   └── test_rag_indexer.py
├── integration/          # Integration tests (60%+ coverage)
│   ├── test_api_endpoints.py
│   ├── test_pipeline.py
│   └── test_database.py
├── e2e/                  # End-to-end tests
│   ├── test_full_pipeline.py
│   └── test_dashboard.py
└── fixtures/             # Test data and fixtures
    ├── audio/            # Sample audio files
    ├── transcriptions/   # Sample transcriptions
    └── responses/        # Mock API responses
```

## Test Categories

### Unit Tests (`tests/unit/`)
**Purpose**: Test individual components in isolation

**Coverage Target**: >80%

**Characteristics**:
- Fast execution (<1s per test)
- No external dependencies (mocked)
- Test single functions/methods
- Use pytest fixtures for setup

**Example**:
```python
def test_whisper_processor_transcribe(mock_whisper_model):
    """Test WhisperProcessor transcribes audio correctly."""
    processor = WhisperProcessor(model="base")
    result = processor.transcribe("test.mp3")
    
    assert result.text is not None
    assert result.language == "en"
    assert len(result.segments) > 0
```

### Integration Tests (`tests/integration/`)
**Purpose**: Test component interactions

**Coverage Target**: >60%

**Characteristics**:
- Moderate execution time (1-5s per test)
- May use real services (test database, etc.)
- Test multiple components working together
- Use Docker containers for services

**Example**:
```python
@pytest.mark.integration
async def test_full_processing_pipeline(test_db, test_audio_file):
    """Test complete pipeline from audio to insights."""
    pipeline = ProcessingPipeline()
    result = await pipeline.process(test_audio_file)
    
    assert result.transcription is not None
    assert len(result.topics) > 0
    assert result.insights is not None
```

### End-to-End Tests (`tests/e2e/`)
**Purpose**: Test complete user workflows

**Characteristics**:
- Slow execution (10-60s per test)
- Use real services
- Test from user perspective
- Cover critical paths only

**Example**:
```python
@pytest.mark.e2e
async def test_user_uploads_and_analyzes_audio(client, test_audio):
    """Test complete user journey: upload → analyze → view results."""
    # Upload
    response = await client.post("/upload", files={"file": test_audio})
    job_id = response.json()["job_id"]
    
    # Wait for processing
    result = await wait_for_completion(job_id, timeout=60)
    
    # Verify results
    assert result["status"] == "completed"
    assert "transcription" in result
    assert "topics" in result
```

## Test Fixtures (`tests/fixtures/`)

### Audio Fixtures
- `sample_short.mp3` - 10 second clip
- `sample_long.mp3` - 5 minute clip
- `sample_noisy.mp3` - Audio with background noise
- `sample_multilang.mp3` - Mixed languages

### Mock Data
- `mock_transcription.json` - Sample transcription
- `mock_topics.json` - Sample topic model output
- `mock_llm_response.json` - Sample GPT response

## Running Tests

### All Tests
```bash
pytest
```

### Specific Category
```bash
pytest tests/unit/              # Unit tests only
pytest tests/integration/       # Integration tests only
pytest tests/e2e/               # E2E tests only
```

### With Markers
```bash
pytest -m unit                  # Unit tests
pytest -m integration           # Integration tests
pytest -m "not slow"            # Skip slow tests
pytest -m asyncio               # Async tests only
```

### With Coverage
```bash
pytest --cov=app --cov-report=html
# View report: open htmlcov/index.html
```

### Parallel Execution
```bash
pytest -n auto  # Use all CPU cores
```

## Writing Tests

### Test Naming
- File: `test_<module>.py`
- Class: `Test<Component>`
- Function: `test_<scenario>_<expected_result>`

### Example Test Template
```python
"""
Tests for WhisperProcessor.

Test cases:
- test_transcribe_success - Happy path
- test_transcribe_invalid_file - Error handling
- test_transcribe_timeout - Timeout handling
"""

import pytest
from app.processors.whisper_processor import WhisperProcessor

@pytest.fixture
def processor():
    """Create WhisperProcessor instance for testing."""
    return WhisperProcessor(model="base")

def test_transcribe_success(processor, sample_audio_file):
    """Test successful transcription of valid audio."""
    # Arrange
    expected_language = "en"
    
    # Act
    result = processor.transcribe(sample_audio_file)
    
    # Assert
    assert result.text is not None
    assert result.language == expected_language
    assert len(result.segments) > 0

def test_transcribe_invalid_file(processor):
    """Test transcription fails gracefully with invalid file."""
    # Arrange
    invalid_file = "not_a_real_file.mp3"
    
    # Act & Assert
    with pytest.raises(FileNotFoundError):
        processor.transcribe(invalid_file)
```

## Mock Configuration

### API Responses
```python
@pytest.fixture
def mock_openai_response(monkeypatch):
    """Mock OpenAI API responses."""
    def mock_completion(*args, **kwargs):
        return {"choices": [{"message": {"content": "Mocked response"}}]}
    
    monkeypatch.setattr("openai.ChatCompletion.create", mock_completion)
```

### Database
```python
@pytest.fixture
async def test_db():
    """Create test database."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()
```

## Coverage Requirements

| Test Type | Coverage Target |
|-----------|----------------|
| Unit | >80% |
| Integration | >60% |
| Overall | >75% |

## Continuous Integration

Tests run automatically on:
- Every commit (pre-commit hooks)
- Pull requests (GitHub Actions)
- Before deployment

## Performance Benchmarks

Track test execution time:
```bash
pytest --durations=10  # Show 10 slowest tests
```

Expected durations:
- Unit tests: <1s each
- Integration tests: <5s each
- E2E tests: <60s each
- Full suite: <5 minutes

## Debugging Tests

### Run Single Test
```bash
pytest tests/unit/test_whisper_processor.py::test_transcribe_success
```

### With Print Statements
```bash
pytest -s tests/unit/test_whisper_processor.py
```

### With Debugger
```bash
pytest --pdb tests/unit/test_whisper_processor.py
```

## Best Practices

1. **Arrange-Act-Assert**: Use AAA pattern in all tests
2. **One Assertion**: Focus each test on one behavior
3. **Independent Tests**: No dependencies between tests
4. **Fast Tests**: Keep unit tests under 1 second
5. **Descriptive Names**: Test names should describe scenario
6. **Use Fixtures**: Reuse setup code via fixtures
7. **Mock External**: Always mock external APIs
8. **Test Edge Cases**: Include error scenarios

## Contributing

When adding new features:
1. Write tests first (TDD)
2. Achieve >80% coverage for unit tests
3. Add integration tests for complex interactions
4. Update this README with new test categories

---

**Last Updated**: October 23, 2025  
**Maintainer**: Alicia Canta
