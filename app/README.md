# 📂 app/ - Application Source Code

## Purpose
Main application package containing all source code for AudioMind.

## Structure

```
app/
├── __init__.py           # Package initializer
├── main.py               # FastAPI application entry point
├── config.py             # Configuration management
├── dependencies.py       # FastAPI dependencies
├── models/               # Database models & schemas
├── processors/           # Audio/ML processing modules
├── api/                  # REST API implementation
├── dashboard/            # Streamlit dashboard
└── utils/                # Utility functions
```

## Modules

### `models/`
**Purpose**: Database models (SQLAlchemy) and Pydantic schemas

**Key Files**:
- `database.py` - Database connection and session management
- `audio.py` - Audio file metadata models
- `transcription.py` - Transcription result models
- `topic.py` - Topic modeling result models
- `schemas.py` - Pydantic request/response schemas

### `processors/`
**Purpose**: Core ML/AI processing pipeline components

**Key Files**:
- `whisper_processor.py` - Audio transcription with Whisper
- `topic_modeler.py` - Hybrid topic modeling (LDA + BERTopic)
- `llm_synthesizer.py` - LLM-powered summarization
- `rag_indexer.py` - RAG indexing and retrieval
- `pipeline.py` - Orchestrates 4-stage pipeline

**Processing Flow**:
1. Audio → WhisperProcessor → Transcription
2. Transcription → TopicModeler → Topics
3. Transcription + Topics → LLMSynthesizer → Insights
4. All data → RAGIndexer → Searchable knowledge base

### `api/`
**Purpose**: REST API endpoints for backend services

**Key Files**:
- `main.py` - FastAPI app with all routes
- `routes/upload.py` - File upload endpoints
- `routes/analysis.py` - Analysis trigger endpoints
- `routes/results.py` - Results retrieval endpoints
- `routes/search.py` - RAG search endpoints
- `dependencies.py` - Dependency injection

### `dashboard/`
**Purpose**: Interactive Streamlit web interface

**Key Files**:
- `main.py` - Streamlit app entry point
- `components/upload.py` - File upload widget
- `components/visualizations.py` - Charts and plots
- `components/search.py` - Search interface
- `components/export.py` - Export functionality

### `utils/`
**Purpose**: Shared utility functions

**Key Files**:
- `audio.py` - Audio file validation and preprocessing
- `text.py` - Text processing utilities
- `logging.py` - Logging configuration
- `metrics.py` - Performance metrics
- `validators.py` - Input validation

## Design Principles

1. **Separation of Concerns**: Each module has a single responsibility
2. **Dependency Injection**: Use FastAPI dependencies for testability
3. **Type Safety**: All functions have type hints
4. **Error Handling**: Comprehensive error handling with proper logging
5. **Async/Await**: Use async operations for I/O-bound tasks
6. **Configuration**: All settings in `config.py` (no hardcoded values)

## Coding Standards

- **Style**: PEP 8, enforced by Black (line length 100)
- **Linting**: Ruff for fast linting
- **Type Checking**: mypy in strict mode
- **Docstrings**: Google style with examples
- **Testing**: Pytest with >80% coverage requirement

## Example Usage

```python
from app.processors.whisper_processor import WhisperProcessor
from app.processors.topic_modeler import HybridTopicModeler
from app.processors.llm_synthesizer import LLMSynthesizer

# Initialize processors
whisper = WhisperProcessor(model="large-v3-turbo")
topic_modeler = HybridTopicModeler()
llm = LLMSynthesizer(model="gpt-4o-mini")

# Process audio
transcription = await whisper.transcribe("audio.mp3")
topics = await topic_modeler.extract_topics(transcription)
insights = await llm.synthesize(transcription, topics)
```

## Entry Points

- **API Server**: `python -m app.main` or `uvicorn app.main:app`
- **Dashboard**: `streamlit run app/dashboard/main.py`
- **Worker**: `celery -A app.worker worker`
- **CLI**: `audiomind --help`

## Dependencies

See `requirements.txt` for full list. Core dependencies:
- FastAPI + Uvicorn (API)
- Streamlit (Dashboard)
- Whisper (Transcription)
- Gensim + BERTopic (Topics)
- LangChain + OpenAI (LLM)
- ChromaDB (RAG)
- SQLAlchemy (Database)
- Celery + Redis (Task Queue)

## Testing

Run tests for this module:
```bash
pytest tests/unit/  # Unit tests
pytest tests/integration/  # Integration tests
pytest -m "not slow"  # Skip slow tests
```

## Contributing

1. Follow the coding standards above
2. Write tests for new features (>80% coverage)
3. Update this README if adding new modules
4. Run `black .`, `ruff check .`, `mypy app` before committing

---

**Last Updated**: October 23, 2025  
**Maintainer**: Alicia Canta
