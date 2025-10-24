"""
FastAPI REST API for AudioMind.

This package contains:
- API routes and endpoints
- Request/response models
- Authentication and authorization
- Rate limiting
- Error handling

Endpoints:
    - POST /upload: Upload audio file
    - GET /jobs/{job_id}: Get job status
    - GET /results/{job_id}: Get analysis results
    - POST /search: RAG search query
    - GET /health: Health check

Usage:
    Start API server:
    >>> uvicorn app.api.main:app --reload
    
    Or programmatically:
    >>> from app.api.main import app
    >>> import uvicorn
    >>> uvicorn.run(app, host="0.0.0.0", port=8000)
"""

__all__ = ["app"]
