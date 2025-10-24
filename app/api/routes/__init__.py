"""
API Routes for AudioMind.

Individual route modules organized by functionality:
- upload.py: File upload endpoints
- analysis.py: Analysis trigger and status
- results.py: Retrieve analysis results
- search.py: RAG search endpoints
- health.py: Health and monitoring endpoints

Each module contains related endpoints with proper documentation.
"""

__all__ = ["upload_router", "analysis_router", "results_router", "search_router", "health_router"]
