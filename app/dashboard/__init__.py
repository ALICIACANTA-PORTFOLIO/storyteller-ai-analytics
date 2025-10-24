"""
Streamlit Dashboard for AudioMind.

Interactive web interface for:
- Audio file upload
- Real-time processing status
- Interactive visualizations
- Topic exploration
- RAG search
- Export functionality

Usage:
    >>> streamlit run app/dashboard/main.py
    
    Or from code:
    >>> from app.dashboard.main import main
    >>> main()

Features:
- File upload with drag-and-drop
- Live processing status
- Interactive topic visualization (pyLDAvis)
- Word clouds and charts
- Semantic search interface
- Export to PDF/CSV/JSON
"""

__all__ = ["main"]
