"""
Audio/ML Processing Modules for AudioMind.

This package contains the core processing pipeline:

1. WhisperProcessor: Audio → Transcription
2. TopicModeler: Text → Topics (LDA + BERTopic)
3. LLMSynthesizer: Text + Topics → Insights
4. RAGIndexer: All Data → Searchable Knowledge Base

Usage:
    >>> from app.processors import WhisperProcessor, HybridTopicModeler
    >>> 
    >>> # Transcribe audio
    >>> whisper = WhisperProcessor(model="large-v3-turbo")
    >>> transcription = await whisper.transcribe("audio.mp3")
    >>> 
    >>> # Extract topics
    >>> topic_modeler = HybridTopicModeler()
    >>> topics = await topic_modeler.extract_topics(transcription.text)
    >>> 
    >>> # Generate insights
    >>> synthesizer = LLMSynthesizer()
    >>> insights = await synthesizer.synthesize(transcription, topics)

Pipeline Orchestration:
    >>> from app.processors.pipeline import ProcessingPipeline
    >>> 
    >>> pipeline = ProcessingPipeline()
    >>> result = await pipeline.process("audio.mp3")
    >>> # Returns: {transcription, topics, insights, rag_indexed}
"""

# Import processors with lazy loading to avoid dependency issues
__all__ = [
    # Transcription
    "WhisperProcessor",
    "TranscriptionResult",
    "TranscriptionSegment",
    # Topic Modeling
    "HybridTopicModeler",
    "Topic",
    "TopicModelingResult",
    # LLM Synthesis (TODO)
    "LLMSynthesizer",
    # RAG Indexing (TODO)
    "RAGIndexer",
    # Pipeline (TODO)
    "ProcessingPipeline",
]


def __getattr__(name: str):
    """Lazy import to avoid loading heavy dependencies unless needed."""
    if name == "WhisperProcessor":
        from .whisper_processor import WhisperProcessor
        return WhisperProcessor
    elif name == "TranscriptionResult":
        from .whisper_processor import TranscriptionResult
        return TranscriptionResult
    elif name == "TranscriptionSegment":
        from .whisper_processor import TranscriptionSegment
        return TranscriptionSegment
    elif name == "HybridTopicModeler":
        from .topic_modeler import HybridTopicModeler
        return HybridTopicModeler
    elif name == "Topic":
        from .topic_modeler import Topic
        return Topic
    elif name == "TopicModelingResult":
        from .topic_modeler import TopicModelingResult
        return TopicModelingResult
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
