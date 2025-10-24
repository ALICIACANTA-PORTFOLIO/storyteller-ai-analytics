"""
HybridTopicModeler - Extracción de tópicos usando LDA + BERTopic.

Diseño agnóstico: Funciona con cualquier texto.
Combina métodos clásicos (LDA) con modernos (BERTopic) para robustez.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import json

import numpy as np
import pandas as pd
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel
from gensim.parsing.preprocessing import (
    STOPWORDS,
    preprocess_string,
    strip_punctuation,
    strip_numeric,
    strip_multiple_whitespaces,
    strip_short,
    remove_stopwords,
    stem_text
)
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from app.config import TopicModelingConfig

logger = logging.getLogger(__name__)


# ============================================================================
# DATACLASSES PARA RESULTADOS
# ============================================================================

@dataclass
class Topic:
    """
    Representa un tópico extraído.
    
    Agnóstico al método de extracción.
    """
    topic_number: int
    label: str
    keywords: List[str]
    keyword_weights: Dict[str, float]
    relevance_score: float
    method_used: str  # 'lda', 'bertopic', 'hybrid'
    document_count: int = 0
    representative_docs: List[str] = field(default_factory=list)


@dataclass
class TopicModelingResult:
    """
    Resultado completo del modelado de tópicos.
    """
    topics: List[Topic]
    num_topics: int
    coherence_scores: Dict[str, float]
    processing_time_seconds: float
    method_used: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario serializable."""
        return {
            "topics": [
                {
                    "topic_number": t.topic_number,
                    "label": t.label,
                    "keywords": t.keywords,
                    "keyword_weights": t.keyword_weights,
                    "relevance_score": t.relevance_score,
                    "method_used": t.method_used,
                    "document_count": t.document_count,
                    "representative_docs": t.representative_docs[:3]  # Top 3
                }
                for t in self.topics
            ],
            "num_topics": self.num_topics,
            "coherence_scores": self.coherence_scores,
            "processing_time_seconds": self.processing_time_seconds,
            "method_used": self.method_used,
            "metadata": self.metadata
        }


# ============================================================================
# HYBRID TOPIC MODELER
# ============================================================================

class HybridTopicModeler:
    """
    Extractor híbrido de tópicos usando LDA + BERTopic.
    
    Características:
    - Agnóstico al dominio (funciona con cualquier texto)
    - Combina métodos clásicos (LDA) y modernos (BERTopic)
    - Detección automática del número óptimo de topics
    - Cálculo de coherencia para evaluación
    - Async/await para integración con FastAPI
    """
    
    def __init__(self, config: TopicModelingConfig):
        """
        Inicializa el modelador de tópicos.
        
        Args:
            config: Configuración de topic modeling
        """
        self.config = config
        self.lda_model: Optional[LdaModel] = None
        self.bert_model: Optional[BERTopic] = None
        self.dictionary: Optional[corpora.Dictionary] = None
        
        logger.info(
            f"🔍 HybridTopicModeler initialized\n"
            f"   Num topics: {config.num_topics or 'auto'}\n"
            f"   LDA iterations: {config.lda_iterations}\n"
            f"   BERTopic model: {config.bert_model}"
        )
    
    # ========================================================================
    # PREPROCESSING
    # ========================================================================
    
    def _preprocess_text(self, text: str) -> List[str]:
        """
        Preprocesa texto para topic modeling.
        
        Args:
            text: Texto a procesar
            
        Returns:
            Lista de tokens limpios
        """
        # Pipeline de gensim
        filters = [
            lambda x: x.lower(),
            strip_punctuation,
            strip_numeric,
            strip_multiple_whitespaces,
            remove_stopwords,
            strip_short,
        ]
        
        tokens = preprocess_string(text, filters)
        
        # Filtrar stopwords adicionales (idioma agnostico)
        custom_stopwords = {
            'audio', 'video', 'file', 'recording', 'track',
            'chapter', 'book', 'story', 'tale', 'narrative'
        }
        tokens = [t for t in tokens if t not in custom_stopwords]
        
        return tokens
    
    def _preprocess_documents(self, texts: List[str]) -> Tuple[List[List[str]], List[str]]:
        """
        Preprocesa múltiples documentos.
        
        Args:
            texts: Lista de textos
            
        Returns:
            Tupla de (documentos tokenizados, textos originales filtrados)
        """
        processed_docs = []
        valid_texts = []
        
        for text in texts:
            tokens = self._preprocess_text(text)
            if len(tokens) >= self.config.min_words_per_doc:
                processed_docs.append(tokens)
                valid_texts.append(text)
        
        logger.info(
            f"📄 Preprocessing complete\n"
            f"   Input documents: {len(texts)}\n"
            f"   Valid documents: {len(valid_texts)}\n"
            f"   Filtered out: {len(texts) - len(valid_texts)}"
        )
        
        return processed_docs, valid_texts
    
    # ========================================================================
    # LDA MODELING
    # ========================================================================
    
    def _train_lda(
        self,
        processed_docs: List[List[str]],
        num_topics: int
    ) -> Tuple[LdaModel, corpora.Dictionary]:
        """
        Entrena modelo LDA.
        
        Args:
            processed_docs: Documentos preprocesados
            num_topics: Número de tópicos
            
        Returns:
            Tupla de (modelo LDA, diccionario)
        """
        # Crear diccionario
        dictionary = corpora.Dictionary(processed_docs)
        
        # Filtrar extremos
        dictionary.filter_extremes(
            no_below=2,  # Mínimo 2 documentos
            no_above=0.5,  # Máximo 50% de documentos
            keep_n=100000
        )
        
        # Crear corpus
        corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
        
        logger.info(
            f"🎓 Training LDA model\n"
            f"   Topics: {num_topics}\n"
            f"   Dictionary size: {len(dictionary)}\n"
            f"   Corpus size: {len(corpus)}\n"
            f"   Iterations: {self.config.lda_iterations}"
        )
        
        # Entrenar LDA
        lda_model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            random_state=42,
            iterations=self.config.lda_iterations,
            alpha=self.config.lda_alpha,
            eta=self.config.lda_eta,
            per_word_topics=True
        )
        
        return lda_model, dictionary
    
    def _calculate_lda_coherence(
        self,
        lda_model: LdaModel,
        processed_docs: List[List[str]],
        dictionary: corpora.Dictionary
    ) -> Dict[str, float]:
        """
        Calcula coherencia del modelo LDA.
        
        Args:
            lda_model: Modelo LDA entrenado
            processed_docs: Documentos preprocesados
            dictionary: Diccionario de gensim
            
        Returns:
            Diccionario con scores de coherencia
        """
        if not self.config.calculate_coherence:
            return {}
        
        coherence_scores = {}
        
        for metric in self.config.coherence_metrics:
            try:
                coherence_model = CoherenceModel(
                    model=lda_model,
                    texts=processed_docs,
                    dictionary=dictionary,
                    coherence=metric
                )
                score = coherence_model.get_coherence()
                coherence_scores[f"lda_{metric}"] = score
                logger.info(f"   {metric}: {score:.4f}")
            except Exception as e:
                logger.warning(f"   Failed to calculate {metric}: {e}")
        
        return coherence_scores
    
    def _extract_lda_topics(
        self,
        lda_model: LdaModel,
        num_keywords: int = 10
    ) -> List[Topic]:
        """
        Extrae tópicos del modelo LDA.
        
        Args:
            lda_model: Modelo LDA entrenado
            num_keywords: Número de keywords por tópico
            
        Returns:
            Lista de Topics
        """
        topics = []
        
        for topic_id in range(lda_model.num_topics):
            # Obtener top words con pesos
            topic_words = lda_model.show_topic(topic_id, topn=num_keywords)
            
            keywords = [word for word, _ in topic_words]
            keyword_weights = {word: float(weight) for word, weight in topic_words}
            
            # Crear label
            label = f"Topic {topic_id + 1}: {', '.join(keywords[:3])}"
            
            # Relevance score = promedio de pesos
            relevance_score = float(np.mean([w for _, w in topic_words]))
            
            topic = Topic(
                topic_number=topic_id,
                label=label,
                keywords=keywords,
                keyword_weights=keyword_weights,
                relevance_score=relevance_score,
                method_used="lda"
            )
            topics.append(topic)
        
        return topics
    
    # ========================================================================
    # BERTOPIC MODELING
    # ========================================================================
    
    def _train_bertopic(
        self,
        texts: List[str],
        num_topics: Optional[int] = None
    ) -> BERTopic:
        """
        Entrena modelo BERTopic.
        
        Args:
            texts: Textos originales
            num_topics: Número de tópicos (None = automático)
            
        Returns:
            Modelo BERTopic entrenado
        """
        logger.info(
            f"🤖 Training BERTopic model\n"
            f"   Embedding model: {self.config.bert_model}\n"
            f"   Min topic size: {self.config.bert_min_topic_size}\n"
            f"   Diversity: {self.config.bert_diversity}"
        )
        
        # Configurar vectorizer
        vectorizer = CountVectorizer(
            stop_words="english",
            min_df=2,
            max_df=0.95
        )
        
        # Crear modelo
        bert_model = BERTopic(
            embedding_model=self.config.bert_model,
            min_topic_size=self.config.bert_min_topic_size,
            nr_topics=num_topics,  # None = automático
            calculate_probabilities=True,
            vectorizer_model=vectorizer,
            verbose=True
        )
        
        # Entrenar
        bert_model.fit_transform(texts)
        
        logger.info(f"   Topics found: {len(bert_model.get_topic_info()) - 1}")  # -1 para outliers
        
        return bert_model
    
    def _extract_bertopic_topics(
        self,
        bert_model: BERTopic,
        num_keywords: int = 10
    ) -> List[Topic]:
        """
        Extrae tópicos del modelo BERTopic.
        
        Args:
            bert_model: Modelo BERTopic entrenado
            num_keywords: Número de keywords por tópico
            
        Returns:
            Lista de Topics
        """
        topics = []
        topic_info = bert_model.get_topic_info()
        
        # Filtrar outliers (topic -1)
        topic_info = topic_info[topic_info['Topic'] != -1]
        
        for _, row in topic_info.iterrows():
            topic_id = int(row['Topic'])
            
            # Obtener keywords con scores
            topic_words = bert_model.get_topic(topic_id)
            if not topic_words:
                continue
            
            keywords = [word for word, _ in topic_words[:num_keywords]]
            keyword_weights = {word: float(score) for word, score in topic_words[:num_keywords]}
            
            # Label desde BERTopic
            label = row.get('Name', f"Topic {topic_id}")
            
            # Document count
            doc_count = int(row.get('Count', 0))
            
            # Representative docs
            repr_docs = row.get('Representative_Docs', [])
            if isinstance(repr_docs, list):
                representative_docs = [str(doc)[:200] for doc in repr_docs[:3]]
            else:
                representative_docs = []
            
            # Relevance score
            relevance_score = float(np.mean([s for _, s in topic_words[:num_keywords]]))
            
            topic = Topic(
                topic_number=topic_id,
                label=label,
                keywords=keywords,
                keyword_weights=keyword_weights,
                relevance_score=relevance_score,
                method_used="bertopic",
                document_count=doc_count,
                representative_docs=representative_docs
            )
            topics.append(topic)
        
        return topics
    
    # ========================================================================
    # OPTIMAL TOPIC NUMBER DETECTION
    # ========================================================================
    
    def _find_optimal_topics_lda(
        self,
        processed_docs: List[List[str]],
        dictionary: corpora.Dictionary
    ) -> int:
        """
        Encuentra número óptimo de tópicos usando coherencia LDA.
        
        Args:
            processed_docs: Documentos preprocesados
            dictionary: Diccionario de gensim
            
        Returns:
            Número óptimo de tópicos
        """
        logger.info(
            f"🔎 Finding optimal number of topics\n"
            f"   Range: {self.config.min_topics} to {self.config.max_topics}"
        )
        
        corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
        coherence_scores = []
        
        for num_topics in range(self.config.min_topics, self.config.max_topics + 1):
            # Entrenar modelo temporal
            lda = LdaModel(
                corpus=corpus,
                id2word=dictionary,
                num_topics=num_topics,
                random_state=42,
                iterations=100,  # Reducido para velocidad
                alpha='auto',
                eta='auto'
            )
            
            # Calcular coherencia
            coherence_model = CoherenceModel(
                model=lda,
                texts=processed_docs,
                dictionary=dictionary,
                coherence='c_v'
            )
            score = coherence_model.get_coherence()
            coherence_scores.append(score)
            
            logger.info(f"   {num_topics} topics: coherence = {score:.4f}")
        
        # Encontrar máximo
        optimal_idx = np.argmax(coherence_scores)
        optimal_topics = self.config.min_topics + optimal_idx
        
        logger.info(f"✅ Optimal number of topics: {optimal_topics}")
        
        return optimal_topics
    
    # ========================================================================
    # HYBRID EXTRACTION
    # ========================================================================
    
    async def extract_topics(
        self,
        text: str,
        num_topics: Optional[int] = None,
        method: str = "hybrid"
    ) -> TopicModelingResult:
        """
        Extrae tópicos de un texto.
        
        Args:
            text: Texto a analizar (puede ser transcripción completa)
            num_topics: Número de tópicos (None = detección automática)
            method: 'lda', 'bertopic', o 'hybrid'
            
        Returns:
            TopicModelingResult con tópicos extraídos
        """
        start_time = time.time()
        
        # Dividir texto en "documentos" (párrafos o segmentos)
        documents = self._split_into_documents(text)
        
        logger.info(
            f"\n{'='*70}\n"
            f"🔍 TOPIC MODELING\n"
            f"{'='*70}\n"
            f"Method: {method}\n"
            f"Documents: {len(documents)}\n"
            f"Num topics: {num_topics or 'auto'}"
        )
        
        # Preprocesar
        processed_docs, valid_texts = await asyncio.to_thread(
            self._preprocess_documents,
            documents
        )
        
        if len(processed_docs) < self.config.min_topics:
            logger.warning(
                f"⚠️ Not enough documents ({len(processed_docs)}) for topic modeling. "
                f"Minimum required: {self.config.min_topics}"
            )
            return TopicModelingResult(
                topics=[],
                num_topics=0,
                coherence_scores={},
                processing_time_seconds=time.time() - start_time,
                method_used=method,
                metadata={"error": "Insufficient documents"}
            )
        
        # Determinar número de tópicos
        if num_topics is None:
            num_topics = self.config.num_topics
        
        topics: List[Topic] = []
        coherence_scores: Dict[str, float] = {}
        
        # MÉTODO HÍBRIDO
        if method == "hybrid":
            # 1. LDA
            logger.info("\n--- LDA Phase ---")
            lda_model, dictionary = await asyncio.to_thread(
                self._train_lda,
                processed_docs,
                num_topics or self.config.min_topics
            )
            
            lda_coherence = await asyncio.to_thread(
                self._calculate_lda_coherence,
                lda_model,
                processed_docs,
                dictionary
            )
            coherence_scores.update(lda_coherence)
            
            lda_topics = self._extract_lda_topics(lda_model)
            
            # 2. BERTopic
            logger.info("\n--- BERTopic Phase ---")
            bert_model = await asyncio.to_thread(
                self._train_bertopic,
                valid_texts,
                num_topics
            )
            
            bert_topics = self._extract_bertopic_topics(bert_model)
            
            # 3. Combinar (priorizar BERTopic por document counts)
            topics = bert_topics if bert_topics else lda_topics
            
            # Guardar modelos
            self.lda_model = lda_model
            self.bert_model = bert_model
            self.dictionary = dictionary
        
        # SOLO LDA
        elif method == "lda":
            if num_topics is None:
                num_topics = await asyncio.to_thread(
                    self._find_optimal_topics_lda,
                    processed_docs,
                    corpora.Dictionary(processed_docs)
                )
            
            lda_model, dictionary = await asyncio.to_thread(
                self._train_lda,
                processed_docs,
                num_topics
            )
            
            coherence_scores = await asyncio.to_thread(
                self._calculate_lda_coherence,
                lda_model,
                processed_docs,
                dictionary
            )
            
            topics = self._extract_lda_topics(lda_model)
            
            self.lda_model = lda_model
            self.dictionary = dictionary
        
        # SOLO BERTOPIC
        elif method == "bertopic":
            bert_model = await asyncio.to_thread(
                self._train_bertopic,
                valid_texts,
                num_topics
            )
            
            topics = self._extract_bertopic_topics(bert_model)
            
            self.bert_model = bert_model
        
        processing_time = time.time() - start_time
        
        # Resultado
        result = TopicModelingResult(
            topics=topics,
            num_topics=len(topics),
            coherence_scores=coherence_scores,
            processing_time_seconds=processing_time,
            method_used=method,
            metadata={
                "input_documents": len(documents),
                "valid_documents": len(valid_texts),
                "timestamp": datetime.now().isoformat()
            }
        )
        
        logger.info(
            f"\n{'='*70}\n"
            f"✅ TOPIC MODELING COMPLETE\n"
            f"{'='*70}\n"
            f"Topics extracted: {result.num_topics}\n"
            f"Processing time: {processing_time:.2f}s\n"
            f"Coherence scores: {coherence_scores}"
        )
        
        return result
    
    def _split_into_documents(self, text: str) -> List[str]:
        """
        Divide texto en documentos (párrafos o segmentos).
        
        Args:
            text: Texto completo
            
        Returns:
            Lista de documentos
        """
        # Split por párrafos (doble salto de línea)
        documents = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        # Si hay pocos párrafos, split por oraciones
        if len(documents) < 5:
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            # Agrupar oraciones en chunks de ~3 oraciones
            chunk_size = 3
            documents = [
                '. '.join(sentences[i:i+chunk_size])
                for i in range(0, len(sentences), chunk_size)
            ]
        
        return documents
    
    # ========================================================================
    # EXPORT & PERSISTENCE
    # ========================================================================
    
    def save_result(self, result: TopicModelingResult, output_path: Path):
        """
        Guarda resultado en JSON.
        
        Args:
            result: Resultado a guardar
            output_path: Ruta del archivo
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Results saved to: {output_path}")
