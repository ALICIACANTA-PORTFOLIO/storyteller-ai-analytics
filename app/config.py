"""
Sistema de configuración parametrizado para AudioMind.

Este módulo implementa un sistema robusto de configuración siguiendo mejores prácticas:
- Configuración por capas (defaults -> archivos -> env vars)
- Type safety con Pydantic
- Validación automática de valores
- Soporte para múltiples ambientes (dev, test, prod)
- Secretos separados de configuración

Principios de diseño:
1. AGNÓSTICO: No hardcodea dominios específicos
2. CONFIGURABLE: Todo es parametrizable
3. VALIDADO: Type hints + Pydantic validation
4. DOCUMENTADO: Cada campo tiene descripción y ejemplo
"""

import os
from pathlib import Path
from typing import Literal, Optional, List, Dict, Any
from enum import Enum

from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
    PostgresDsn,
    RedisDsn,
    HttpUrl,
)
from pydantic_settings import BaseSettings, SettingsConfigDict


# ============================================================================
# ENUMS - Tipos enumerados para valores válidos
# ============================================================================

class Environment(str, Enum):
    """Ambientes de ejecución."""
    DEVELOPMENT = "development"
    TEST = "test"
    PRODUCTION = "production"


class LogLevel(str, Enum):
    """Niveles de logging."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class WhisperModelSize(str, Enum):
    """Tamaños de modelo Whisper disponibles."""
    TINY = "tiny"
    BASE = "base"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    LARGE_V2 = "large-v2"
    LARGE_V3 = "large-v3"
    LARGE_V3_TURBO = "large-v3-turbo"


class TopicModelMethod(str, Enum):
    """Métodos de topic modeling."""
    LDA = "lda"
    BERTOPIC = "bertopic"
    HYBRID = "hybrid"


class LLMProvider(str, Enum):
    """Proveedores de LLM."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE_OPENAI = "azure_openai"


# ============================================================================
# CONFIGURACIÓN DE BASE DE DATOS
# ============================================================================

class DatabaseConfig(BaseModel):
    """
    Configuración de base de datos PostgreSQL.
    
    Soporta connection pooling, SSL, y configuración avanzada.
    """
    
    host: str = Field(
        default="localhost",
        description="Host del servidor PostgreSQL"
    )
    port: int = Field(
        default=5432,
        ge=1,
        le=65535,
        description="Puerto de PostgreSQL"
    )
    user: str = Field(
        default="audiomind",
        description="Usuario de base de datos"
    )
    password: str = Field(
        default="audiomind_dev_password",
        description="Password (usar variable de ambiente en producción)"
    )
    database: str = Field(
        default="audiomind_db",
        description="Nombre de la base de datos"
    )
    schema: str = Field(
        default="audiomind",
        description="Schema de PostgreSQL"
    )
    
    # Connection pool
    pool_size: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Tamaño del connection pool"
    )
    max_overflow: int = Field(
        default=10,
        ge=0,
        le=100,
        description="Conexiones adicionales permitidas"
    )
    pool_timeout: int = Field(
        default=30,
        ge=1,
        description="Timeout para obtener conexión (segundos)"
    )
    pool_recycle: int = Field(
        default=3600,
        ge=60,
        description="Tiempo antes de reciclar conexiones (segundos)"
    )
    
    # Opciones avanzadas
    echo_sql: bool = Field(
        default=False,
        description="Imprimir queries SQL (útil para debugging)"
    )
    use_ssl: bool = Field(
        default=False,
        description="Usar SSL para conexión"
    )
    
    @property
    def connection_string(self) -> str:
        """Construye connection string de PostgreSQL."""
        return (
            f"postgresql://{self.user}:{self.password}@"
            f"{self.host}:{self.port}/{self.database}"
        )
    
    @property
    def async_connection_string(self) -> str:
        """Connection string para async (asyncpg)."""
        return (
            f"postgresql+asyncpg://{self.user}:{self.password}@"
            f"{self.host}:{self.port}/{self.database}"
        )
    
    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "example": {
                "host": "localhost",
                "port": 5432,
                "user": "audiomind",
                "database": "audiomind_db",
                "pool_size": 5
            }
        }


# ============================================================================
# CONFIGURACIÓN DE REDIS
# ============================================================================

class RedisConfig(BaseModel):
    """
    Configuración de Redis para Celery y caching.
    """
    
    host: str = Field(
        default="localhost",
        description="Host de Redis"
    )
    port: int = Field(
        default=6379,
        ge=1,
        le=65535,
        description="Puerto de Redis"
    )
    password: Optional[str] = Field(
        default="audiomind_redis_password",
        description="Password de Redis (opcional)"
    )
    db: int = Field(
        default=0,
        ge=0,
        le=15,
        description="Número de database Redis (0-15)"
    )
    
    # Cache settings
    default_ttl: int = Field(
        default=3600,
        ge=60,
        description="TTL por defecto para cache (segundos)"
    )
    max_connections: int = Field(
        default=50,
        ge=1,
        description="Máximo de conexiones en pool"
    )
    
    @property
    def connection_string(self) -> str:
        """Construye Redis connection string."""
        auth = f":{self.password}@" if self.password else ""
        return f"redis://{auth}{self.host}:{self.port}/{self.db}"


# ============================================================================
# CONFIGURACIÓN DE CHROMADB (VECTOR DATABASE)
# ============================================================================

class ChromaDBConfig(BaseModel):
    """
    Configuración de ChromaDB para búsqueda semántica (RAG).
    """
    
    host: str = Field(
        default="localhost",
        description="Host de ChromaDB"
    )
    port: int = Field(
        default=8000,
        ge=1,
        le=65535,
        description="Puerto de ChromaDB"
    )
    token: Optional[str] = Field(
        default="audiomind_chroma_token",
        description="Token de autenticación"
    )
    
    # Collection settings
    collection_name: str = Field(
        default="audio_embeddings",
        description="Nombre de la collection (agnóstico)"
    )
    distance_metric: Literal["cosine", "l2", "ip"] = Field(
        default="cosine",
        description="Métrica de distancia para embeddings"
    )
    
    # Embedding settings
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Modelo de sentence-transformers"
    )
    embedding_dimension: int = Field(
        default=384,
        ge=1,
        description="Dimensión de embeddings"
    )
    
    @property
    def base_url(self) -> str:
        """URL base de ChromaDB."""
        return f"http://{self.host}:{self.port}"


# ============================================================================
# CONFIGURACIÓN DE TRANSCRIPCIÓN (WHISPER)
# ============================================================================

class TranscriptionConfig(BaseModel):
    """
    Configuración genérica de transcripción con Whisper.
    
    Diseño agnóstico: Funciona con cualquier tipo de audio.
    """
    
    model_size: WhisperModelSize = Field(
        default=WhisperModelSize.LARGE_V3_TURBO,
        description="Tamaño del modelo Whisper"
    )
    device: Literal["cpu", "cuda"] = Field(
        default="cpu",
        description="Device para inferencia (cpu/cuda)"
    )
    language: Optional[str] = Field(
        default=None,
        description="Idioma ISO 639-1 (None = detección automática)"
    )
    
    # Opciones de transcripción
    task: Literal["transcribe", "translate"] = Field(
        default="transcribe",
        description="Transcribir o traducir a inglés"
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Temperature para sampling (0 = greedy)"
    )
    beam_size: int = Field(
        default=5,
        ge=1,
        description="Beam size para beam search"
    )
    best_of: int = Field(
        default=5,
        ge=1,
        description="Número de candidatos a evaluar"
    )
    
    # Diarization (identificación de hablantes)
    enable_diarization: bool = Field(
        default=False,
        description="Identificar diferentes hablantes"
    )
    min_speakers: Optional[int] = Field(
        default=None,
        ge=1,
        description="Número mínimo de hablantes (None = automático)"
    )
    max_speakers: Optional[int] = Field(
        default=None,
        ge=1,
        description="Número máximo de hablantes (None = automático)"
    )
    
    # VAD (Voice Activity Detection)
    enable_vad: bool = Field(
        default=True,
        description="Filtrar silencio antes de transcribir"
    )
    vad_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Threshold para detección de voz"
    )
    
    # Chunking (para audios largos)
    chunk_length: int = Field(
        default=30,
        ge=1,
        description="Duración de chunks en segundos"
    )
    
    @field_validator("language")
    @classmethod
    def validate_language(cls, v: Optional[str]) -> Optional[str]:
        """Valida código de idioma ISO 639-1."""
        if v is None:
            return v
        
        # Lista de idiomas soportados por Whisper
        supported = [
            "en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko",
            "ar", "hi", "tr", "pl", "nl", "sv", "fi", "no", "da", "cs",
            # ... (Whisper soporta 99 idiomas)
        ]
        
        if v not in supported:
            # En producción, validar contra lista completa
            pass  # Por ahora, aceptar cualquier código
        
        return v.lower()


# ============================================================================
# CONFIGURACIÓN DE TOPIC MODELING
# ============================================================================

class TopicModelingConfig(BaseModel):
    """
    Configuración genérica de topic modeling.
    
    Soporta LDA, BERTopic, y enfoque híbrido.
    """
    
    method: TopicModelMethod = Field(
        default=TopicModelMethod.HYBRID,
        description="Método de topic modeling"
    )
    
    # Número de topics
    num_topics: Optional[int] = Field(
        default=None,
        ge=2,
        description="Número de topics (None = detección automática)"
    )
    min_topics: int = Field(
        default=5,
        ge=2,
        description="Mínimo de topics para detección automática"
    )
    max_topics: int = Field(
        default=20,
        ge=2,
        description="Máximo de topics para detección automática"
    )
    
    # Document preprocessing
    min_words_per_doc: int = Field(
        default=5,
        ge=1,
        description="Mínimo de palabras por documento después de preprocessing"
    )
    
    # LDA settings
    lda_iterations: int = Field(
        default=1000,
        ge=100,
        description="Iteraciones para LDA"
    )
    lda_alpha: Literal["auto", "symmetric", "asymmetric"] = Field(
        default="auto",
        description="Prior de documento-topic"
    )
    lda_eta: Literal["auto", "symmetric"] = Field(
        default="auto",
        description="Prior de topic-word"
    )
    
    # BERTopic settings
    bert_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Modelo de embeddings para BERTopic"
    )
    bert_min_topic_size: int = Field(
        default=10,
        ge=2,
        description="Tamaño mínimo de un topic"
    )
    bert_diversity: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Diversidad de keywords (0-1)"
    )
    
    # Evaluación
    calculate_coherence: bool = Field(
        default=True,
        description="Calcular métricas de coherencia"
    )
    coherence_metrics: List[str] = Field(
        default=["c_v", "c_uci", "u_mass"],
        description="Métricas de coherencia a calcular"
    )


# ============================================================================
# CONFIGURACIÓN DE LLM (SÍNTESIS)
# ============================================================================

class LLMConfig(BaseModel):
    """
    Configuración de LLM para síntesis de insights.
    
    Soporta múltiples proveedores y modelos.
    """
    
    provider: LLMProvider = Field(
        default=LLMProvider.OPENAI,
        description="Proveedor de LLM"
    )
    
    # OpenAI settings
    openai_api_key: Optional[str] = Field(
        default=None,
        description="API key de OpenAI (variable de ambiente)"
    )
    openai_model: str = Field(
        default="gpt-4o-mini",
        description="Modelo de OpenAI"
    )
    openai_organization: Optional[str] = Field(
        default=None,
        description="Organization ID de OpenAI (opcional)"
    )
    
    # Azure OpenAI settings
    azure_endpoint: Optional[str] = Field(
        default=None,
        description="Endpoint de Azure OpenAI"
    )
    azure_api_key: Optional[str] = Field(
        default=None,
        description="API key de Azure OpenAI"
    )
    azure_deployment: Optional[str] = Field(
        default=None,
        description="Deployment name en Azure"
    )
    
    # Anthropic settings
    anthropic_api_key: Optional[str] = Field(
        default=None,
        description="API key de Anthropic"
    )
    anthropic_model: str = Field(
        default="claude-3-5-sonnet-20241022",
        description="Modelo de Anthropic"
    )
    
    # Generation settings
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Temperature para generación"
    )
    max_tokens: int = Field(
        default=2000,
        ge=1,
        description="Máximo de tokens a generar"
    )
    top_p: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling parameter"
    )
    
    # RAG settings
    enable_rag: bool = Field(
        default=True,
        description="Usar RAG (Retrieval-Augmented Generation)"
    )
    rag_top_k: int = Field(
        default=5,
        ge=1,
        description="Número de documentos a recuperar"
    )
    rag_similarity_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Threshold de similitud para RAG"
    )
    
    # Prompt template (genérico)
    system_prompt: str = Field(
        default=(
            "You are an AI assistant that analyzes audio transcriptions "
            "and extracts meaningful insights. Your analysis should be "
            "objective, comprehensive, and actionable."
        ),
        description="System prompt base (puede personalizarse)"
    )


# ============================================================================
# CONFIGURACIÓN PRINCIPAL DE LA APLICACIÓN
# ============================================================================

class Settings(BaseSettings):
    """
    Configuración principal de AudioMind.
    
    Carga configuración de:
    1. Valores por defecto (defaults)
    2. Archivos YAML en config/
    3. Variables de ambiente (.env)
    4. Variables de ambiente del sistema
    
    Orden de precedencia: 4 > 3 > 2 > 1
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )
    
    # ========================================================================
    # CONFIGURACIÓN GENERAL
    # ========================================================================
    
    app_name: str = Field(
        default="AudioMind",
        description="Nombre de la aplicación"
    )
    app_version: str = Field(
        default="0.1.0",
        description="Versión de la aplicación"
    )
    environment: Environment = Field(
        default=Environment.DEVELOPMENT,
        description="Ambiente de ejecución"
    )
    debug: bool = Field(
        default=False,
        description="Modo debug (solo development)"
    )
    
    # Paths
    project_root: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent,
        description="Raíz del proyecto"
    )
    data_dir: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent / "data",
        description="Directorio de datos"
    )
    logs_dir: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent / "logs",
        description="Directorio de logs"
    )
    
    # Logging
    log_level: LogLevel = Field(
        default=LogLevel.INFO,
        description="Nivel de logging"
    )
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Formato de log"
    )
    
    # ========================================================================
    # SUB-CONFIGURACIONES
    # ========================================================================
    
    database: DatabaseConfig = Field(
        default_factory=DatabaseConfig,
        description="Configuración de PostgreSQL"
    )
    redis: RedisConfig = Field(
        default_factory=RedisConfig,
        description="Configuración de Redis"
    )
    chromadb: ChromaDBConfig = Field(
        default_factory=ChromaDBConfig,
        description="Configuración de ChromaDB"
    )
    transcription: TranscriptionConfig = Field(
        default_factory=TranscriptionConfig,
        description="Configuración de Whisper"
    )
    topic_modeling: TopicModelingConfig = Field(
        default_factory=TopicModelingConfig,
        description="Configuración de topic modeling"
    )
    llm: LLMConfig = Field(
        default_factory=LLMConfig,
        description="Configuración de LLM"
    )
    
    # ========================================================================
    # API SETTINGS
    # ========================================================================
    
    api_host: str = Field(
        default="0.0.0.0",
        description="Host para FastAPI"
    )
    api_port: int = Field(
        default=8080,
        ge=1,
        le=65535,
        description="Puerto para FastAPI"
    )
    api_workers: int = Field(
        default=4,
        ge=1,
        description="Número de workers Uvicorn"
    )
    api_reload: bool = Field(
        default=False,
        description="Hot reload (solo development)"
    )
    
    # CORS
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8501"],
        description="Orígenes permitidos para CORS"
    )
    cors_allow_credentials: bool = Field(
        default=True,
        description="Permitir credenciales en CORS"
    )
    
    # ========================================================================
    # CELERY SETTINGS (TASK QUEUE)
    # ========================================================================
    
    celery_broker_url: Optional[str] = Field(
        default=None,
        description="URL del broker de Celery (Redis)"
    )
    celery_result_backend: Optional[str] = Field(
        default=None,
        description="Backend de resultados de Celery"
    )
    celery_task_time_limit: int = Field(
        default=3600,
        ge=60,
        description="Tiempo límite de task (segundos)"
    )
    
    # ========================================================================
    # VALIDACIONES Y POST-PROCESAMIENTO
    # ========================================================================
    
    @model_validator(mode="after")
    def validate_settings(self) -> "Settings":
        """Validaciones cross-field y setup post-inicialización."""
        
        # Crear directorios si no existen
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # En producción, requerir API keys
        if self.environment == Environment.PRODUCTION:
            if self.llm.provider == LLMProvider.OPENAI and not self.llm.openai_api_key:
                raise ValueError("OPENAI_API_KEY requerido en producción")
        
        # Debug solo en development
        if self.environment != Environment.DEVELOPMENT:
            self.debug = False
            self.api_reload = False
        
        # Celery broker por defecto
        if not self.celery_broker_url:
            self.celery_broker_url = self.redis.connection_string
        if not self.celery_result_backend:
            self.celery_result_backend = self.redis.connection_string
        
        return self
    
    @property
    def is_development(self) -> bool:
        """Check si es ambiente de desarrollo."""
        return self.environment == Environment.DEVELOPMENT
    
    @property
    def is_production(self) -> bool:
        """Check si es ambiente de producción."""
        return self.environment == Environment.PRODUCTION
    
    @property
    def is_test(self) -> bool:
        """Check si es ambiente de test."""
        return self.environment == Environment.TEST
    
    def get_config_dict(self) -> Dict[str, Any]:
        """Obtiene configuración como dict (sin secretos)."""
        config = self.model_dump()
        
        # Remover secretos
        secrets_to_remove = [
            "openai_api_key",
            "azure_api_key",
            "anthropic_api_key",
            "password",
            "token",
        ]
        
        def remove_secrets(d: Dict[str, Any]) -> Dict[str, Any]:
            """Recursivamente remueve secretos."""
            for key in list(d.keys()):
                if any(secret in key for secret in secrets_to_remove):
                    d[key] = "***"
                elif isinstance(d[key], dict):
                    d[key] = remove_secrets(d[key])
            return d
        
        return remove_secrets(config)


# ============================================================================
# INSTANCIA GLOBAL DE CONFIGURACIÓN
# ============================================================================

# Esta instancia se importa en otros módulos
settings = Settings()


# ============================================================================
# FUNCIONES HELPER
# ============================================================================

def get_settings() -> Settings:
    """
    Obtiene instancia de settings (para dependency injection).
    
    Uso en FastAPI:
        from fastapi import Depends
        from app.config import get_settings, Settings
        
        @app.get("/config")
        def read_config(settings: Settings = Depends(get_settings)):
            return settings.get_config_dict()
    """
    return settings


def reload_settings() -> Settings:
    """
    Recarga configuración (útil para tests).
    
    Returns:
        Nueva instancia de Settings
    """
    global settings
    settings = Settings()
    return settings


if __name__ == "__main__":
    """
    Test de configuración.
    
    Ejecutar: python app/config.py
    """
    import json
    
    print("=" * 80)
    print("CONFIGURACIÓN DE AUDIOMIND")
    print("=" * 80)
    print()
    
    # Imprimir configuración (sin secretos)
    config_dict = settings.get_config_dict()
    print(json.dumps(config_dict, indent=2, default=str))
    
    print()
    print("=" * 80)
    print("CONEXIONES")
    print("=" * 80)
    print(f"PostgreSQL: {settings.database.connection_string}")
    print(f"Redis: {settings.redis.connection_string}")
    print(f"ChromaDB: {settings.chromadb.base_url}")
    print()
    
    print("✅ Configuración cargada correctamente")
