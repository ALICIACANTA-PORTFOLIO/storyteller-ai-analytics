# 🎯 FILOSOFÍA DE DISEÑO: SISTEMA AGNÓSTICO

> **Este documento define el principio fundamental de AudioMind**

---

## 💡 Principio Core

**AudioMind es agnóstico al dominio - funciona con CUALQUIER tipo de audio**

NO es:
- ❌ Una herramienta para analizar podcasts
- ❌ Una plataforma para entrevistas de clientes
- ❌ Un sistema de análisis educativo

ES:
- ✅ **Una plataforma genérica de inteligencia de audio**
- ✅ **Configurable para cualquier caso de uso**
- ✅ **Extensible sin modificar código core**

---

## 🚫 Anti-Patrones (Lo que NO hacer)

### ❌ Hardcodear Dominios Específicos

```python
# ❌ MAL - Acoplado a casos específicos
class AudioAnalyzer:
    def analyze_podcast(self, audio):
        """Analiza un podcast."""
        topics = self.extract_podcast_topics(audio)
        return PodcastInsights(...)
    
    def analyze_interview(self, audio):
        """Analiza una entrevista."""
        pain_points = self.extract_pain_points(audio)
        return InterviewInsights(...)
```

**Problemas**:
1. Requiere nuevo código para cada dominio
2. Lógica duplicada
3. No extensible por usuarios
4. Limita la aplicabilidad del sistema

### ❌ Nombres Específicos de Dominio

```python
# ❌ MAL
class PodcastTranscription:
    episode_title: str
    host_names: List[str]
    guest_name: str
    
# ❌ MAL
PODCAST_PROMPT = "Analiza este episodio de podcast..."
```

### ❌ Supuestos Hardcodeados

```python
# ❌ MAL - Asume estructura específica
def analyze(audio):
    # Asume que siempre hay host y guest
    segments = split_by_speaker(num_speakers=2)
    host_parts = [s for s in segments if s.speaker == "host"]
```

---

## ✅ Patrones Correctos

### ✅ Interfaz Genérica y Configurable

```python
# ✅ BIEN - Genérico, funciona con cualquier audio
class AudioAnalyzer:
    def analyze(
        self, 
        audio: Path,
        config: Optional[AnalysisConfig] = None
    ) -> AudioAnalysis:
        """
        Analiza cualquier tipo de audio.
        
        Args:
            audio: Path al archivo de audio
            config: Configuración opcional. Si no se provee,
                   usa configuración por defecto inteligente.
        
        Returns:
            AudioAnalysis con transcripción, topics, insights
        
        Examples:
            # Funciona con CUALQUIER audio
            analyzer.analyze("podcast.mp3")
            analyzer.analyze("customer_call.wav")
            analyzer.analyze("lecture.m4a")
            analyzer.analyze("focus_group.opus")
        """
        ...
```

### ✅ Nombres Genéricos

```python
# ✅ BIEN - Modelos de datos genéricos
@dataclass
class AudioAnalysis:
    """Resultado genérico aplicable a cualquier audio."""
    transcription: Transcription
    metadata: AudioMetadata
    topics: Optional[TopicModel] = None
    entities: Optional[List[Entity]] = None
    summary: Optional[str] = None
    
    # Extensible para casos específicos
    custom_fields: Dict[str, Any] = field(default_factory=dict)
```

### ✅ Configuración sobre Código

```yaml
# ✅ BIEN - Usuario controla comportamiento
analysis:
  transcription:
    model: "whisper-large-v3"
    language: "auto"  # O específico
    diarization: true
  
  topics:
    method: "hybrid"  # LDA + BERTopic
    num_topics: "auto"  # O fijo: 5, 10, etc.
  
  synthesis:
    model: "gpt-4o-mini"
    extract:
      - themes
      - insights
      - questions
    custom_instructions: |
      Focus on {your_custom_aspect}
```

### ✅ Prompts Dinámicos

```python
# ✅ BIEN - Prompt se construye según contexto
class PromptBuilder:
    def build(self, context: AudioContext, config: Config) -> str:
        """Construye prompt adaptado al audio específico."""
        
        # Base genérica
        prompt = self.base_template
        
        # Adaptar según características del audio
        if context.duration > 3600:
            prompt += self.long_audio_instructions
        
        if context.num_speakers > 1:
            prompt += self.multi_speaker_instructions
        
        # Usuario puede inyectar instrucciones custom
        if config.custom_instructions:
            prompt += f"\n\nAdditional focus:\n{config.custom_instructions}"
        
        return prompt
```

### ✅ Extensibilidad (Plugin System)

```python
# ✅ BIEN - Usuario puede extender sin tocar código core
class CustomExtractor(BaseExtractor):
    """User-defined custom logic."""
    
    def extract(self, transcription: Transcription) -> Dict[str, Any]:
        # Tu lógica específica de dominio
        return custom_results

# Registro dinámico
analyzer.register_extractor("my_custom_logic", CustomExtractor())
```

---

## 📋 Checklist de Implementación

Al implementar cualquier módulo, validar:

### Nombres
- [ ] No usa términos específicos de dominio ("podcast", "interview", "lecture")
- [ ] Usa nombres genéricos y descriptivos
- [ ] Clases y funciones aplicables a cualquier audio

### Configuración
- [ ] Comportamiento controlado por config, no hardcodeado
- [ ] Valores por defecto inteligentes (funciona sin config)
- [ ] Usuario puede personalizar sin tocar código

### Supuestos
- [ ] No asume estructura del audio (número de hablantes, duración, etc.)
- [ ] Detecta características dinámicamente
- [ ] Maneja casos edge gracefully

### Prompts (LLM)
- [ ] Se construyen dinámicamente según contexto
- [ ] No hay templates fijos específicos de dominio
- [ ] Usuario puede inyectar instrucciones custom

### Outputs
- [ ] Estructuras de datos genéricas
- [ ] Campo `custom_fields` para extensibilidad
- [ ] Serializables a JSON/dict

### Extensibilidad
- [ ] Plugin system permite lógica custom
- [ ] No requiere modificar código core
- [ ] Interfaces bien documentadas

### Tests
- [ ] Tests con múltiples tipos de audio
- [ ] Valida que funciona genéricamente
- [ ] Edge cases de diferentes dominios

---

## 🎯 Valor de Portfolio

### Lo Que Esto Demuestra

**Pensamiento Arquitectónico**:
- No solo resuelves un problema específico
- Diseñas sistemas escalables y reutilizables
- Piensas en abstracciones correctas

**Madurez Técnica**:
- Entiendes trade-offs de diseño
- Sabes cuándo generalizar vs. especializar
- Aplicas principios SOLID

**Visión de Producto**:
- Piensas en múltiples usuarios/casos de uso
- Diseño user-centric (usuario controla)
- Escalabilidad a largo plazo

### Diferenciadores

| Proyecto Típico | AudioMind (Agnóstico) |
|-----------------|----------------------|
| "Analiza podcasts" | "Analiza cualquier audio" |
| Hardcodea casos | Usuario configura |
| 1 caso de uso | N casos de uso |
| Difícil extender | Plugin system |
| Código específico | Código genérico + config |

---

## 📚 Documentación de Referencia

### 🔴 **LEER PRIMERO**
- `docs/architecture/DESIGN_PRINCIPLES.md` - 6 principios con ejemplos código

### Implementación
- `app/processors/__init__.py` - Ver docstrings genéricos
- `config/` - Ejemplos de configuración

### Actualizaciones
- `docs/architecture/DOMAIN_AGNOSTIC_UPDATE.md` - Changelog del cambio

---

## 🚀 Próximos Pasos

Al implementar cada módulo:

1. ✅ **Leer** `DESIGN_PRINCIPLES.md`
2. ✅ **Usar** checklist de diseño agnóstico
3. ✅ **Escribir** tests con múltiples tipos de audio
4. ✅ **Validar** que funciona genéricamente
5. ✅ **Documentar** configurabilidad

---

## 💬 Ejemplos de Messaging

### ❌ Incorrecto
- "AudioMind analiza tus podcasts"
- "Plataforma para entrevistas de clientes"
- "Transcribe clases educativas"

### ✅ Correcto
- "AudioMind analiza **cualquier** audio"
- "Plataforma **agnóstica** de inteligencia de audio"
- "Funciona con podcasts, entrevistas, clases, reuniones, y más"
- "**Tú defines** qué extraer y cómo analizarlo"

---

## ⚠️ Recordatorio Final

> **Cada línea de código debe preguntarse**: ¿Funciona esto con CUALQUIER audio, o estoy asumiendo un caso específico?

**Este es el diferenciador clave del proyecto.**

---

**Fecha**: Octubre 23, 2025  
**Estado**: ✅ **PRINCIPIO DEFINIDO - APLICAR EN IMPLEMENTACIÓN**
