# 📚 Referencias Bibliográficas y Fuentes Técnicas

Este documento lista las fuentes académicas, libros técnicos y papers que fundamentan las técnicas implementadas en este proyecto.

## 🧠 Natural Language Processing (NLP)

### Libros de Referencia

1. **Natural Language Processing with Python**
   - Autores: Steven Bird, Ewan Klein, Edward Loper
   - Editorial: O'Reilly Media
   - Uso: Fundamentos de NLP, tokenización, y procesamiento de corpus

2. **Practical Natural Language Processing**
   - Autores: Sowmya Vajjala, Bodhisattwa Majumder, Anuj Gupta, Harshit Surana
   - Editorial: O'Reilly Media
   - Uso: Implementaciones prácticas de pipelines NLP

3. **Mastering Regular Expressions**
   - Autor: Jeffrey E.F. Friedl
   - Editorial: O'Reilly Media
   - Uso: Técnicas avanzadas de regex para parsing de texto

## 🤖 Transformers y Modelos de Lenguaje

### Papers Académicos

4. **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**
   - Autores: Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova
   - Conferencia: NAACL 2019
   - URL: https://arxiv.org/abs/1810.04805
   - Uso: Fundamentos de modelos BERT para embeddings y topic modeling

5. **Large Language Models Meet NLP: A Survey**
   - Autores: Varios
   - Año: 2023
   - Uso: Estado del arte en LLMs y sus aplicaciones en NLP

6. **SuperGLUE: A Stickier Benchmark for General-Purpose Language Understanding Systems**
   - Autores: Alex Wang et al.
   - Conferencia: NeurIPS 2019
   - URL: https://arxiv.org/abs/1905.00537
   - Uso: Benchmarks para evaluación de modelos de lenguaje

## 📊 Topic Modeling

### Papers y Documentación

7. **Evolution of Topic Modeling**
   - Tipo: Review Paper
   - Uso: Historia y evolución de técnicas de topic modeling (LDA → BERTopic)

8. **Latent Dirichlet Allocation (LDA)**
   - Autores: David M. Blei, Andrew Y. Ng, Michael I. Jordan
   - Journal: JMLR 2003
   - URL: https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf
   - Uso: Algoritmo base para topic modeling probabilístico

9. **BERTopic**
   - Autor: Maarten Grootendorst
   - Documentación: https://maartengr.github.io/BERTopic/
   - GitHub: https://github.com/MaartenGr/BERTopic
   - Uso: Topic modeling con embeddings contextuales

## 🎙️ Speech Recognition

### Modelos y Papers

10. **Whisper: Robust Speech Recognition via Large-Scale Weak Supervision**
    - Autores: Alec Radford, Jong Wook Kim, et al. (OpenAI)
    - Año: 2022
    - URL: https://arxiv.org/abs/2212.04356
    - GitHub: https://github.com/openai/whisper
    - Uso: Transcripción automática de audio

## 📖 Material de Curso

### Materiales Didácticos

11. **MNA - Maestría en Ciencia de Datos - NLP**
    - Institución: [Nombre de Universidad]
    - Curso: Natural Language Processing
    - Semanas: 1, 2, 3, 4, 6, 8
    - Temas: Historia de NLP, Corpus, Tokenización, DTM/TF-IDF, LSI, Transformers
    - Uso: Base teórica y fundamentos académicos

## 🔧 Recursos Técnicos

12. **Python Regular Expressions - Cheat Sheet**
    - Tipo: Referencia rápida
    - Uso: Guía de consulta para expresiones regulares

13. **ArXiv Paper 101306.101310**
    - Uso: [Especificar tema relacionado con el proyecto]

## 📝 Cómo usar estas referencias

### En Código
```python
# Ejemplo de cita en código
def extract_topics_lda(self, corpus, num_topics=10):
    """
    Extract topics using Latent Dirichlet Allocation.
    
    Based on: Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003).
    Latent dirichlet allocation. JMLR, 3, 993-1022.
    """
```

### En Documentación
Al documentar técnicas implementadas, siempre incluir:
- **Qué**: Descripción de la técnica
- **Por qué**: Justificación de su uso
- **Fuente**: Referencia académica o paper

## ⚖️ Nota sobre Derechos de Autor

- **NO se incluyen** los PDFs completos en el repositorio
- Se respetan los derechos de autor de todos los autores
- Se proporcionan **citas apropiadas** y enlaces a fuentes originales
- Para obtener los textos completos, consultar:
  - Bibliotecas universitarias
  - Plataformas académicas (ArXiv, Google Scholar)
  - Editoriales oficiales (O'Reilly, IEEE, ACM)

## 🔗 Enlaces Útiles

- **ArXiv**: https://arxiv.org/ (Papers de acceso abierto)
- **Google Scholar**: https://scholar.google.com/ (Búsqueda académica)
- **Papers with Code**: https://paperswithcode.com/ (Papers + implementaciones)
- **Hugging Face**: https://huggingface.co/ (Modelos pre-entrenados)

---

**Última actualización**: Octubre 2025

**Mantenimiento**: Actualizar este archivo al agregar nuevas técnicas o fuentes
