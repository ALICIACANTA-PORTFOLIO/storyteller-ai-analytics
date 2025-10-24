# 🐳 Docker Setup - AudioMind

> Configuración de infraestructura con Docker para desarrollo local

## 📦 Servicios Incluidos

El `docker-compose.yml` incluye los siguientes servicios:

### Servicios Core
- **PostgreSQL 16** - Base de datos principal (puerto 5432)
- **Redis 7** - Cache y broker de Celery (puerto 6379)
- **ChromaDB** - Base de datos vectorial para RAG (puerto 8000)

### Servicios Administrativos (Opcional)
- **PgAdmin 4** - Interface web para PostgreSQL (puerto 5050)

---

## 🚀 Quick Start

### 1. Copiar variables de ambiente

```bash
# Copiar template y configurar
cp .env.example .env

# Editar .env con tus valores
# Los servicios Docker usan estas variables
```

### 2. Levantar servicios core

```bash
# Levantar todos los servicios core
docker-compose up -d

# Ver logs
docker-compose logs -f

# Ver estado
docker-compose ps
```

### 3. Verificar servicios

```bash
# PostgreSQL
docker-compose exec postgres psql -U audiomind -d audiomind_db -c "SELECT version();"

# Redis
docker-compose exec redis redis-cli -a audiomind_redis_password ping

# ChromaDB
curl http://localhost:8000/api/v1/heartbeat
```

### 4. (Opcional) Levantar PgAdmin

```bash
# Levantar con profile admin
docker-compose --profile admin up -d pgadmin

# Acceder a: http://localhost:5050
# Email: admin@audiomind.local
# Password: admin_password (cambiar en .env)
```

---

## 🔧 Comandos Útiles

### Gestión de Servicios

```bash
# Iniciar todos
docker-compose up -d

# Detener todos
docker-compose down

# Detener y eliminar volúmenes (⚠️ BORRA DATOS)
docker-compose down -v

# Restart un servicio específico
docker-compose restart postgres

# Ver logs de un servicio
docker-compose logs -f postgres

# Ejecutar comando en contenedor
docker-compose exec postgres bash
```

### PostgreSQL

```bash
# Conectarse a PostgreSQL
docker-compose exec postgres psql -U audiomind -d audiomind_db

# Backup de base de datos
docker-compose exec postgres pg_dump -U audiomind audiomind_db > backup.sql

# Restore de backup
docker-compose exec -T postgres psql -U audiomind -d audiomind_db < backup.sql

# Ver tablas
docker-compose exec postgres psql -U audiomind -d audiomind_db -c "\dt audiomind.*"
```

### Redis

```bash
# Conectarse a Redis
docker-compose exec redis redis-cli -a audiomind_redis_password

# Ver todas las keys
docker-compose exec redis redis-cli -a audiomind_redis_password KEYS "*"

# Limpiar cache
docker-compose exec redis redis-cli -a audiomind_redis_password FLUSHDB
```

### ChromaDB

```bash
# Ver collections
curl http://localhost:8000/api/v1/collections

# Health check
curl http://localhost:8000/api/v1/heartbeat
```

---

## 📊 Persistencia de Datos

Los datos persisten en volúmenes Docker:

```bash
# Ver volúmenes
docker volume ls | grep storyteller-ai-analytics

# Inspeccionar volumen
docker volume inspect storyteller-ai-analytics_postgres_data

# Eliminar volúmenes (⚠️ BORRA DATOS)
docker volume rm storyteller-ai-analytics_postgres_data
```

**Ubicación de datos en Windows:**
```
%USERPROFILE%\AppData\Local\Docker\wsl\data\
```

---

## 🔒 Seguridad

### Cambiar Passwords por Defecto

**⚠️ IMPORTANTE**: Los passwords por defecto son solo para desarrollo.

Edita `.env` y cambia:

```bash
# PostgreSQL
DB_PASSWORD=tu-password-seguro-aqui

# Redis
REDIS_PASSWORD=tu-password-seguro-aqui

# ChromaDB
CHROMA_TOKEN=tu-token-seguro-aqui

# PgAdmin
PGADMIN_PASSWORD=tu-password-seguro-aqui
```

### Generar Passwords Seguros

```bash
# Windows PowerShell
[System.Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes((New-Guid)))

# O usa un password manager
```

---

## 🐛 Troubleshooting

### Puerto ya en uso

```bash
# Ver qué proceso usa el puerto
netstat -ano | findstr :5432

# Cambiar puerto en .env
DB_PORT=5433
```

### Contenedor no inicia

```bash
# Ver logs detallados
docker-compose logs postgres

# Recrear contenedor
docker-compose up -d --force-recreate postgres
```

### Problemas de conexión desde Python

```python
# Verificar connection string
from app.config import settings
print(settings.database.connection_string)

# Test de conexión
from sqlalchemy import create_engine, text

engine = create_engine(settings.database.connection_string)
with engine.connect() as conn:
    result = conn.execute(text("SELECT 1"))
    print("✅ Conexión exitosa")
```

### Reiniciar todo desde cero

```bash
# ⚠️ ESTO BORRA TODOS LOS DATOS

# 1. Bajar servicios y eliminar volúmenes
docker-compose down -v

# 2. Eliminar imágenes (opcional)
docker-compose down --rmi all

# 3. Limpiar Docker
docker system prune -a

# 4. Volver a levantar
docker-compose up -d
```

---

## 🏗️ Arquitectura

```
┌─────────────────────────────────────────────────────────┐
│                   AudioMind Application                 │
│                  (Python/FastAPI/Streamlit)             │
└─────────────────────────────────────────────────────────┘
           │                 │                 │
           │                 │                 │
           v                 v                 v
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ PostgreSQL   │   │    Redis     │   │  ChromaDB    │
│              │   │              │   │              │
│ Port: 5432   │   │ Port: 6379   │   │ Port: 8000   │
│              │   │              │   │              │
│ - Metadata   │   │ - Cache      │   │ - Embeddings │
│ - Transcripts│   │ - Celery     │   │ - RAG Search │
│ - Topics     │   │              │   │              │
└──────────────┘   └──────────────┘   └──────────────┘
       │
       │ (admin)
       v
┌──────────────┐
│   PgAdmin    │
│ Port: 5050   │
│ (optional)   │
└──────────────┘
```

---

## 📝 Notas

### PostgreSQL

- **Versión**: 16 Alpine (lightweight)
- **Schema**: `audiomind` (creado automáticamente)
- **Extensiones**: `uuid-ossp`, `pg_trgm`
- **Connection Pool**: Configurado en `app/config.py`

### Redis

- **Versión**: 7 Alpine
- **Uso**: Cache de aplicación + Celery broker
- **Persistencia**: Activada (RDB + AOF)

### ChromaDB

- **Versión**: Latest
- **Uso**: Búsqueda semántica (RAG)
- **Persistencia**: Volumen montado en `/chroma/chroma`

---

## 🔄 Actualizar Servicios

```bash
# Pull latest images
docker-compose pull

# Recrear contenedores con nuevas imágenes
docker-compose up -d --force-recreate

# Ver versiones actuales
docker-compose images
```

---

## 📚 Referencias

- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [PostgreSQL Docker Hub](https://hub.docker.com/_/postgres)
- [Redis Docker Hub](https://hub.docker.com/_/redis)
- [ChromaDB Documentation](https://docs.trychroma.com/)

---

**Creado**: Octubre 24, 2025  
**Mantenido por**: Julio César García Escoto  
**Estado**: ✅ Listo para desarrollo
