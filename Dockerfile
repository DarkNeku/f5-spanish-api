FROM python:3.10-slim

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    wget \
    build-essential \
    libsndfile1 \
    espeak-ng \
    espeak-ng-data \
    libespeak-ng-dev \
    && rm -rf /var/lib/apt/lists/*

# Establecer directorio de trabajo
WORKDIR /app

# Configurar variables de entorno
ENV PYTHONPATH=/app
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/transformers
ENV TORCH_HOME=/app/.cache/torch

# Crear directorios de cache
RUN mkdir -p /app/.cache/huggingface /app/.cache/transformers /app/.cache/torch

# Copiar archivos de requirements primero (para cache de Docker)
COPY requirements.txt .

# Instalar dependencias Python
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Instalar F5-TTS desde GitHub
RUN pip install --no-cache-dir git+https://github.com/SWivid/F5-TTS.git

# Copiar código de la aplicación
COPY main.py .

# Exponer puerto
EXPOSE 8000

# Crear usuario no-root para seguridad
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Comando para ejecutar la aplicación
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]