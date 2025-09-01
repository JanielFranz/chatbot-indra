# Usar Python 3.11 slim como base
FROM python:3.11-slim

# Establecer variables de entorno
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Instalar dependencias del sistema necesarias para OCR y PDF
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-spa \
    poppler-utils \
    libpq-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Crear directorio de trabajo
WORKDIR /app

# Copiar requirements y instalar dependencias Python
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.2.2+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install --no-cache-dir -r requirements.txt -v


# Copiar el código de la aplicación
COPY . .

# Crear directorio para archivos temporales
RUN mkdir -p /app/temp

# Exponer puertos (8000 para FastAPI, 7860 para Gradio)
EXPOSE 8000 7860

# Script de inicio para ejecutar ambas aplicaciones
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Comando por defecto
CMD ["/app/start.sh"]