#!/bin/bash

# Iniciar FastAPI en segundo plano
echo "Iniciando backend FastAPI..."
uvicorn src.main:app --host 0.0.0.0 --port 8000 &

# Esperar un momento para que FastAPI se inicie
sleep 10

# Iniciar Gradio
echo "Iniciando aplicación Gradio..."
python src/ui/gradio/app.py
