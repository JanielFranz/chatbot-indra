from fastapi import FastAPI
from src.ingestion.controller import router as ingestion_router

app = FastAPI(
    title="RAG Challenge API",
    description="API para procesamiento de documentos PDF y generación de embeddings",
    version="1.0.0"
)

# Incluir el router de ingestion
app.include_router(ingestion_router)

@app.get("/")
async def root():
    """Endpoint raíz que proporciona información básica de la API."""
    return {
        "message": "RAG Challenge API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/ingestion/health"
    }
