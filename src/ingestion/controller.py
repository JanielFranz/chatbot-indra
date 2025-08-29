from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import Optional
import os

from src.ingestion.service import IngestionService, get_ingestion_service


# Router para endpoints de ingesta
router = APIRouter(prefix="/ingestion", tags=["ingestion"])


# Modelos de request/response
class ProcessDocumentRequest(BaseModel):
    file_path: str


class ProcessDocumentResponse(BaseModel):
    success: bool
    message: str
    total_chunks: Optional[int] = None
    total_images: Optional[int] = None


@router.post("", response_model=ProcessDocumentResponse)
async def process_document(
    #request: ProcessDocumentRequest,
    # Aquí es donde ocurre la magia de DI - FastAPI inyecta automáticamente el servicio
    service: IngestionService = Depends(get_ingestion_service)
):
    """
    Endpoint para procesar un documento PDF y generar embeddings.

    La inyección de dependencias funciona exactamente como @Autowired en Spring Boot:
    FastAPI automáticamente crea e inyecta todas las dependencias del IngestionService.
    """
    try:
        #if not os.path.exists(request.file_path):
            #raise HTTPException(status_code=404, detail=f"Archivo no encontrado: {request.file_path}")

        #result_message = service.transform_pdf_to_embeddings(request.file_path)
        result_message = service.transform_pdf_to_embeddings()
        return ProcessDocumentResponse(
            success=True,
            message=result_message
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando documento: {str(e)}")


@router.get("/health")
async def health_check():
    """Endpoint simple para verificar que el servicio está funcionando."""
    return {"status": "healthy", "service": "ingestion"}

# Ejemplo de cómo usar el servicio directamente (para scripts o testing)
async def process_rag_challenge_pdf():
    """
    Función de ejemplo que procesa el PDF del rag-challenge.
    Muestra cómo usar el servicio fuera de un endpoint.
    """
    # Crear el servicio con todas sus dependencias (sin FastAPI Depends)
    from src.ingestion.service import create_ingestion_service
    service = create_ingestion_service()

    # Procesar el PDF
    file_path = "src/data/rag-challenge.pdf"
    result = service.transform_pdf_to_embeddings(file_path)

    return result
