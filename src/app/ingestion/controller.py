from fastapi import APIRouter, Depends, HTTPException
from typing import Optional

from .service import IngestionService, get_ingestion_service
from .models import ProcessDocumentResponse


# Router para endpoints de ingesta
router = APIRouter(prefix="/ingestion", tags=["ingestion"])


@router.post("", response_model=ProcessDocumentResponse)
async def process_document(
    service: IngestionService = Depends(get_ingestion_service)
):
    """
    Endpoint para procesar un documento PDF y generar embeddings.

    La inyección de dependencias funciona exactamente como @Autowired en Spring Boot:
    FastAPI automáticamente crea e inyecta todas las dependencias del IngestionService.
    """
    try:
        result_message = service.transform_pdf_to_embeddings()
        return ProcessDocumentResponse(
            success=True,
            message=result_message["message"]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando documento: {str(e)}")


@router.get("/health")
async def health_check():
    """Endpoint de salud para el servicio de ingesta"""
    return {"status": "healthy", "service": "ingestion"}
