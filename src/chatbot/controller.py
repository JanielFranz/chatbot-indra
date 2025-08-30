from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import logging

from .service import ChatbotService, get_chatbot_service

# Router para los endpoints del chatbot
router = APIRouter(
    prefix="/chatbot",
    tags=["Chatbot"],
    responses={404: {"description": "Not found"}},
)

# Modelos Pydantic para request/response
class QuestionRequest(BaseModel):
    """Modelo para la pregunta del usuario"""
    question: str = Field(..., min_length=1, max_length=500, description="Pregunta del usuario")

    class Config:
        schema_extra = {
            "example": {
                "question": "¿Cuáles son los beneficios del machine learning?"
            }
        }

class SourceInfo(BaseModel):
    """Información de una fuente encontrada"""
    text: str = Field(..., description="Fragmento de texto relevante")
    page_number: Optional[int] = Field(None, description="Número de página del documento")
    similarity: float = Field(..., description="Puntuación de similitud")
    chunk_id: str = Field(..., description="ID del chunk de texto")

class ChatbotResponse(BaseModel):
    """Respuesta del chatbot"""
    success: bool = Field(..., description="Indica si la operación fue exitosa")
    answer: str = Field(..., description="Respuesta generada para la pregunta")
    sources: List[SourceInfo] = Field(default=[], description="Fuentes de información utilizadas")
    images: List[str] = Field(default=[], description="Rutas de imágenes relacionadas")
    total_results: Optional[int] = Field(None, description="Número total de resultados encontrados")
    question: str = Field(..., description="Pregunta original del usuario")

class ErrorResponse(BaseModel):
    """Respuesta de error"""
    success: bool = Field(False, description="Indica que hubo un error")
    error: str = Field(..., description="Descripción del error")
    answer: str = Field(..., description="Mensaje de error para el usuario")
    sources: List[SourceInfo] = Field(default=[], description="Lista vacía en caso de error")
    images: List[str] = Field(default=[], description="Lista vacía en caso de error")

@router.post(
    "/ask",
    response_model=ChatbotResponse,
    summary="Hacer una pregunta al chatbot",
    description="Envía una pregunta al chatbot y recibe una respuesta basada en el documento procesado",
    responses={
        200: {
            "description": "Respuesta exitosa del chatbot",
            "model": ChatbotResponse
        },
        400: {
            "description": "Error en la solicitud",
            "model": ErrorResponse
        },
        500: {
            "description": "Error interno del servidor",
            "model": ErrorResponse
        }
    }
)
async def ask_question(
    request: QuestionRequest,
    chatbot_service: ChatbotService = Depends(get_chatbot_service)
) -> Dict[str, Any]:
    """
    Procesa una pregunta del usuario y devuelve una respuesta basada en el documento.

    - **question**: La pregunta que quieres hacer al chatbot

    El chatbot buscará en el vector database para encontrar información relevante
    y generará una respuesta usando un modelo de lenguaje.
    """
    try:
        if not request.question.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="La pregunta no puede estar vacía"
            )

        # Llamar al servicio para obtener la respuesta
        result = chatbot_service.answer_user_question(request.question.strip())

        if not result.get("success", False):
            # Si el servicio indica que no fue exitoso pero no es un error crítico
            return result

        return result

    except Exception as e:
        logging.error(f"Error en el endpoint ask_question: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "success": False,
                "error": f"Error interno del servidor: {str(e)}",
                "answer": "Lo siento, ocurrió un error inesperado. Por favor, inténtalo de nuevo.",
                "sources": [],
                "images": []
            }
        )

@router.get(
    "/health",
    summary="Verificar estado del chatbot",
    description="Endpoint para verificar que el servicio del chatbot está funcionando correctamente"
)
async def health_check(
    chatbot_service: ChatbotService = Depends(get_chatbot_service)
) -> Dict[str, str]:
    """
    Verifica que el servicio del chatbot esté funcionando.

    Útil para monitoreo y diagnóstico del sistema.
    """
    try:
        # Verificar que los componentes principales estén inicializados
        if (chatbot_service.embeddings_generator is None or
            chatbot_service.vector_store is None or
            chatbot_service.logger is None):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Servicio no disponible: componentes no inicializados"
            )

        return {
            "status": "healthy",
            "message": "Chatbot service is running",
            "service": "multimodal-rag-chatbot"
        }

    except Exception as e:
        logging.error(f"Error en health check: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Servicio no disponible: {str(e)}"
        )
