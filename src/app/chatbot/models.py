from pydantic import BaseModel, Field
from typing import List, Optional


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
