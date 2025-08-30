from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pydantic import BaseModel, Field


@dataclass
class ExtractedContent:
    """Clase para almacenar contenido extraído del PDF."""
    text_chunks: List[str]
    images: List[Dict[str, Any]]  # {image: PIL.Image, page: int, bbox: tuple}
    metadata: List[Dict[str, Any]]  # Metadatos para cada chunk


class ProcessDocumentRequest(BaseModel):
    """Modelo para la solicitud de procesamiento de documento"""
    file_path: str = Field(..., description="Ruta del archivo PDF a procesar")


class ProcessDocumentResponse(BaseModel):
    """Modelo para la respuesta de procesamiento de documento"""
    success: bool = Field(..., description="Indica si el procesamiento fue exitoso")
    message: str = Field(..., description="Mensaje descriptivo del resultado")
    total_chunks: Optional[int] = Field(None, description="Número total de chunks procesados")
    total_images: Optional[int] = Field(None, description="Número total de imágenes extraídas")
