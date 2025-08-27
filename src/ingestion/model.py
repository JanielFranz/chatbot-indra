from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

@dataclass
class ExtractedContent:
    """Clase para almacenar contenido extraído del PDF."""
    text_chunks: List[str]
    images: List[Dict[str, Any]]  # {image: PIL.Image, page: int, bbox: tuple}
    metadata: List[Dict[str, Any]]  # Metadatos para cada chunk
