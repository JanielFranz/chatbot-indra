from typing import List, Dict, Any
import numpy as np
import logging
from sentence_transformers import SentenceTransformer


class EmbeddingsGenerator:
    """
    Generador de embeddings usando SentenceTransformers.
    Maneja la creación de embeddings para texto usando modelos pre-entrenados.
    """

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.embedding_model_name = embedding_model
        self.embedding_model = SentenceTransformer(embedding_model)

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Genera embeddings para una lista de textos."""
        if not texts:
            return np.array([])

        embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
        return embeddings

    def generate_embedding(self, text: str) -> np.ndarray:
        """Genera embedding para un solo texto."""
        return self.embedding_model.encode([text], convert_to_numpy=True)[0]

    def get_embedding_dimension(self) -> int:
        """Retorna la dimensión de los embeddings."""
        return self.embedding_model.get_sentence_embedding_dimension()

    def get_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas del generador de embeddings."""
        return {
            "model_name": self.embedding_model_name,
            "embedding_dimension": self.get_embedding_dimension(),
            "device": str(self.embedding_model.device)
        }
