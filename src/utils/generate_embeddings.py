from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from sentence_transformers import SentenceTransformer

class EmbeddingsGenerator:

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.embedding_model_name = embedding_model
        self.embedding_model = SentenceTransformer(embedding_model)


    def generate_embeddings(self, text_chunks: List[str], ) -> np.ndarray:
        """
        Genera embeddings para los chunks de texto.

        Args:
            text_chunks (List[str]): Lista de chunks de texto

        Returns:
            np.ndarray: Array de embeddings (n_chunks, embedding_dim)
        """
        self.logger.info(f"Generando embeddings para {len(text_chunks)} chunks...")

        # Filtrar chunks vacíos
        valid_chunks = [chunk for chunk in text_chunks if chunk.strip()]

        if not valid_chunks:
            return np.array([])

        # Generar embeddings
        embeddings = self.embedding_model.encode(
            valid_chunks,
            convert_to_numpy=True,
            show_progress_bar=True
        )

        self.logger.info(f"Embeddings generados: {embeddings.shape}")
        return embeddings