from src.database.core_faiss import FAISSVectorStore
from .models import ExtractedContent
from src.utils.preprocessing.pdf_processor import PDFPreprocessor
from src.utils.embeddings.generator import EmbeddingsGenerator
import numpy as np
import logging
from typing import Dict, Any
from fastapi import Depends
from src.container import (
    pdf_processor_dependency,
    embeddings_generator_dependency,
    vector_store_dependency,
    logger_dependency
)


class IngestionService:
    """
    Servicio de ingesta con inyección de dependencias usando FastAPI.
    Similar a un @Service en Spring Boot pero usando FastAPI Depends().

    Este servicio maneja todo el proceso de extracción, embedding y almacenamiento
    de documentos PDF en la base de datos vectorial FAISS.
    """

    def __init__(
        self,
        pdf_processor: PDFPreprocessor,
        embeddings_generator: EmbeddingsGenerator,
        vector_store: FAISSVectorStore,
        logger: logging.Logger
    ):
        """Constructor con inyección de dependencias."""
        self.pdf_processor = pdf_processor
        self.embeddings_generator = embeddings_generator
        self.vector_store = vector_store
        self.logger = logger

    def transform_pdf_to_embeddings(self) -> Dict[str, Any]:
        """
        Procesa un PDF y genera embeddings almacenándolos en el vector store.

        Returns:
            Dict[str, Any]: Resultado del procesamiento con estadísticas
        """
        import os
        # Obtener la ruta absoluta del archivo PDF
        current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        file_path = os.path.join(current_dir, "data", "rag-challenge.pdf")

        # Verificar que el archivo existe
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No se encontró el archivo PDF en: {file_path}")

        try:
            self.logger.info(f"Iniciando procesamiento de PDF: {file_path}")

            # 1. Extraer contenido del PDF
            content: ExtractedContent = self.pdf_processor.extract_content_from_pdf(file_path)

            if not content.text_chunks:
                raise ValueError("No se pudo extraer texto del PDF")

            # 2. Generar embeddings para los chunks de texto
            self.logger.info(f"Generando embeddings para {len(content.text_chunks)} chunks")
            embeddings: np.ndarray = self.embeddings_generator.generate_embeddings(content.text_chunks)

            # 3. Preparar datos estructurados para FAISS
            processed_data = {
                "embeddings": embeddings,
                "text_chunks": content.text_chunks,
                "metadata": content.metadata,
                "images": content.images,
                "total_chunks": len(content.text_chunks),
                "total_images": len(content.images),
                "embedding_dimension": embeddings.shape[1] if len(embeddings) > 0 else 0,
                "source_file": file_path
            }

            self.logger.info(
                f"Procesamiento completado: {processed_data['total_chunks']} chunks, "
                f"{processed_data['total_images']} imágenes, dimensión: {processed_data['embedding_dimension']}"
            )

            # 4. Almacenar en el vector store usando processed_data
            assigned_ids = self.vector_store.add_embeddings(processed_data=processed_data)
            self.logger.info(f"Datos almacenados en vector store con IDs: {len(assigned_ids)} asignados")

            # 5. Retornar resultado con estadísticas
            result = {
                "success": True,
                "message": f"Procesamiento completado: {processed_data['total_chunks']} chunks, {processed_data['total_images']} imágenes",
                "file_path": file_path,
                "total_chunks": processed_data['total_chunks'],
                "total_images": processed_data['total_images'],
                "embedding_dimension": processed_data['embedding_dimension'],
                "assigned_ids": assigned_ids,
                "vector_store_stats": self.vector_store.get_stats()
            }

            return result

        except Exception as e:
            error_msg = f"Error durante el procesamiento de {file_path}: {str(e)}"
            self.logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "file_path": file_path
            }

    def get_processing_summary(self) -> Dict[str, Any]:
        """
        Retorna un resumen del estado actual del vector store.

        Returns:
            Dict con estadísticas del vector store
        """
        return self.vector_store.get_stats()


# Factory function para el servicio usando FastAPI Depends
def get_ingestion_service(
    pdf_processor: PDFPreprocessor = Depends(pdf_processor_dependency),
    embeddings_generator: EmbeddingsGenerator = Depends(embeddings_generator_dependency),
    vector_store: FAISSVectorStore = Depends(vector_store_dependency),
    logger: logging.Logger = Depends(logger_dependency)
) -> IngestionService:
    """
    Factory function para crear IngestionService con todas sus dependencias.
    Esto es equivalente a @Autowired en Spring Boot pero usando FastAPI Depends().
    """
    return IngestionService(
        pdf_processor=pdf_processor,
        embeddings_generator=embeddings_generator,
        vector_store=vector_store,
        logger=logger
    )


# Factory function para uso fuera de FastAPI (scripts, testing, etc.)
def create_ingestion_service() -> IngestionService:
    """
    Factory function para crear IngestionService fuera del contexto de FastAPI.
    Usa las funciones de factory directamente sin Depends().
    """
    from src.container import (
        create_pdf_processor,
        create_embeddings_generator,
        create_vector_store,
        get_logger
    )

    return IngestionService(
        pdf_processor=create_pdf_processor(),
        embeddings_generator=create_embeddings_generator(),
        vector_store=create_vector_store(),
        logger=get_logger()
    )
