"""
Inyección de dependencias usando FastAPI Depends.
Mucho más simple que crear un contenedor personalizado.
"""
import logging
from functools import lru_cache
from fastapi import Depends
from dotenv import load_dotenv

# Cargar variables de entorno desde .env al inicio
load_dotenv()

from src.database.core_faiss import FAISSVectorStore
from src.utils.preprocess import PDFPreprocessor
from src.utils.generate_embeddings import EmbeddingsGenerator
from src.logging import configure_logging, LogLevels


# Configurar logging al inicio
configure_logging(LogLevels.info)


# Factory functions para crear dependencias (similar a @Bean en Spring Boot)
@lru_cache()
def get_logger() -> logging.Logger:
    """Crea un logger singleton."""
    return logging.getLogger('multimodal_rag')


@lru_cache()
def get_pdf_processor() -> PDFPreprocessor:
    """Crea un PDFPreprocessor singleton."""
    return PDFPreprocessor()


@lru_cache()
def get_embeddings_generator() -> EmbeddingsGenerator:
    """Crea un EmbeddingsGenerator singleton."""
    return EmbeddingsGenerator()


@lru_cache()
def get_vector_store() -> FAISSVectorStore:
    """Crea un FAISSVectorStore singleton."""
    return FAISSVectorStore()


# Funciones de dependencia para usar con FastAPI Depends()
def pdf_processor_dependency() -> PDFPreprocessor:
    """Dependency provider para PDFPreprocessor."""
    return get_pdf_processor()


def embeddings_generator_dependency() -> EmbeddingsGenerator:
    """Dependency provider para EmbeddingsGenerator."""
    return get_embeddings_generator()


def vector_store_dependency() -> FAISSVectorStore:
    """Dependency provider para FAISSVectorStore."""
    return get_vector_store()


def logger_dependency() -> logging.Logger:
    """Dependency provider para Logger."""
    return get_logger()
