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
from src.utils.preprocessing.pdf_processor import PDFPreprocessor
from src.utils.embeddings.generator import EmbeddingsGenerator
from src.logging import configure_logging, LogLevels

# Importaciones del módulo LLM
from src.llm.chain_manager import LLMChainManager
from src.llm.providers.groq_provider import GroqProvider
from src.llm.prompts.prompt_manager import PromptManager


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


# Factory functions para LLM
@lru_cache()
def get_groq_provider() -> GroqProvider:
    """Crea un GroqProvider singleton."""
    return GroqProvider(logger=get_logger())


@lru_cache()
def get_prompt_manager() -> PromptManager:
    """Crea un PromptManager singleton."""
    return PromptManager(logger=get_logger())


@lru_cache()
def get_llm_chain_manager() -> LLMChainManager:
    """Crea un LLMChainManager singleton."""
    return LLMChainManager(
        provider=get_groq_provider(),
        prompt_manager=get_prompt_manager(),
        logger=get_logger()
    )


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


# Funciones de dependencia para LLM
def groq_provider_dependency() -> GroqProvider:
    """Dependency provider para GroqProvider."""
    return get_groq_provider()


def prompt_manager_dependency() -> PromptManager:
    """Dependency provider para PromptManager."""
    return get_prompt_manager()


def llm_chain_manager_dependency() -> LLMChainManager:
    """Dependency provider para LLMChainManager."""
    return get_llm_chain_manager()


# Factory functions para uso fuera de FastAPI (scripts, testing, etc.)
def create_pdf_processor() -> PDFPreprocessor:
    """Factory function para crear PDFPreprocessor fuera del contexto de FastAPI."""
    return PDFPreprocessor()


def create_embeddings_generator() -> EmbeddingsGenerator:
    """Factory function para crear EmbeddingsGenerator fuera del contexto de FastAPI."""
    return EmbeddingsGenerator()


def create_vector_store() -> FAISSVectorStore:
    """Factory function para crear FAISSVectorStore fuera del contexto de FastAPI."""
    return FAISSVectorStore()


# Factory functions para LLM (uso fuera de FastAPI)
def create_groq_provider() -> GroqProvider:
    """Factory function para crear GroqProvider fuera del contexto de FastAPI."""
    return GroqProvider()


def create_prompt_manager() -> PromptManager:
    """Factory function para crear PromptManager fuera del contexto de FastAPI."""
    return PromptManager()


def create_llm_chain_manager() -> LLMChainManager:
    """Factory function para crear LLMChainManager fuera del contexto de FastAPI."""
    return LLMChainManager(
        provider=create_groq_provider(),
        prompt_manager=create_prompt_manager()
    )
