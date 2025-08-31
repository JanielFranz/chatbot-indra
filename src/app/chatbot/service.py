from src.database.core_faiss import FAISSVectorStore
from src.utils.embeddings.generator import EmbeddingsGenerator
from src.llm.chain_manager import LLMChainManager
import logging
from typing import Dict, Any
from fastapi import Depends
import os
from dotenv import load_dotenv

# Cargar variables de entorno desde .env
load_dotenv()

from src.container import (
    vector_store_dependency,
    embeddings_generator_dependency,
    logger_dependency,
    llm_chain_manager_dependency
)


class ChatbotService:
    """
    Servicio de chatbot que maneja las consultas de usuarios
    y genera respuestas usando RAG (Retrieval Augmented Generation).
    """

    def __init__(self,
                 embeddings_generator: EmbeddingsGenerator,
                 vector_store: FAISSVectorStore,
                 llm_chain_manager: LLMChainManager,
                 logger: logging.Logger):
        self.embeddings_generator = embeddings_generator
        self.vector_store = vector_store
        self.llm_chain_manager = llm_chain_manager
        self.logger = logger

    def answer_user_question(self, question: str) -> Dict[str, Any]:
        """
        Responde una pregunta del usuario buscando en el vector store.

        Args:
            question (str): Pregunta del usuario

        Returns:
            Dict con la respuesta, metadatos e imágenes relacionadas
        """
        try:
            self.logger.debug(f"Procesando pregunta: {question}")

            # Generar embedding de la pregunta
            question_as_embedding = self.embeddings_generator.generate_embedding(question)

            # Buscar en el vector store con metadatos
            distances, results = self.vector_store.search(
                question_as_embedding,
                k=3,  # Obtener top 3 resultados
                return_metadata=True
            )

            if not results:
                return {
                    "success": False,
                    "message": "No se encontraron resultados relevantes",
                    "answer": "Lo siento, no pude encontrar información relevante para responder tu pregunta.",
                    "sources": [],
                    "images": []
                }

            # Procesar resultados
            best_match = results[0]
            context_text = best_match.get('text', '')

            # Recopilar información de fuentes
            sources = []
            related_images = []

            for result in results:
                source_info = {
                    "text": result.get('text', '')[:200] + "...",
                    "page_number": result.get('page_number', 'N/A'),
                    "similarity": result.get('similarity', 0.0),
                    "chunk_id": result.get('id', 'N/A')
                }
                sources.append(source_info)

                # Agregar imágenes relacionadas si existen
                if 'associated_images' in result and result['associated_images'] > 0:
                    image_info = result.get('image_info', [])
                    for img_info in image_info:
                        # Extraer solo la ruta de la imagen como string
                        if isinstance(img_info, dict) and 'image_path' in img_info:
                            related_images.append(img_info['image_path'])
                        elif isinstance(img_info, str):
                            # Si ya es un string, usarlo directamente
                            related_images.append(img_info)

            # Remover duplicados manteniendo el orden
            related_images = list(dict.fromkeys(related_images))

            # Generar respuesta usando el LLM Chain Manager
            llm_response = self.llm_chain_manager.generate_rag_response(
                context=context_text,
                question=question,
                config={'max_context_length': 2000}
            )

            if llm_response.get('success', False):
                answer = llm_response['answer']
            else:
                # Usar respuesta de fallback si el LLM falla
                answer = llm_response.get('answer', context_text[:500] + "...")
                self.logger.warning(f"LLM falló, usando fallback: {llm_response.get('error', 'Unknown error')}")

            return {
                "success": True,
                "answer": answer,
                "sources": sources,
                "images": related_images,
                "total_results": len(results),
                "question": question,
                "llm_metadata": {
                    "provider": llm_response.get('provider', 'unknown'),
                    "prompt_stats": llm_response.get('prompt_stats', {}),
                    "fallback_used": llm_response.get('fallback', False)
                }
            }

        except Exception as e:
            self.logger.error(f"Error answering question: {e}")
            return {
                "success": False,
                "error": str(e),
                "answer": "Lo siento, ocurrió un error al procesar tu pregunta.",
                "sources": [],
                "images": []
            }


# Factory function para el servicio usando FastAPI Depends
def get_chatbot_service(
    embeddings_generator: EmbeddingsGenerator = Depends(embeddings_generator_dependency),
    vector_store: FAISSVectorStore = Depends(vector_store_dependency),
    llm_chain_manager: LLMChainManager = Depends(llm_chain_manager_dependency),
    logger: logging.Logger = Depends(logger_dependency)
) -> ChatbotService:
    """
    Factory function para crear ChatbotService con todas sus dependencias.
    Esto es equivalente a @Autowired en Spring Boot pero usando FastAPI Depends().
    """
    return ChatbotService(
        embeddings_generator=embeddings_generator,
        vector_store=vector_store,
        llm_chain_manager=llm_chain_manager,
        logger=logger
    )


# Factory function para uso fuera de FastAPI (scripts, testing, etc.)
def create_chatbot_service() -> ChatbotService:
    """
    Factory function para crear ChatbotService fuera del contexto de FastAPI.
    """
    from src.container import (
        create_embeddings_generator,
        create_vector_store,
        create_llm_chain_manager,
        get_logger
    )

    return ChatbotService(
        embeddings_generator=create_embeddings_generator(),
        vector_store=create_vector_store(),
        llm_chain_manager=create_llm_chain_manager(),
        logger=get_logger()
    )
