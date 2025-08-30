from src.database.core_faiss import FAISSVectorStore
from src.utils.embeddings.generator import EmbeddingsGenerator
import logging
from typing import Dict, Any
from fastapi import Depends
import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Cargar variables de entorno desde .env
load_dotenv()

from src.container import (
    vector_store_dependency,
    embeddings_generator_dependency,
    logger_dependency
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


class ChatbotService:
    """
    Servicio de chatbot que maneja las consultas de usuarios
    y genera respuestas usando RAG (Retrieval Augmented Generation).
    """

    def __init__(self, embeddings_generator: EmbeddingsGenerator,
                 vector_store: FAISSVectorStore, logger: logging.Logger):
        self.embeddings_generator = embeddings_generator
        self.vector_store = vector_store
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
            answer_text = best_match.get('text', '')

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
                    related_images.extend(result.get('image_paths', []))

            # Generar respuesta usando LLM
            llm_answer = self._generate_llm_response(answer_text, question)

            return {
                "success": True,
                "answer": llm_answer,
                "sources": sources,
                "images": related_images,
                "total_results": len(results),
                "question": question
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

    def _generate_llm_response(self, context: str, question: str) -> str:
        """
        Genera una respuesta usando el LLM con el contexto encontrado.

        Args:
            context: Texto de contexto encontrado en la búsqueda
            question: Pregunta original del usuario

        Returns:
            Respuesta generada por el LLM
        """
        try:
            llm = ChatGroq(
                model='openai/gpt-oss-120b',
                temperature=0.7,
                api_key=GROQ_API_KEY
            )

            prompt = f"Basándote en este contexto: '{context[:500]}' responde esta pregunta de manera clara y concisa: '{question}'"
            llm_response = llm.invoke(prompt)

            answer = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)
            self.logger.info(f"Respuesta generada por LLM: {answer}")

            return answer

        except Exception as e:
            self.logger.error(f"Error generando respuesta LLM: {e}")
            return context[:500] + "..."  # Fallback al contexto original


# Factory function para el servicio usando FastAPI Depends
def get_chatbot_service(
    embeddings_generator: EmbeddingsGenerator = Depends(embeddings_generator_dependency),
    vector_store: FAISSVectorStore = Depends(vector_store_dependency),
    logger: logging.Logger = Depends(logger_dependency)
) -> ChatbotService:
    """
    Factory function para crear ChatbotService con todas sus dependencias.
    Esto es equivalente a @Autowired en Spring Boot pero usando FastAPI Depends().
    """
    return ChatbotService(
        embeddings_generator=embeddings_generator,
        vector_store=vector_store,
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
        get_logger
    )

    return ChatbotService(
        embeddings_generator=create_embeddings_generator(),
        vector_store=create_vector_store(),
        logger=get_logger()
    )
