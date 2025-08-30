from openai import api_key

from src.database.core_faiss import FAISSVectorStore
from src.utils.generate_embeddings import EmbeddingsGenerator
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
            self.logger.debug(f"question to answer: {question}")

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

            llm = ChatGroq(
                model='openai/gpt-oss-120b',
                temperature=0.7,
                api_key=GROQ_API_KEY
            )

            try:
                test_prompt = f"Following this context: '{answer_text[:500]}' answer this question quickly and concisely: '{question}'"
                llm_response = llm.invoke(test_prompt)
                llm_answer = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)
                self.logger.info(f"answer from LLM: {llm_answer}")
            except Exception as e:
                self.logger.error(e)

            return {
                "success": True,
                "answer": answer_text,
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


# Factory function para el servicio usando FastAPI Depends
def get_chatbot_service(
    embeddings_generator: EmbeddingsGenerator = Depends(embeddings_generator_dependency),
    vector_store: FAISSVectorStore = Depends(vector_store_dependency),
    logger: logging.Logger = Depends(logger_dependency)
) -> ChatbotService:
    """
    Factory function para crear ChatbotService con todas sus dependencias.
    Esto es equivalente a @Autowired en Spring Boot pero usando FastAPI Depends().
    SOLO funciona dentro de endpoints de FastAPI.
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
    Usa las funciones de factory directamente sin Depends().
    Similar a ApplicationContext.getBean() en Spring Boot.
    """
    from src.container import get_embeddings_generator, get_vector_store, get_logger

    return ChatbotService(
        embeddings_generator=get_embeddings_generator(),
        vector_store=get_vector_store(),
        logger=get_logger()
    )
