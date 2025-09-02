from src.database.core_faiss import FAISSVectorStore
from src.guardrails.input.input_guardrail import input_guardrail_validation_chain
from src.guardrails.output.output_guardrail import full_output_guardrail_chain
from src.rerankers.reranker import reranker_runnable
from src.rewritter.rewriter import rewriter_runnable
from src.utils.embeddings.generator import EmbeddingsGenerator
from src.llm.chain_manager import LLMChainManager
import logging
from typing import Dict, Any
from fastapi import Depends
import os
from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

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
    y genera respuestas usando RAG (Retrieval Augmented Generation) con guardrails integrados.
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

        # Crear chain completo al inicializar
        self._complete_rag_chain = self._create_complete_rag_chain()

    def _create_complete_rag_chain(self):
        """
        Crea una chain completa que incluye: validación entrada -> RAG -> validación salida.
        """

        def retrieve_context(validated_question: str) -> Dict[str, Any]:
            """Busca contexto en el vector store para la pregunta validada."""
            try:
                # Generar embedding
                question_embedding = self.embeddings_generator.generate_embedding(validated_question)

                # Buscar en vector store
                distances, results = self.vector_store.search(
                    question_embedding,
                    k=3,
                    return_metadata=True
                )

                if not results:
                    return {
                        "question": validated_question,
                        "context": "",
                        "sources": [],
                        "images": [],
                        "has_results": False
                    }

                # Procesar resultados
                best_match = results[0]
                context_text = best_match.get('text', '')

                # Recopilar fuentes e imágenes
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

                    # Extraer imágenes
                    if 'associated_images' in result and result['associated_images'] > 0:
                        image_info = result.get('image_info', [])
                        for img_info in image_info:
                            if isinstance(img_info, dict) and 'image_path' in img_info:
                                related_images.append(img_info['image_path'])
                            elif isinstance(img_info, str):
                                related_images.append(img_info)

                # Remover duplicados
                related_images = list(dict.fromkeys(related_images))

                return {
                    "question": validated_question,
                    "context": context_text,
                    "sources": sources,
                    "images": related_images,
                    "has_results": True,
                    "total_results": len(results)
                }

            except Exception as e:
                self.logger.error(f"Error in retrieve_context: {e}")
                return {
                    "question": validated_question,
                    "context": "",
                    "sources": [],
                    "images": [],
                    "has_results": False,
                    "error": str(e)
                }

        def generate_llm_response(context_data: Dict[str, Any]) -> Dict[str, Any]:
            """Genera respuesta usando LLM con el contexto recuperado."""
            try:
                question = context_data.get("question", "")
                context = context_data.get("context", "")
                has_results = context_data.get("has_results", False)

                if not has_results:
                    # Sin resultados - crear respuesta por defecto
                    answer = "Lo siento, no pude encontrar información relevante para responder tu pregunta. Por favor, intenta reformular tu pregunta o verifica que esté relacionada con el contenido del documento."
                    llm_metadata = {"fallback_used": True, "reason": "no_context"}
                else:
                    # Generar respuesta con LLM
                    llm_response = self.llm_chain_manager.generate_rag_response(
                        context=context,
                        question=question,
                        config={'max_context_length': 2000}
                    )

                    if llm_response.get('success', False):
                        answer = llm_response['answer']
                        llm_metadata = {
                            "provider": llm_response.get('provider', 'unknown'),
                            "prompt_stats": llm_response.get('prompt_stats', {}),
                            "fallback_used": llm_response.get('fallback', False),
                            "enhanced": llm_response.get('enhanced', False)
                        }
                    else:
                        # Fallback si LLM falla
                        answer = llm_response.get('answer', context[:500] + "...")
                        llm_metadata = {
                            "provider": llm_response.get('provider', 'unknown'),
                            "error": llm_response.get('error', 'Unknown error'),
                            "fallback_used": True,
                            "reason": "llm_error"
                        }
                        self.logger.warning(f"LLM falló, usando fallback: {llm_response.get('error', 'Unknown error')}")

                # Preparar respuesta para output guardrail
                return {
                    "success": has_results,
                    "answer": answer,
                    "question": question,
                    "sources": context_data.get("sources", []),
                    "images": context_data.get("images", []),
                    "total_results": context_data.get("total_results", 0),
                    "llm_metadata": llm_metadata,
                    "has_results": has_results
                }

            except Exception as e:
                self.logger.error(f"Error in generate_llm_response: {e}")
                return {
                    "success": False,
                    "answer": "Lo siento, ocurrió un error al procesar tu pregunta.",
                    "question": context_data.get("question", ""),
                    "sources": [],
                    "images": [],
                    "total_results": 0,
                    "llm_metadata": {"error": str(e), "fallback_used": True},
                    "has_results": False
                }

        # Crear chain completo
        complete_chain = (
            input_guardrail_validation_chain          # 1. Validar entrada
            | rewriter_runnable
            | RunnableLambda(retrieve_context)        # 2. Buscar contexto
            | reranker_runnable
            | RunnableLambda(generate_llm_response)   # 3. Generar respuesta LLM
            | full_output_guardrail_chain             # 4. Validar y mejorar salida
        )

        return complete_chain

    def answer_user_question(self, question: str) -> Dict[str, Any]:
        """
        Responde una pregunta del usuario usando la chain completa integrada.

        Args:
            question (str): Pregunta del usuario

        Returns:
            Dict con la respuesta validada, metadatos e imágenes relacionadas
        """
        try:
            self.logger.debug(f"Procesando pregunta con chain completa: {question}")

            # Ejecutar chain completo
            result = self._complete_rag_chain.invoke(question)

            # Agregar metadatos adicionales
            result["original_question"] = question
            result["chains_used"] = {
                "input_validation": True,
                "context_retrieval": True,
                "llm_generation": True,
                "output_validation": True
            }

            self.logger.info(f"Pregunta procesada exitosamente usando chain completa")
            return result

        except ValueError as e:
            # Error de validación (input o output guardrails)
            self.logger.warning(f"Validation error: {e}")
            return {
                "success": False,
                "error": "validation_error",
                "answer": str(e),
                "question": question,
                "sources": [],
                "images": [],
                "validation_failed": True,
                "chains_used": {
                    "input_validation": "failed" if "Input" in str(e) else "passed",
                    "output_validation": "failed" if "Response" in str(e) else "not_applied"
                }
            }
        except Exception as e:
            # Error técnico general
            self.logger.error(f"Error in answer_user_question: {e}")
            return {
                "success": False,
                "error": str(e),
                "answer": "Lo siento, ocurrió un error técnico al procesar tu pregunta.",
                "question": question,
                "sources": [],
                "images": [],
                "chains_used": {"error": True}
            }

    def ask_simple_question(self, question: str) -> Dict[str, Any]:
        """
        Hace una pregunta simple usando solo validación de entrada, LLM directo y validación de salida.

        Args:
            question (str): Pregunta del usuario

        Returns:
            Dict con la respuesta validada
        """
        try:
            self.logger.debug(f"Procesando pregunta simple: {question}")

            # Chain simple: validación entrada -> LLM directo -> validación salida
            def simple_llm_response(validated_question: str) -> Dict[str, Any]:
                qa_response = self.llm_chain_manager.ask_question(validated_question)
                return {
                    "success": qa_response.get('success', True),
                    "answer": qa_response.get('answer', 'No se pudo generar una respuesta.'),
                    "question": validated_question,
                    "provider": qa_response.get('provider', 'unknown')
                }

            simple_chain = (
                input_guardrail_validation_chain
                | RunnableLambda(simple_llm_response)
                | full_output_guardrail_chain
            )

            result = simple_chain.invoke(question)
            result["original_question"] = question
            result["chains_used"] = {
                "input_validation": True,
                "simple_qa": True,
                "output_validation": True
            }

            return result

        except ValueError as e:
            return {
                "success": False,
                "error": "validation_error",
                "answer": str(e),
                "question": question,
                "validation_failed": True
            }
        except Exception as e:
            self.logger.error(f"Error in simple question: {e}")
            return {
                "success": False,
                "error": str(e),
                "answer": "Lo siento, ocurrió un error al procesar tu pregunta.",
                "question": question
            }

    def get_health_status(self) -> Dict[str, Any]:
        """
        Obtiene el estado de salud del servicio y sus chains.

        Returns:
            Estado de salud completo
        """
        try:
            # Probar chain completo con pregunta de prueba
            test_result = self._test_complete_chain()

            # Estado de componentes individuales
            llm_health = self.llm_chain_manager.is_healthy()
            vector_store_health = {"available": self.vector_store is not None}
            embeddings_health = {"available": self.embeddings_generator is not None}

            overall_health = (
                test_result.get("working", False) and
                llm_health.get("overall", False) and
                vector_store_health.get("available", False) and
                embeddings_health.get("available", False)
            )

            return {
                "overall": overall_health,
                "complete_chain": test_result,
                "components": {
                    "llm_chain_manager": llm_health,
                    "vector_store": vector_store_health,
                    "embeddings_generator": embeddings_health
                }
            }

        except Exception as e:
            self.logger.error(f"Error checking health status: {e}")
            return {
                "overall": False,
                "error": str(e),
                "complete_chain": {"working": False, "error": str(e)}
            }

    def _test_complete_chain(self) -> Dict[str, Any]:
        """Prueba el funcionamiento del chain completo."""
        try:
            test_question = "What is this document about?"
            result = self._complete_rag_chain.invoke(test_question)
            return {
                "working": True,
                "last_test": "passed",
                "test_question": test_question,
                "response_received": bool(result.get("answer"))
            }
        except Exception as e:
            return {
                "working": False,
                "last_test": "failed",
                "error": str(e)
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
