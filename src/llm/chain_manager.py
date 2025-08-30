"""
Chain Manager - Orquestador principal para el manejo de LLMs.
"""
import logging
from typing import Dict, Any, Optional, List
from .providers.groq_provider import GroqProvider
from .prompts.prompt_manager import PromptManager


class LLMChainManager:
    """
    Orquestador principal para el manejo de cadenas LLM.
    Gestiona proveedores, prompts y la generación de respuestas.
    """

    def __init__(self,
                 provider: Optional[GroqProvider] = None,
                 prompt_manager: Optional[PromptManager] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Inicializa el Chain Manager.

        Args:
            provider: Proveedor LLM (por defecto Groq)
            prompt_manager: Gestor de prompts
            logger: Logger para registrar eventos
        """
        self.logger = logger or logging.getLogger(__name__)

        # Inicializar componentes
        self.provider = provider or self._create_default_provider()
        self.prompt_manager = prompt_manager or PromptManager(logger=self.logger)

        self.logger.info("LLMChainManager inicializado exitosamente")

    def _create_default_provider(self) -> GroqProvider:
        """
        Crea el proveedor por defecto (Groq).

        Returns:
            Instancia de GroqProvider configurada
        """
        try:
            return GroqProvider(
                model="llama3-8b-8192",
                temperature=0.7,
                logger=self.logger
            )
        except Exception as e:
            self.logger.error(f"Error creando proveedor por defecto: {e}")
            raise

    def generate_rag_response(self, context: str, question: str,
                            config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Genera una respuesta RAG completa.

        Args:
            context: Contexto encontrado en la búsqueda
            question: Pregunta del usuario
            config: Configuración personalizada

        Returns:
            Diccionario con la respuesta y metadatos
        """
        try:
            self.logger.debug(f"Generando respuesta RAG para: {question[:50]}...")

            # Crear el prompt optimizado
            prompt = self.prompt_manager.create_rag_prompt(
                context=context,
                question=question,
                config=config
            )

            # Obtener estadísticas del prompt
            prompt_stats = self.prompt_manager.get_prompt_stats(prompt)

            # Generar respuesta usando el proveedor
            response = self.provider.generate_response(prompt)

            # Preparar resultado
            result = {
                "success": True,
                "answer": response,
                "prompt_stats": prompt_stats,
                "provider": self.provider.model,
                "context_length": len(context),
                "question": question
            }

            self.logger.info("Respuesta RAG generada exitosamente")

            return result

        except Exception as e:
            self.logger.error(f"Error generando respuesta RAG: {e}")

            # Respuesta de fallback
            fallback_response = self._create_fallback_response(context, question, str(e))

            return {
                "success": False,
                "error": str(e),
                "answer": fallback_response,
                "fallback": True,
                "question": question
            }

    def _create_fallback_response(self, context: str, question: str, error: str) -> str:
        """
        Crea una respuesta de fallback cuando el LLM falla.

        Args:
            context: Contexto original
            question: Pregunta original
            error: Error ocurrido

        Returns:
            Respuesta de fallback basada en el contexto
        """
        self.logger.warning(f"Creando respuesta fallback debido a: {error}")

        # Si hay contexto, devolver una porción del mismo
        if context and len(context.strip()) > 0:
            # Tomar los primeros 500 caracteres del contexto como respuesta
            fallback = context[:500].strip()
            if len(context) > 500:
                fallback += "..."

            return f"Basándome en la información disponible: {fallback}"

        # Si no hay contexto útil
        return "Lo siento, no pude procesar tu pregunta en este momento. Por favor, intenta reformularla o intenta más tarde."

    def is_healthy(self) -> Dict[str, Any]:
        """
        Verifica el estado de salud del chain manager.

        Returns:
            Diccionario con el estado de cada componente
        """
        health_status = {
            "overall": True,
            "components": {
                "provider": False,
                "prompt_manager": False
            },
            "errors": []
        }

        try:
            # Verificar proveedor
            if self.provider and self.provider.is_available():
                health_status["components"]["provider"] = True
            else:
                health_status["errors"].append("Provider no disponible")

            # Verificar prompt manager
            if self.prompt_manager:
                health_status["components"]["prompt_manager"] = True
            else:
                health_status["errors"].append("Prompt manager no disponible")

            # Estado general
            health_status["overall"] = all(health_status["components"].values())

        except Exception as e:
            health_status["overall"] = False
            health_status["errors"].append(f"Error en health check: {str(e)}")

        return health_status

    def get_provider_info(self) -> Dict[str, Any]:
        """
        Obtiene información del proveedor actual.

        Returns:
            Información del proveedor
        """
        if not self.provider:
            return {"error": "No hay proveedor configurado"}

        return {
            "model": getattr(self.provider, 'model', 'unknown'),
            "temperature": getattr(self.provider, 'temperature', 'unknown'),
            "available": self.provider.is_available(),
            "type": "groq"
        }
