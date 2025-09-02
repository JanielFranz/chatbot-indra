"""
Gestor de prompts para diferentes tipos de consultas.
"""
import logging
from typing import Dict, Any, Optional
from .templates import PromptTemplates


class PromptManager:
    """
    Gestor centralizado para el manejo de prompts y su configuración.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Inicializa el gestor de prompts.

        Args:
            logger: Logger para registrar eventos
        """
        self.logger = logger or logging.getLogger(__name__)
        self.templates = PromptTemplates()

        # Configuraciones por defecto para diferentes tipos de prompt
        self.default_configs = {
            'rag': {
                'max_context_length': 2000,
                'include_sources': True,
                'language': 'es'
            },
            'summarization': {
                'max_length': 1500,
                'style': 'concise'
            },
            'analysis': {
                'include_keywords': True,
                'include_topic': True
            }
        }

    def create_rag_prompt(self, context: str, question: str, images_length: int,
                         config: Optional[Dict[str, Any]] = None) -> str:
        """
        Crea un prompt optimizado para RAG.

        Args:
            context: Contexto encontrado en la búsqueda
            question: Pregunta del usuario
            config: Configuración personalizada para el prompt

        Returns:
            Prompt formateado y optimizado
        """
        try:
            # Usar configuración por defecto si no se proporciona una
            final_config = self.default_configs['rag'].copy()
            if config:
                final_config.update(config)

            # Formatear el prompt
            prompt = self.templates.format_rag_prompt(
                context=context,
                question=question,
                images_length=images_length,
                max_context_length=final_config['max_context_length']
            )

            self.logger.debug(f"Prompt RAG creado exitosamente (longitud: {len(prompt)})")

            return prompt

        except Exception as e:
            self.logger.error(f"Error creando prompt RAG: {e}")
            # Fallback a un prompt básico
            return f"Basándote en este contexto: '{context[:500]}' responde esta pregunta: '{question}'"


    def _clean_context(self, context: str) -> str:
        """
        Limpia y normaliza el contexto.

        Args:
            context: Contexto raw

        Returns:
            Contexto limpio
        """
        if not context or not isinstance(context, str):
            return "No se encontró contexto relevante."

        # Eliminar espacios extra y caracteres especiales problemáticos
        context = ' '.join(context.split())

        # Limitar longitud máxima
        if len(context) > 3000:
            context = context[:3000] + "..."

        return context

    def _clean_question(self, question: str) -> str:
        """
        Limpia y normaliza la pregunta.

        Args:
            question: Pregunta raw

        Returns:
            Pregunta limpia
        """
        if not question or not isinstance(question, str):
            return "¿Puedes ayudarme con información?"

        # Eliminar espacios extra
        question = ' '.join(question.split())

        # Asegurar que termine con signo de interrogación
        if not question.endswith('?'):
            question += "?"

        return question

    def get_prompt_stats(self, prompt: str) -> Dict[str, Any]:
        """
        Obtiene estadísticas del prompt generado.

        Args:
            prompt: Prompt a analizar

        Returns:
            Diccionario con estadísticas
        """
        return {
            'length': len(prompt),
            'word_count': len(prompt.split()),
            'estimated_tokens': len(prompt) // 4,  # Estimación aproximada
            'has_context': 'CONTEXTO:' in prompt,
            'has_question': 'PREGUNTA' in prompt
        }
