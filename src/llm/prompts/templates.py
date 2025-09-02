"""
Plantillas de prompts para el sistema RAG.
"""
from typing import Dict, Any


class PromptTemplates:
    """
    Clase que contiene las plantillas de prompts para diferentes casos de uso.
    """

    @staticmethod
    def get_rag_prompt_template() -> str:
        """
        Plantilla para generar respuestas usando RAG.

        Returns:
            Template string para RAG
        """
        return """You are an expert assistant that answers questions based only on the provided context.

CONTEXT:
{context}

USER QUESTION:
{question}

IMAGES LENGTH:
{images_length}

INSTRUCTIONS:
1. Answer only using the provided context.
2. If you cannot answer with the given context, clearly state that you do not have enough information.
3. Be precise, clear, and concise.
4. If there is specific information such as numbers, dates, or names, include them exactly as they appear in the context.
5. Structure your answer logically and make it easy to understand.
6. If the image length is greater than 0, naturally mention that there are related images available for the user to view, which may provide additional visual context to support your answer.

ANSWER:"""

    @staticmethod
    def format_rag_prompt(context: str, question: str, images_length: int,  max_context_length: int = 2000) -> str:
        """
        Formatea el prompt RAG con contexto y pregunta específicos.

        Args:
            context: Contexto encontrado en la búsqueda
            question: Pregunta del usuario
            max_context_length: Longitud máxima del contexto

        Returns:
            Prompt formateado listo para usar
        """
        # Truncar contexto si es muy largo
        if len(context) > max_context_length:
            context = context[:max_context_length] + "..."

        template = PromptTemplates.get_rag_prompt_template()
        return template.format(context=context, question=question, images_length= images_length)
