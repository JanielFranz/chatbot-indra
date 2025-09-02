from langchain_core.runnables import RunnableLambda

from src.llm.providers.groq_provider import GroqProvider
import logging

def rewrite_user_question(user_question: str) -> str:
    logger = logging.getLogger(__name__)


    try:
        groq_provider = GroqProvider(
            model="llama-3.1-8b-instant",
            temperature=0.1,  # Temperatura baja para respuestas más consistentes
            logger=logger
        )

        rewrite_prompt = f"""
You are a question rewriter for a document retrieval system. Your job is to improve user questions to make them more specific and searchable while keeping them as natural language questions.

IMPORTANT RULES:
1. ALWAYS return a natural language question, never SQL, code, or database syntax
2. Make vague questions more specific and detailed
3. Add context that helps find relevant information in documents
4. Keep the question format (question words like "what", "who", "where", "how", etc.)
5. Do NOT convert to database queries or technical syntax

EXAMPLES:
- Input: "Show me authors" → Output: "Who are the authors mentioned in this document?"
- Input: "What's this about?" → Output: "What is the main topic and purpose of this document?"
- Input: "Tell me more" → Output: "What are the key details and important information in this document?"

USER QUESTION:
{user_question}

Rewrite this question to be more specific and searchable while keeping it as a natural language question:"""

        question_rewritted = groq_provider.generate_response(rewrite_prompt)
        logger.info(f"Rewrite result: {question_rewritted}")

        return question_rewritted
    except Exception as e:
        logger.error(f"Error in question rewriting: {e}. Returning original question.")
        return user_question

rewriter_runnable = RunnableLambda(rewrite_user_question)
