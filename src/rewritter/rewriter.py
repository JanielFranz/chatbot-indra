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
You are an expert at rewriting user questions for a retrieval system.
Your task is to transform a potentially short or ambiguous user question into a detailed, standalone query that is optimized for finding information in a document.
The rewritten query should be self-contained and not conversational.

USER QUESTION:
{user_question}

return the rewritten question only.

"""

        question_rewritted = groq_provider.generate_response(rewrite_prompt)
        logger.info(f"Rewrite result: {question_rewritted}")

        return question_rewritted
    except Exception as e:
        logger.error(f"Error in question rewriting: {e}. Returning original question.")
        return user_question

rewriter_runnable = RunnableLambda(rewrite_user_question)


