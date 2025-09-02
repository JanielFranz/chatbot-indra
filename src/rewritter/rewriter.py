from langchain_core.runnables import RunnableLambda

from src.llm.providers.groq_provider import GroqProvider
import logging

def rewrite_user_question(user_question: str) -> str:
    logger = logging.getLogger(__name__)

    document_context = (f"The document discusses the challenges associated with "
                        f"unstructured data in various types of documents. "
                        f"Key points include:\n\n1. **Types of Documents Affected**: "
                        f"The document highlights the issue of unstructured data in "
                        f"millions of documents, such as:\n\t* Invoices\n\t* Contracts\n\t* Insurance claims\n\t* Medical records\n\t* Financial statements\n2. "
                        f"**Untapped Data Insights**: It is estimated that 80-90% of the data in these documents remains untapped, containing valuable insights that could benefit businesses.\n3. **Manual Data Entry Challenges**: "
                        f"The document mentions the problem of manual data entry from:\n\t* PDFs\n\t* Scanned images\n\t* Forms\n\nThis "
                        f"highlights the need for efficient and automated solutions to extract and utilize the valuable information hidden within these documents.")
    try:
        groq_provider = GroqProvider(
            model="llama-3.1-8b-instant",
            temperature=0.1,  # Temperatura baja para respuestas más consistentes
            logger=logger
        )

        rewrite_prompt = f"""
You are a query rewriter. Rewrite this user question so it is complete,
unambiguous, and relevant to the context of the document.

PDF CONTEXT:
{document_context}

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


