from langchain_core.runnables import RunnableLambda
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.utils import Input, Output
import logging
import re
from typing import Dict, Any, List
from src.llm.providers.groq_provider import GroqProvider


def output_basic_validations(response_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validaciones básicas de salida sin LLM.
    """
    if not isinstance(response_data, dict):
        raise ValueError("❌ Response must be a dictionary.")

    answer = response_data.get('answer', '')

    if not answer or len(answer.strip()) == 0:
        raise ValueError("❌ Response cannot be empty.")

    # Verificar longitud mínima
    if len(answer.strip()) < 10:
        raise ValueError("❌ Response too short, must be at least 10 characters.")

    # Verificar longitud máxima
    if len(answer) > 5000:
        raise ValueError("❌ Response too long, must be under 5000 characters.")

    # Limpiar la respuesta
    response_data['answer'] = answer.strip()

    return response_data


def sanitize_content(response_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitiza el contenido de la respuesta removiendo elementos potencialmente problemáticos.
    """
    answer = response_data.get('answer', '')

    # Remover HTML tags si existen
    answer = re.sub(r'<[^>]+>', '', answer)

    # Remover URLs maliciosas potenciales
    answer = re.sub(r'http[s]?://[^\s]+', '[URL removed for security]', answer)

    # Remover secuencias de caracteres repetitivos excesivos
    answer = re.sub(r'(.)\1{10,}', r'\1\1\1', answer)

    # Limpiar espacios múltiples
    answer = re.sub(r'\s+', ' ', answer)

    response_data['answer'] = answer.strip()

    return response_data


def output_validations_with_llm(response_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validaciones avanzadas de salida usando LLM de Groq.
    Verifica calidad, coherencia y seguridad de la respuesta.
    """
    logger = logging.getLogger(__name__)

    # Primero aplicar validaciones básicas
    validated_response = output_basic_validations(response_data)

    try:
        # Inicializar proveedor de Groq
        groq_provider = GroqProvider(
            model="llama-3.1-8b-instant",
            temperature=0.1,  # Temperatura baja para respuestas más consistentes
            logger=logger
        )

        answer = validated_response.get('answer', '')
        question = validated_response.get('question', 'Unknown question')

        # Prompt para validación de contenido de salida
        validation_prompt = f"""
You are an output validation system for a document Q&A chatbot. Analyze the following response and determine if it's appropriate and helpful.

Original Question: "{question}"
Generated Answer: "{answer}"

Check for:
1. Inappropriate content (offensive language, harmful advice)
2. Answer coherence and relevance to the question
3. Professional tone and helpfulness
4. Potential misinformation or hallucinations
5. Completeness of the response

Respond with ONLY one of these:
- VALID: if the response is appropriate and helpful
- INVALID_INAPPROPRIATE: if contains inappropriate content
- INVALID_INCOHERENT: if the answer doesn't make sense or is irrelevant
- INVALID_UNPROFESSIONAL: if tone is unprofessional or unhelpful
- INVALID_MISINFORMATION: if contains potential misinformation
- INVALID_INCOMPLETE: if the response is too vague or incomplete

Response:"""

        response = groq_provider.generate_response(validation_prompt)
        validation_result = response.strip().upper()

        logger.info(f"LLM output validation result: {validation_result} for response length: {len(answer)}")

        # Procesar resultado de validación
        if validation_result == "VALID":
            validated_response['validation_status'] = 'passed'
            validated_response['validation_message'] = 'Response passed all validation checks'
            return validated_response
        elif validation_result == "INVALID_INAPPROPRIATE":
            raise ValueError("❌ Response contains inappropriate content. Please try rephrasing your question.")
        elif validation_result == "INVALID_INCOHERENT":
            raise ValueError("❌ Response is incoherent or irrelevant. Please try asking a more specific question.")
        elif validation_result == "INVALID_UNPROFESSIONAL":
            raise ValueError("❌ Response tone is unprofessional. Please try again.")
        elif validation_result == "INVALID_MISINFORMATION":
            raise ValueError("❌ Response may contain misinformation. Please verify the information independently.")
        elif validation_result == "INVALID_INCOMPLETE":
            raise ValueError("❌ Response is too incomplete. Please try asking a more specific question.")
        else:
            # Si la respuesta no es reconocida, usar validación básica como fallback
            logger.warning(f"Unrecognized LLM validation result: {validation_result}. Using basic validation.")
            validated_response['validation_status'] = 'basic_only'
            validated_response['validation_message'] = 'LLM validation inconclusive, basic validation passed'
            return validated_response

    except ValueError:
        # Re-raise validation errors
        raise
    except Exception as e:
        logger.error(f"Error in LLM output validation: {e}. Falling back to basic validation.")
        # En caso de error con el LLM, usar solo validaciones básicas
        validated_response['validation_status'] = 'basic_only'
        validated_response['validation_message'] = f'LLM validation failed: {str(e)}, basic validation passed'
        return validated_response


def enhance_response_quality(response_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Mejora la calidad de la respuesta agregando estructura y metadatos útiles.
    """
    logger = logging.getLogger(__name__)

    try:
        groq_provider = GroqProvider(
            model="llama-3.1-8b-instant",
            temperature=0.2,
            logger=logger
        )

        answer = response_data.get('answer', '')
        question = response_data.get('question', '')

        enhancement_prompt = f"""
You are a response enhancement system. Improve the following answer by making it more structured and helpful while keeping the core content unchanged.

Original Question: "{question}"
Original Answer: "{answer}"

Enhance the answer by:
1. Adding clear structure with bullet points or numbering if appropriate
2. Ensuring proper formatting
3. Adding helpful context if missing
4. Making it more readable and professional

Return ONLY the enhanced answer without any additional commentary:"""

        enhanced_answer = groq_provider.generate_response(enhancement_prompt)

        # Solo usar la respuesta mejorada si es significativamente mejor
        if len(enhanced_answer.strip()) > len(answer) * 0.8:  # Al menos 80% del tamaño original
            response_data['answer'] = enhanced_answer.strip()
            response_data['enhanced'] = True
            response_data['enhancement_applied'] = 'llm_enhancement'
            logger.info("Response enhanced successfully using LLM")
        else:
            response_data['enhanced'] = False
            response_data['enhancement_applied'] = 'none'
            logger.warning("LLM enhancement produced insufficient result, keeping original")

        return response_data

    except Exception as e:
        logger.error(f"Error in response enhancement: {e}")
        response_data['enhanced'] = False
        response_data['enhancement_applied'] = 'failed'
        return response_data




# Crear runnables para usar en chains
output_basic_validations_runnable = RunnableLambda(output_basic_validations)
sanitize_content_runnable = RunnableLambda(sanitize_content)
output_validations_with_llm_runnable = RunnableLambda(output_validations_with_llm)
enhance_response_quality_runnable = RunnableLambda(enhance_response_quality)



# Chain completo: validación + mejora
full_output_guardrail_chain = (
    RunnablePassthrough()
    | output_basic_validations_runnable      # Validaciones básicas
    | sanitize_content_runnable              # Sanitización
    | output_validations_with_llm_runnable   # Validaciones LLM
    | enhance_response_quality_runnable      # Mejora de calidad

)
