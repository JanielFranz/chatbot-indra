from langchain_core.runnables import RunnableLambda
from sentence_transformers import CrossEncoder
import logging
from typing import Dict, Any

# Inicializar el cross-encoder globalmente para eficiencia
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank_results(context_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reordena los resultados de búsqueda usando un cross-encoder
    para mejorar la relevancia basada en la pregunta del usuario.
    """
    logger = logging.getLogger(__name__)

    try:
        # Extraer datos necesarios
        question = context_data.get("question", "")
        sources = context_data.get("sources", [])

        # Si no hay resultados o solo uno, retornar sin cambios
        if not sources or len(sources) <= 1:
            logger.info("No reranking needed: insufficient results")
            return context_data

        # Extraer textos para reranking
        docs_to_rank = []
        for source in sources:
            text = source.get("text", "").strip()
            if text:
                docs_to_rank.append(text)

        if not docs_to_rank:
            logger.warning("No text content found for reranking")
            return context_data

        # Crear pares [query, doc] para el cross-encoder
        cross_inp = [[question, doc] for doc in docs_to_rank]

        # Obtener scores de relevancia
        scores = cross_encoder.predict(cross_inp)

        # Combinar sources originales con scores y reordenar
        sources_with_scores = list(zip(sources[:len(scores)], scores))
        reranked_sources = sorted(sources_with_scores, key=lambda x: x[1], reverse=True)

        # Extraer solo las sources reordenadas (top 3)
        top_sources = [source for source, score in reranked_sources[:3]]

        # Actualizar context_data con resultados reordenados
        reranked_data = context_data.copy()
        reranked_data["sources"] = top_sources

        # Actualizar contexto principal con el mejor resultado
        if top_sources:
            reranked_data["context"] = top_sources[0].get("text", "")

        # Agregar metadatos de reranking
        reranked_data["reranking_applied"] = True
        reranked_data["original_count"] = len(sources)
        reranked_data["reranked_count"] = len(top_sources)

        logger.info(f"The reranked data is: {reranked_data}")
        logger.info(f"Reranking completed: {len(sources)} -> {len(top_sources)} results")
        return reranked_data

    except Exception as e:
        logger.error(f"Error in reranking: {e}. Using original results.")
        # En caso de error, retornar datos originales sin modificar
        fallback_data = context_data.copy()
        fallback_data["reranking_applied"] = False
        fallback_data["reranking_error"] = str(e)
        return fallback_data

# Crear el runnable
reranker_runnable = RunnableLambda(rerank_results)
