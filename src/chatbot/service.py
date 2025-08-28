from src.database.core_faiss import FAISSVectorStore
from src.utils.generate_embeddings import EmbeddingsGenerator
import logging

from src.container import (
    vector_store_dependency,
    embeddings_generator_dependency
)

class ChatbotService:

    def __init__(self, embeddings_generator: EmbeddingsGenerator,
                 vector_store: FAISSVectorStore, logger: logging.Logger):

        self.embeddings_generator = embeddings_generator
        self.vector_store = vector_store
        self.logger = logger

    def asnwer_user_question(self, question: str) -> str:
        try:
            self.logger.debug(f"question to answer: {question}")

            question_as_embedding = self.embeddings_generator.generate_embedding(question)

            results_from_embbeding = self.vector_store.search(question_as_embedding)

            return results_from_embbeding[0]["text"]

        except Exception as e:
            self.logger.error(e)





