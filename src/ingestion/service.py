from src.database.core_faiss import FAISSVectorStore
from src.ingestion.model import ExtractedContent
from src.utils.preprocess import PDFPreprocessor
from src.utils.generate_embeddings import EmbeddingsGenerator
import numpy as np


def transform_pdf_to_embeddings(file_path: str) -> str :

    pdfPreProcessor = PDFPreprocessor()
    embeddingsGenerator = EmbeddingsGenerator()
    vectorStore = FAISSVectorStore()

    # Process the pdf
    content: ExtractedContent = pdfPreProcessor.extract_content_from_pdf(file_path)

    # Generate embeddings
    embeddings: np.ndarray = embeddingsGenerator.generate_embeddings(content.text_chunks)

    processed_data = {
        "embeddings": embeddings,
        "text_chunks": content.text_chunks,
        "metadata": content.metadata,
        "images": content.images,
        "total_chunks": len(content.text_chunks),
        "total_images": len(content.images),
        "embedding_dimension": embeddings.shape[1] if len(embeddings) > 0 else 0
    }
    self.logger.info(
        f"Procesamiento completado: {processed_data['total_chunks']} chunks, {processed_data['total_images']} imágenes")

    vectorStore.add_embeddings(processed_data)
    self.logger.info(f"Datos almacenados")

    return f"Procesamiento completado: {processed_data['total_chunks']} chunks, {processed_data['total_images']} imágenes"