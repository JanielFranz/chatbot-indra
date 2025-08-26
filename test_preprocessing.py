#!/usr/bin/env python3
"""
Script de ejemplo para probar el preprocesamiento del PDF rag-challenge.pdf
y almacenar los datos en FAISS.
"""

import os
import sys
import logging
from pathlib import Path

# Agregar el directorio src al path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.preprocess import PDFPreprocessor, process_pdf_document
from src.database.core_faiss import FAISSVectorStore


def setup_logging():
    """Configurar logging para el ejemplo."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def main():
    """Función principal para probar el preprocesamiento completo."""
    setup_logging()
    logger = logging.getLogger(__name__)

    # Rutas de archivos
    pdf_path = "src/data/rag-challenge.pdf"
    faiss_index_path = "vector_database/rag_index"

    # Verificar que el PDF existe
    if not os.path.exists(pdf_path):
        logger.error(f"PDF no encontrado: {pdf_path}")
        return

    logger.info("🚀 Iniciando preprocesamiento del PDF...")

    try:
        # 1. Procesar el PDF
        logger.info("📄 Extrayendo contenido del PDF...")
        processed_data = process_pdf_document(
            pdf_path=pdf_path,
            chunk_size=500,
            chunk_overlap=50,
            embedding_model="all-MiniLM-L6-v2"
        )

        # Mostrar estadísticas
        logger.info(f"✅ Contenido extraído:")
        logger.info(f"   - Chunks de texto: {processed_data['total_chunks']}")
        logger.info(f"   - Imágenes: {processed_data['total_images']}")
        logger.info(f"   - Dimensión embeddings: {processed_data['embedding_dimension']}")

        # 2. Crear y poblar la base de datos vectorial FAISS
        logger.info("🔍 Creando base de datos vectorial FAISS...")

        vector_store = FAISSVectorStore(
            dimension=processed_data['embedding_dimension'],
            index_type="flat"
        )

        # Preparar metadatos para FAISS
        image_paths = []
        page_numbers = []
        chunk_ids = []

        for i, metadata in enumerate(processed_data['metadata']):
            # Obtener ruta de imagen asociada (si existe)
            associated_images = metadata.get('associated_images', [])
            img_path = associated_images[0]['path'] if associated_images else None
            image_paths.append(img_path)

            page_numbers.append(metadata['page_number'])
            chunk_ids.append(metadata['chunk_id'])

        # Agregar embeddings a FAISS
        ids = vector_store.add_embeddings(
            embeddings=processed_data['embeddings'],
            texts=processed_data['text_chunks'],
            image_paths=image_paths,
            page_numbers=page_numbers,
            chunk_ids=chunk_ids
        )

        logger.info(f"✅ {len(ids)} embeddings agregados a FAISS")

        # 3. Guardar el índice
        os.makedirs("vector_database", exist_ok=True)
        vector_store.save_index(faiss_index_path)
        logger.info(f"💾 Índice guardado en: {faiss_index_path}")

        # 4. Probar búsqueda
        logger.info("🔎 Probando búsqueda de ejemplo...")

        # Crear query de ejemplo
        test_preprocessor = PDFPreprocessor()
        query_text = "What is machine learning?"
        query_embedding = test_preprocessor.embedding_model.encode([query_text])

        # Buscar documentos similares
        distances, results = vector_store.search(
            query_embedding=query_embedding,
            k=3,
            return_metadata=True
        )

        logger.info(f"📋 Resultados de búsqueda para: '{query_text}'")
        for i, result in enumerate(results):
            logger.info(f"   {i+1}. Página {result['page_number']}, Similitud: {result['similarity']:.3f}")
            logger.info(f"      Texto: {result['text'][:100]}...")
            if result.get('image_path'):
                logger.info(f"      Imagen: {result['image_path']}")

        # 5. Mostrar estadísticas finales
        stats = vector_store.get_stats()
        logger.info(f"📊 Estadísticas finales:")
        logger.info(f"   - Total vectores en FAISS: {stats['total_vectors']}")
        logger.info(f"   - Dimensión: {stats['dimension']}")
        logger.info(f"   - Tipo de índice: {stats['index_type']}")

        logger.info("🎉 ¡Preprocesamiento completado exitosamente!")

    except Exception as e:
        logger.error(f"❌ Error durante el preprocesamiento: {str(e)}")
        raise


def demo_load_and_search():
    """Función de demostración para cargar un índice existente y hacer búsquedas."""
    setup_logging()
    logger = logging.getLogger(__name__)

    faiss_index_path = "vector_database/rag_index"

    if not os.path.exists(f"{faiss_index_path}.faiss"):
        logger.error("No se encontró índice FAISS. Ejecuta main() primero.")
        return

    logger.info("📖 Cargando índice FAISS existente...")

    # Cargar índice
    vector_store = FAISSVectorStore()
    vector_store.load_index(faiss_index_path)

    # Crear modelo para queries
    preprocessor = PDFPreprocessor()

    # Queries de ejemplo
    test_queries = [
        "What is artificial intelligence?",
        "How does deep learning work?",
        "What are neural networks?",
        "Explain machine learning algorithms"
    ]

    for query in test_queries:
        logger.info(f"🔍 Búsqueda: '{query}'")

        # Generar embedding del query
        query_embedding = preprocessor.embedding_model.encode([query])

        # Buscar
        distances, results = vector_store.search(
            query_embedding=query_embedding,
            k=2,
            return_metadata=True
        )

        for i, result in enumerate(results):
            logger.info(f"   {i+1}. Página {result['page_number']}, Similitud: {result['similarity']:.3f}")
            logger.info(f"      Texto: {result['text'][:150]}...")

        print("-" * 80)


if __name__ == "__main__":
    # Ejecutar preprocesamiento completo
    main()

    # Opcional: probar cargar y buscar
    print("\n" + "="*80)
    print("DEMO: Cargar índice y realizar búsquedas")
    print("="*80)
    demo_load_and_search()
