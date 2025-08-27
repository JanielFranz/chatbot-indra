"""
Script de ejemplo para procesar el PDF rag-challenge usando inyección de dependencias.
Este script demuestra cómo usar el sistema DI fuera de FastAPI endpoints.
"""
import asyncio
import os
from src.ingestion.controller import process_rag_challenge_pdf
from src.ingestion.service import create_ingestion_service


def main():
    """
    Función principal que procesa el PDF del rag-challenge.
    Demuestra el uso del patrón de inyección de dependencias.
    """
    print("🚀 Iniciando procesamiento del PDF rag-challenge...")

    # Verificar que el archivo existe
    pdf_path = "src/data/rag-challenge.pdf"
    if not os.path.exists(pdf_path):
        print(f"❌ Error: No se encontró el archivo {pdf_path}")
        return

    try:
        # Usar el servicio con inyección de dependencias (fuera de FastAPI)
        print("📄 Procesando PDF con inyección de dependencias...")
        service = create_ingestion_service()
        result = service.transform_pdf_to_embeddings(pdf_path)

        print(f"✅ {result}")
        print("🎉 Procesamiento completado exitosamente!")

    except Exception as e:
        print(f"❌ Error durante el procesamiento: {str(e)}")


async def async_main():
    """
    Versión asíncrona usando la función del controlador.
    """
    print("🚀 Iniciando procesamiento asíncrono del PDF rag-challenge...")

    try:
        result = await process_rag_challenge_pdf()
        print(f"✅ {result}")
        print("🎉 Procesamiento asíncrono completado!")

    except Exception as e:
        print(f"❌ Error durante el procesamiento asíncrono: {str(e)}")


if __name__ == "__main__":
    print("=" * 60)
    print("PROCESAMIENTO DE PDF CON INYECCIÓN DE DEPENDENCIAS")
    print("=" * 60)

    # Ejecutar versión síncrona
    main()

    print("\n" + "-" * 40)
    print("Ejecutando versión asíncrona...")
    print("-" * 40)

    # Ejecutar versión asíncrona
    asyncio.run(async_main())
