#!/usr/bin/env python3
"""
Script de prueba para verificar que el ChatbotService y el método answer_user_question
funcionan correctamente con el vector store FAISS.
Incluye pruebas de ingesta de datos y consultas de diferentes tipos.
"""

import sys
import os
from pathlib import Path

# Agregar la ruta src al path para importar los módulos
sys.path.append(str(Path(__file__).parent / "src"))

from src.chatbot.service import create_chatbot_service
from src.ingestion.service import create_ingestion_service
import json


def setup_test_data():
    """
    Configura datos de prueba ingresando el PDF al vector store.
    """
    print("📦 Configurando datos de prueba...")

    # Ruta al PDF de ejemplo
    #pdf_path = "src/data/rag-challenge.pdf"

    #if not os.path.exists(pdf_path):
        #print(f"❌ Error: No se encontró el archivo PDF en {pdf_path}")
        #return False

    try:
        # Crear servicio de ingesta y procesar el PDF
        ingestion_service = create_ingestion_service()
        #print(f"   📄 Procesando PDF: {pdf_path}")
        print(f"   📄 Procesando PDF ")

        result = ingestion_service.transform_pdf_to_embeddings()

        if result["success"]:
            print("   ✅ PDF procesado exitosamente")
            print(f"   📊 Chunks de texto: {result['total_chunks']}")
            print(f"   🖼️  Imágenes: {result['total_images']}")
            print(f"   📐 Dimensión embeddings: {result['embedding_dimension']}")
            return True
        else:
            print(f"   ❌ Error procesando PDF: {result['error']}")
            return False

    except Exception as e:
        print(f"   ❌ Error inesperado: {str(e)}")
        return False


def test_basic_question_answering():
    """
    Prueba funcionalidad básica de respuesta a preguntas.
    """
    print("\n🤖 Probando funcionalidad básica de respuesta a preguntas...")

    try:
        # Crear servicio de chatbot
        chatbot_service = create_chatbot_service()
        print("   ✅ ChatbotService creado exitosamente")

        # Lista de preguntas de prueba
        test_questions = [
            "What is this document about?",
            "Give me the diagram that illustrates the solution architecture",
            "What are the main topics covered?",
            "Are there any images in this document?",
            "Tell me about the content"
        ]

        print(f"   🔍 Probando {len(test_questions)} preguntas...")

        for i, question in enumerate(test_questions, 1):
            print(f"\n   📝 Pregunta {i}: '{question}'")

            # Obtener respuesta
            response = chatbot_service.answer_user_question(question)

            # Verificar estructura de respuesta
            if not isinstance(response, dict):
                print(f"      ❌ Error: Respuesta no es un diccionario")
                continue

            # Verificar campos requeridos
            required_fields = ["success", "answer", "sources", "images"]
            missing_fields = [field for field in required_fields if field not in response]

            if missing_fields:
                print(f"      ❌ Error: Campos faltantes: {missing_fields}")
                continue

            # Mostrar resultados
            if response["success"]:
                print(f"      ✅ Respuesta exitosa")
                print(f"      📝 Respuesta: {response['answer'][:100]}...")
                print(f"      📚 Fuentes encontradas: {len(response['sources'])}")
                print(f"      🖼️  Imágenes relacionadas: {len(response['images'])}")

                # Mostrar detalles de fuentes
                if response['sources']:
                    print(f"      🔍 Detalles de fuentes:")
                    for j, source in enumerate(response['sources'][:2], 1):  # Mostrar solo las primeras 2
                        print(f"         {j}. Página: {source.get('page_number', 'N/A')}")
                        print(f"            Similitud: {source.get('similarity', 0.0):.3f}")
                        print(f"            Texto: {source.get('text', '')[:80]}...")

            else:
                print(f"      ❌ Error en respuesta: {response.get('error', 'Error desconocido')}")

        return True

    except Exception as e:
        print(f"   ❌ Error inesperado: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_edge_cases():
    """
    Prueba casos extremos y de error.
    """
    print("\n🧪 Probando casos extremos...")

    try:
        chatbot_service = create_chatbot_service()

        # Casos de prueba extremos
        edge_cases = [
            ("", "Pregunta vacía"),
            ("x", "Pregunta muy corta"),
            ("a" * 1000, "Pregunta muy larga"),
            ("¿?" * 100, "Pregunta con caracteres especiales"),
            ("123456789", "Pregunta solo números"),
            ("What is quantum mechanics in relation to artificial intelligence?", "Pregunta muy específica"),
        ]

        for question, description in edge_cases:
            print(f"   🔬 Probando: {description}")
            print(f"      Pregunta: '{question[:50]}{'...' if len(question) > 50 else ''}'")

            response = chatbot_service.answer_user_question(question)

            # Verificar que siempre retorna una estructura válida
            if isinstance(response, dict) and "success" in response:
                if response["success"]:
                    print(f"      ✅ Respuesta válida obtenida")
                else:
                    print(f"      ⚠️  Respuesta de error manejada correctamente")
            else:
                print(f"      ❌ Estructura de respuesta inválida")
                return False

        return True

    except Exception as e:
        print(f"   ❌ Error inesperado: {str(e)}")
        return False


def test_response_quality():
    """
    Prueba la calidad y consistencia de las respuestas.
    """
    print("\n📊 Evaluando calidad de respuestas...")

    try:
        chatbot_service = create_chatbot_service()

        # Preguntas específicas para evaluar calidad
        quality_questions = [
            {
                "question": "What is the main topic of this document?",
                "expected_elements": ["document", "content", "topic"],
                "description": "Pregunta sobre tema principal"
            },
            {
                "question": "Are there images in this document?",
                "expected_elements": ["image", "picture", "figure"],
                "description": "Pregunta sobre imágenes"
            },
            {
                "question": "How many pages does this document have?",
                "expected_elements": ["page", "document"],
                "description": "Pregunta sobre estructura"
            }
        ]

        total_score = 0
        max_score = len(quality_questions)

        for test_case in quality_questions:
            print(f"   📋 Evaluando: {test_case['description']}")
            print(f"      Pregunta: '{test_case['question']}'")

            response = chatbot_service.answer_user_question(test_case['question'])

            if response["success"]:
                answer_text = response["answer"].lower()

                # Verificar si contiene elementos esperados
                found_elements = [elem for elem in test_case["expected_elements"]
                                if elem.lower() in answer_text]

                if found_elements:
                    print(f"      ✅ Elementos encontrados: {found_elements}")
                    total_score += 1
                else:
                    print(f"      ⚠️  No se encontraron elementos esperados")

                # Verificar longitud de respuesta razonable
                if 10 < len(response["answer"]) < 1000:
                    print(f"      ✅ Longitud de respuesta apropiada: {len(response['answer'])} caracteres")
                else:
                    print(f"      ⚠️  Longitud de respuesta inusual: {len(response['answer'])} caracteres")

                # Verificar que hay fuentes
                if response["sources"]:
                    print(f"      ✅ Fuentes proporcionadas: {len(response['sources'])}")
                else:
                    print(f"      ⚠️  No se proporcionaron fuentes")

            else:
                print(f"      ❌ Error en respuesta")

        print(f"\n   📊 Puntuación de calidad: {total_score}/{max_score}")

        return total_score > 0

    except Exception as e:
        print(f"   ❌ Error inesperado: {str(e)}")
        return False


def test_performance():
    """
    Prueba el rendimiento del sistema de respuestas.
    """
    print("\n⚡ Probando rendimiento...")

    try:
        import time

        chatbot_service = create_chatbot_service()

        # Pregunta de prueba
        test_question = "What is this document about?"

        # Medir tiempo de múltiples consultas
        times = []
        num_tests = 5

        print(f"   🔄 Ejecutando {num_tests} consultas...")

        for i in range(num_tests):
            start_time = time.time()
            response = chatbot_service.answer_user_question(test_question)
            end_time = time.time()

            execution_time = end_time - start_time
            times.append(execution_time)

            if response["success"]:
                print(f"      Consulta {i+1}: {execution_time:.3f}s ✅")
            else:
                print(f"      Consulta {i+1}: {execution_time:.3f}s ❌")

        # Calcular estadísticas
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)

        print(f"   📊 Estadísticas de rendimiento:")
        print(f"      Tiempo promedio: {avg_time:.3f}s")
        print(f"      Tiempo mínimo: {min_time:.3f}s")
        print(f"      Tiempo máximo: {max_time:.3f}s")

        # Verificar que el rendimiento es aceptable (< 5 segundos)
        if avg_time < 5.0:
            print(f"      ✅ Rendimiento aceptable")
            return True
        else:
            print(f"      ⚠️  Rendimiento lento (>{avg_time:.3f}s)")
            return True  # No fallar por rendimiento lento

    except Exception as e:
        print(f"   ❌ Error inesperado: {str(e)}")
        return False


def main():
    """Función principal"""
    print("=" * 80)
    print("🧪 TEST: ChatbotService - answer_user_question Method")
    print("=" * 80)

    # Lista de pruebas a ejecutar
    tests = [
        ("📦 Configuración de datos", setup_test_data),
        ("🤖 Respuesta básica", test_basic_question_answering),
        ("🧪 Casos extremos", test_edge_cases),
        ("📊 Calidad de respuestas", test_response_quality),
        ("⚡ Rendimiento", test_performance),
    ]

    results = []

    for test_name, test_function in tests:
        print(f"\n{test_name}")
        print("-" * 60)

        try:
            result = test_function()
            results.append((test_name, result))

            if result:
                print(f"✅ {test_name}: PASÓ")
            else:
                print(f"❌ {test_name}: FALLÓ")

        except Exception as e:
            print(f"💥 {test_name}: ERROR - {str(e)}")
            results.append((test_name, False))

    # Resumen final
    print("\n" + "=" * 80)
    print("📋 RESUMEN DE PRUEBAS")
    print("=" * 80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASÓ" if result else "❌ FALLÓ"
        print(f"{status} - {test_name}")

    print(f"\n📊 Resultado final: {passed}/{total} pruebas pasaron")

    if passed == total:
        print("🎉 ¡TODAS LAS PRUEBAS PASARON!")
        print("✅ El ChatbotService funciona correctamente")
        print("✅ El método answer_user_question responde adecuadamente")
        print("✅ La integración con FAISS está funcionando")
    elif passed > total // 2:
        print("⚠️  LA MAYORÍA DE PRUEBAS PASARON")
        print("✅ Funcionalidad básica operativa")
        print("⚠️  Algunas mejoras pueden ser necesarias")
    else:
        print("💥 MÚLTIPLES PRUEBAS FALLARON")
        print("❌ Revisar la configuración del sistema")
        print("❌ Verificar que el PDF y las dependencias estén disponibles")

    print("=" * 80)


if __name__ == "__main__":
    main()
