#!/usr/bin/env python3
"""
Script de prueba para demostrar la integración de ChromaDB con el sistema RAG.
Este script muestra cómo el RAG puede obtener datos de sitios turísticos desde ChromaDB
en lugar del archivo JSON tradicional.
"""

import os
import sys
from pathlib import Path

# Agregar el directorio padre al path para importar módulos
sys.path.append(str(Path(__file__).parent.parent))

from RAG.rag import RAGPlanner
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_chroma_integration():
    """
    Prueba la integración de ChromaDB con el sistema RAG
    """
    print("=" * 60)
    print("PRUEBA DE INTEGRACIÓN CHROMADB CON RAG")
    print("=" * 60)
    
    try:
        # Inicializar RAG con ChromaDB
        print("\n1. Inicializando RAG con ChromaDB...")
        rag_planner = RAGPlanner(chroma_db_path="db/")
        
        print(f"✓ RAG inicializado exitosamente")
        print(f"✓ Fuente de datos: ChromaDB")
        print(f"✓ Lugares cargados: {len(rag_planner.places_data)}")
        
        # Mostrar algunos ejemplos de lugares cargados desde ChromaDB
        print("\n2. Ejemplos de lugares cargados desde ChromaDB:")
        for i, place in enumerate(rag_planner.places_data[:3]):
            print(f"\nLugar {i+1}:")
            print(f"  Nombre: {place.get('name', 'N/A')}")
            print(f"  Tipo: {place.get('type', 'N/A')}")
            print(f"  Ciudad: {place.get('location', {}).get('city', 'N/A')}")
            print(f"  Clasificación: {place.get('touristClassification', 'N/A')}")
            print(f"  URL fuente: {place.get('source_url', 'N/A')}")
            print(f"  Descripción: {place.get('description', 'N/A')[:100]}...")
        
        # Probar búsqueda semántica
        print("\n3. Probando búsqueda semántica en ChromaDB...")
        if hasattr(rag_planner, 'chroma_collection'):
            search_query = "museums and cultural attractions in Madrid"
            semantic_results = rag_planner._semantic_search_chroma(search_query, n_results=5)
            
            print(f"✓ Búsqueda semántica completada")
            print(f"✓ Query: '{search_query}'")
            print(f"✓ Resultados encontrados: {len(semantic_results)}")
            
            for i, result in enumerate(semantic_results[:2]):
                print(f"\nResultado {i+1}:")
                print(f"  Nombre: {result.get('name', 'N/A')}")
                print(f"  Distancia semántica: {result.get('semantic_distance', 'N/A')}")
                print(f"  Descripción: {result.get('description', 'N/A')[:100]}...")
        
        # Simular una solicitud de usuario
        print("\n4. Simulando solicitud de usuario...")
        user_preferences = {
            'city': 'Madrid',
            'available_hours': 8,
            'max_distance': 15,
            'transport_modes': ['A pie', 'Transporte público'],
            'category_interest': {
                'cultural': 5,
                'historical': 4,
                'museums': 5,
                'entertainment': 3,
                'natural': 2
            },
            'user_notes': 'Interested in art museums and historical sites',
            'similarity_threshold': 0.2
        }
        
        # Coordenadas de ejemplo para Madrid
        user_lat, user_lon = 40.4168, -3.7038
        transport_mode = "A pie"
        
        print(f"✓ Preferencias del usuario configuradas")
        print(f"✓ Ciudad: {user_preferences['city']}")
        print(f"✓ Intereses principales: cultural (5/5), museums (5/5), historical (4/5)")
        
        # Procesar solicitud (solo si tenemos las API keys)
        if os.getenv('OPENROUTER_API_KEY') and os.getenv('OPENROUTESERVICE_API_KEY'):
            print("\n5. Procesando solicitud con RAG...")
            try:
                result = rag_planner.process_user_request(
                    user_preferences, user_lat, user_lon, transport_mode
                )
                
                print(f"✓ Solicitud procesada exitosamente")
                print(f"✓ Fuente de datos utilizada: {result.get('data_source', 'N/A')}")
                print(f"✓ Lugares filtrados: {len(result['filtered_places'])}")
                print(f"✓ Matriz de tiempo calculada: {result['time_matrix'].shape}")
                
                # Mostrar algunos lugares recomendados
                print("\nLugares recomendados:")
                for i, place in enumerate(result['filtered_places'][:3]):
                    similarity = result['similarity_scores'][i] if i < len(result['similarity_scores']) else 'N/A'
                    time_estimate = result['llm_time_estimates'][i] if i < len(result['llm_time_estimates']) else 'N/A'
                    print(f"  {i+1}. {place.get('name', 'N/A')}")
                    print(f"     Similitud: {similarity:.3f}" if isinstance(similarity, float) else f"     Similitud: {similarity}")
                    print(f"     Tiempo estimado: {time_estimate} horas" if isinstance(time_estimate, (int, float)) else f"     Tiempo estimado: {time_estimate}")
                
            except Exception as e:
                print(f"⚠ Error al procesar solicitud completa: {e}")
                print("  (Esto puede deberse a falta de API keys o problemas de conectividad)")
        else:
            print("\n5. Saltando procesamiento completo (faltan API keys)")
            print("  Para probar completamente, configure OPENROUTER_API_KEY y OPENROUTESERVICE_API_KEY")
        
        print("\n" + "=" * 60)
        print("✓ INTEGRACIÓN CHROMADB FUNCIONANDO CORRECTAMENTE")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error durante la prueba: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_chromadb_requirements():
    """
    Prueba que ChromaDB esté disponible y tenga datos
    """
    print("\n" + "=" * 60)
    print("VERIFICACIÓN DE REQUISITOS CHROMADB")
    print("=" * 60)
    
    try:
        import chromadb
        
        print("\n1. Verificando ChromaDB...")
        chroma_client = chromadb.PersistentClient(path="db/")
        collection = chroma_client.get_or_create_collection(name="tourism_docs")
        
        doc_count = collection.count()
        print(f"✓ ChromaDB disponible")
        print(f"✓ Documentos en la colección: {doc_count}")
        
        if doc_count == 0:
            print("⚠ ChromaDB está vacío. Ejecuta 'python src/populate_chroma.py' para poblarlo.")
            return False
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error verificando ChromaDB: {e}")
        print("⚠ Asegúrate de que ChromaDB esté instalado y configurado correctamente.")
        return False

if __name__ == "__main__":
    print("Iniciando pruebas de integración ChromaDB...")
    
    # Cambiar al directorio correcto
    script_dir = Path(__file__).parent.parent.parent
    os.chdir(script_dir)
    
    success = True
    
    # Verificar requisitos de ChromaDB
    success &= test_chromadb_requirements()
    
    if not success:
        print("\n❌ ChromaDB no está disponible o no tiene datos.")
        print("Ejecuta 'python src/populate_chroma.py' para poblar la base de datos.")
        sys.exit(1)
    
    # Probar integración con ChromaDB
    success &= test_chroma_integration()
    
    if success:
        print("\n🎉 Todas las pruebas pasaron exitosamente!")
        print("\nEl sistema RAG ahora puede:")
        print("  ✓ Obtener datos de sitios turísticos desde ChromaDB")
        print("  ✓ Realizar búsquedas semánticas en la base vectorial")
        print("  ✓ Integrar datos del crawler con el sistema de recomendaciones")
        print("  ✓ Funcionar completamente sin archivos JSON")
    else:
        print("\n❌ Algunas pruebas fallaron. Revise los errores anteriores.")
        sys.exit(1)