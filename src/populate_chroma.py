#!/usr/bin/env python3
"""
Script para poblar ChromaDB con datos de sitios turísticos usando el crawler.
Este script ejecuta el crawler para extraer información de sitios web de turismo
y almacenarla en ChromaDB para su uso posterior por el sistema RAG.
"""

import os
import sys
from pathlib import Path
import logging

# Agregar el directorio actual al path
sys.path.append(str(Path(__file__).parent))

from crawler.crawler import TourismCrawler, ChromaDBManager, NLTKManager
import chromadb

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_chromadb():
    """
    Configura ChromaDB y retorna el manager
    """
    try:
        # Crear directorio db si no existe
        db_path = Path("db")
        db_path.mkdir(exist_ok=True)
        
        # Inicializar ChromaDB
        chroma_client = chromadb.PersistentClient(path="db/")
        collection = chroma_client.get_or_create_collection(name="tourism_docs")
        chroma_manager = ChromaDBManager(collection)
        
        logger.info(f"ChromaDB configurado. Documentos existentes: {collection.count()}")
        return chroma_manager
        
    except Exception as e:
        logger.error(f"Error configurando ChromaDB: {e}")
        raise

def run_crawler(max_documents=20):
    """
    Ejecuta el crawler para poblar ChromaDB con sitios turísticos
    
    Args:
        max_documents: Número máximo de documentos a extraer
    """
    print("=" * 60)
    print("POBLANDO CHROMADB CON SITIOS TURÍSTICOS")
    print("=" * 60)
    
    try:
        # Configurar ChromaDB
        print("\n1. Configurando ChromaDB...")
        chroma_manager = setup_chromadb()
        
        # Configurar NLTK
        print("\n2. Configurando NLTK...")
        nltk_manager = NLTKManager(language='english')
        
        # URLs de sitios de turismo para crawlear
        start_urls = [
            "https://www.spain.info/",
            "https://www.turismo.madrid.es/",
            "https://www.barcelonaturisme.com/",
            "https://www.andalucia.org/",
            "https://www.turismocastillayleon.com/"
        ]
        
        print(f"\n3. Iniciando crawler con {len(start_urls)} URLs...")
        print("URLs objetivo:")
        for url in start_urls:
            print(f"  - {url}")
        
        # Crear y ejecutar crawler
        crawler = TourismCrawler(
            start_urls=start_urls,
            chroma_manager=chroma_manager,
            nltk_manager=nltk_manager,
            max_documents=max_documents,
            user_agent="TourismCrawler/1.0 (Tourist Guide App)"
        )
        
        print(f"\n4. Ejecutando crawler (máximo {max_documents} documentos)...")
        crawler.crawl()
        
        # Verificar resultados
        print("\n5. Verificando resultados...")
        chroma_manager.test_collection()
        
        print("\n" + "=" * 60)
        print("✓ CHROMADB POBLADO EXITOSAMENTE")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"Error durante el crawling: {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_chromadb_content():
    """
    Verifica el contenido de ChromaDB después del crawling
    """
    try:
        print("\n" + "=" * 60)
        print("VERIFICACIÓN DEL CONTENIDO DE CHROMADB")
        print("=" * 60)
        
        # Conectar a ChromaDB
        chroma_client = chromadb.PersistentClient(path="db/")
        collection = chroma_client.get_or_create_collection(name="tourism_docs")
        
        # Obtener estadísticas
        total_docs = collection.count()
        print(f"\nTotal de documentos en ChromaDB: {total_docs}")
        
        if total_docs > 0:
            # Obtener algunos documentos de ejemplo
            sample_docs = collection.get(limit=3, include=["metadatas", "documents"])
            
            print(f"\nEjemplos de documentos almacenados:")
            for i, (doc, metadata) in enumerate(zip(sample_docs["documents"], sample_docs["metadatas"])):
                print(f"\nDocumento {i+1}:")
                print(f"  URL: {metadata.get('source_url', 'N/A')}")
                print(f"  Entidades: {metadata.get('named_entities', 'N/A')}")
                print(f"  Tokens: {metadata.get('lemmatized_token_count', 'N/A')}")
                print(f"  Contenido (preview): {doc[:200]}...")
            
            # Probar búsqueda semántica
            print(f"\nProbando búsqueda semántica...")
            search_results = collection.query(
                query_texts=["museums and cultural attractions"],
                n_results=3,
                include=["metadatas", "documents", "distances"]
            )
            
            if search_results["documents"] and search_results["documents"][0]:
                print(f"Resultados de búsqueda semántica:")
                for i, (doc, metadata, distance) in enumerate(zip(
                    search_results["documents"][0],
                    search_results["metadatas"][0], 
                    search_results["distances"][0]
                )):
                    print(f"\nResultado {i+1}:")
                    print(f"  Distancia: {distance:.4f}")
                    print(f"  URL: {metadata.get('source_url', 'N/A')}")
                    print(f"  Contenido: {doc[:150]}...")
            else:
                print("No se encontraron resultados en la búsqueda semántica")
        
        print("\n✓ Verificación completada")
        return True
        
    except Exception as e:
        logger.error(f"Error verificando ChromaDB: {e}")
        return False

def main():
    """
    Función principal
    """
    print("Iniciando población de ChromaDB con sitios turísticos...")
    
    # Cambiar al directorio correcto
    script_dir = Path(__file__).parent.parent
    os.chdir(script_dir)
    
    # Verificar si ChromaDB ya tiene contenido
    try:
        chroma_client = chromadb.PersistentClient(path="db/")
        collection = chroma_client.get_or_create_collection(name="tourism_docs")
        existing_docs = collection.count()
        
        if existing_docs > 0:
            print(f"\nChromaDB ya contiene {existing_docs} documentos.")
            response = input("¿Desea agregar más documentos? (y/n): ").lower().strip()
            if response != 'y':
                print("Operación cancelada.")
                verify_chromadb_content()
                return
    except Exception as e:
        logger.warning(f"No se pudo verificar el estado de ChromaDB: {e}")
    
    # Solicitar número de documentos
    try:
        max_docs = input("\n¿Cuántos documentos desea extraer? (default: 20): ").strip()
        max_docs = int(max_docs) if max_docs else 20
    except ValueError:
        max_docs = 20
        print("Usando valor por defecto: 20 documentos")
    
    # Ejecutar crawler
    success = run_crawler(max_documents=max_docs)
    
    if success:
        # Verificar contenido
        verify_chromadb_content()
        
        print("\n🎉 Proceso completado exitosamente!")
        print("\nAhora puede:")
        print("  1. Ejecutar el sistema RAG con ChromaDB habilitado")
        print("  2. Usar el script de prueba: python src/RAG/test_chroma_integration.py")
        print("  3. Verificar la base de datos: python src/ver_db.py")
    else:
        print("\n❌ El proceso falló. Revise los errores anteriores.")
        sys.exit(1)

if __name__ == "__main__":
    main()