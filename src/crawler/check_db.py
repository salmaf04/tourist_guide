#!/usr/bin/env python3
"""
Script para verificar y explorar la base de datos ChromaDB después de ejecutar el crawler completo.
"""

import sys
import os
from pathlib import Path
from datetime import datetime

# Agregar el directorio src al path
sys.path.append(str(Path(__file__).parent.parent))

import chromadb
import json


class DatabaseChecker:
    """Clase para verificar y explorar la base de datos ChromaDB"""
    
    def __init__(self, db_path="../db"):
        self.db_path = db_path
        self.chroma_client = None
        self.collection = None
        
    def connect(self):
        """Conecta a la base de datos ChromaDB"""
        try:
            self.chroma_client = chromadb.PersistentClient(path=self.db_path)
            self.collection = self.chroma_client.get_or_create_collection(name="tourist_places")
            print(f"✅ Conectado a ChromaDB: {self.db_path}")
            return True
        except Exception as e:
            print(f"❌ Error conectando a ChromaDB: {e}")
            return False
    
    def get_complete_stats(self):
        """Obtiene estadísticas completas de la base de datos"""
        try:
            count = self.collection.count()
            print(f"\n📊 ESTADÍSTICAS COMPLETAS DE LA BASE DE DATOS:")
            print(f"   Total de lugares turísticos: {count}")
            
            if count == 0:
                print("   ⚠️  La base de datos está vacía")
                print("   💡 Ejecuta 'python main.py' para llenar la base de datos")
                return
            
            # Obtener todos los metadatos para estadísticas detalladas
            all_data = self.collection.get()
            metadatas = all_data['metadatas']
            documents = all_data['documents']
            
            # Estadísticas por ciudad
            cities = {}
            categories = {}
            description_lengths = []
            
            for metadata, doc in zip(metadatas, documents):
                city = metadata.get('city', 'Unknown')
                category = metadata.get('category', 'Unknown')
                description = metadata.get('description', '')
                
                cities[city] = cities.get(city, 0) + 1
                categories[category] = categories.get(category, 0) + 1
                description_lengths.append(len(description))
            
            print(f"\n🏙️  DISTRIBUCIÓN POR CIUDADES ({len(cities)} ciudades):")
            for city, count in sorted(cities.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / len(metadatas)) * 100
                print(f"   {city}: {count} lugares ({percentage:.1f}%)")
            
            print(f"\n🏷️  DISTRIBUCIÓN POR CATEGORÍAS ({len(categories)} categorías):")
            for category, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / len(metadatas)) * 100
                print(f"   {category}: {count} lugares ({percentage:.1f}%)")
            
            # Estadísticas de calidad de datos
            avg_desc_length = sum(description_lengths) / len(description_lengths) if description_lengths else 0
            print(f"\n📝 CALIDAD DE DATOS:")
            print(f"   Longitud promedio de descripción: {avg_desc_length:.0f} caracteres")
            print(f"   Lugares con coordenadas: {sum(1 for m in metadatas if m.get('coordinates') and m.get('coordinates') != 'None')}")
            print(f"   Lugares sin coordenadas: {sum(1 for m in metadatas if not m.get('coordinates') or m.get('coordinates') == 'None')}")
                
        except Exception as e:
            print(f"❌ Error obteniendo estadísticas: {e}")
    
    def show_detailed_sample(self, limit=5):
        """Muestra una muestra detallada de los lugares"""
        try:
            results = self.collection.get(limit=limit)
            
            if not results['documents']:
                print("⚠️  No hay datos para mostrar")
                return
            
            print(f"\n📍 MUESTRA DETALLADA DE {len(results['documents'])} LUGARES:")
            print("=" * 100)
            
            for i, (doc, metadata) in enumerate(zip(results['documents'], results['metadatas'])):
                print(f"\n🏛️  LUGAR {i+1}:")
                print(f"   🏷️  Nombre: {metadata.get('name', 'Sin nombre')}")
                print(f"   🏙️  Ciudad: {metadata.get('city', 'N/A')}")
                print(f"   🎯 Categoría: {metadata.get('category', 'N/A')}")
                print(f"   📍 Coordenadas: {metadata.get('coordinates', 'N/A')}")
                print(f"   📝 Descripción:")
                description = metadata.get('description', 'N/A')
                if len(description) > 200:
                    print(f"      {description[:200]}...")
                else:
                    print(f"      {description}")
                print(f"   📄 Texto completo: {len(doc)} caracteres")
                print(f"   🔗 Texto embedding (primeros 100 chars): {doc[:100]}...")
                print("-" * 80)
            
        except Exception as e:
            print(f"❌ Error mostrando muestra: {e}")
    
    def perform_comprehensive_search(self, query, limit=5):
        """Realiza búsqueda semántica completa"""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=limit
            )
            
            if not results['documents'][0]:
                print(f"❌ No se encontraron resultados para: '{query}'")
                return
            
            print(f"\n🔍 BÚSQUEDA COMPLETA PARA: '{query}'")
            print("=" * 80)
            
            for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0], 
                results['metadatas'][0],
                results['distances'][0]
            )):
                similarity = 1 - distance
                print(f"\n🎯 RESULTADO {i+1} (Similitud: {similarity:.3f}):")
                print(f"   🏷️  Nombre: {metadata.get('name', 'Sin nombre')}")
                print(f"   🏙️  Ciudad: {metadata.get('city', 'N/A')}")
                print(f"   🎯 Categoría: {metadata.get('category', 'N/A')}")
                print(f"   📍 Coordenadas: {metadata.get('coordinates', 'N/A')}")
                print(f"   📝 Descripción: {metadata.get('description', 'N/A')[:150]}...")
                print(f"   🔗 Relevancia del texto: {doc[:100]}...")
                print("-" * 60)
            
        except Exception as e:
            print(f"❌ Error en búsqueda: {e}")
    
    def run_comprehensive_tests(self):
        """Ejecuta pruebas completas del sistema"""
        print(f"\n🧪 EJECUTANDO PRUEBAS COMPLETAS DEL SISTEMA:")
        print("=" * 60)
        
        # Pruebas de búsqueda variadas
        test_queries = [
            "museos de arte y cultura",
            "parques y jardines naturales", 
            "lugares históricos y monumentos",
            "atracciones turísticas principales",
            "iglesias y sitios religiosos",
            "arquitectura y edificios importantes"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n🔎 PRUEBA {i}: Buscando '{query}'")
            
            results = self.collection.query(
                query_texts=[query],
                n_results=3
            )
            
            if results['documents'][0]:
                print(f"   ✅ Encontrados {len(results['documents'][0])} resultados:")
                for j, (metadata, distance) in enumerate(zip(results['metadatas'][0], results['distances'][0])):
                    similarity = 1 - distance
                    print(f"      {j+1}. {metadata.get('name', 'N/A')} "
                          f"({metadata.get('city', 'N/A')}) - "
                          f"Similitud: {similarity:.3f}")
            else:
                print(f"   ❌ No se encontraron resultados")
        
        print(f"\n✅ Pruebas completas finalizadas")
    
    def export_complete_analysis(self):
        """Exporta un análisis completo de los datos"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"database_analysis_{timestamp}.json"
            
            all_data = self.collection.get()
            
            # Crear análisis completo
            analysis = {
                'timestamp': timestamp,
                'total_places': len(all_data['documents']),
                'statistics': {},
                'sample_data': [],
                'quality_metrics': {}
            }
            
            # Estadísticas por ciudad y categoría
            cities = {}
            categories = {}
            
            for metadata in all_data['metadatas']:
                city = metadata.get('city', 'Unknown')
                category = metadata.get('category', 'Unknown')
                
                cities[city] = cities.get(city, 0) + 1
                categories[category] = categories.get(category, 0) + 1
            
            analysis['statistics']['cities'] = cities
            analysis['statistics']['categories'] = categories
            
            # Muestra de datos (primeros 10)
            for i in range(min(10, len(all_data['documents']))):
                sample_place = {
                    'name': all_data['metadatas'][i].get('name', ''),
                    'city': all_data['metadatas'][i].get('city', ''),
                    'category': all_data['metadatas'][i].get('category', ''),
                    'description': all_data['metadatas'][i].get('description', ''),
                    'coordinates': all_data['metadatas'][i].get('coordinates', ''),
                    'text_length': len(all_data['documents'][i])
                }
                analysis['sample_data'].append(sample_place)
            
            # Métricas de calidad
            description_lengths = [len(m.get('description', '')) for m in all_data['metadatas']]
            analysis['quality_metrics'] = {
                'avg_description_length': sum(description_lengths) / len(description_lengths) if description_lengths else 0,
                'places_with_coordinates': sum(1 for m in all_data['metadatas'] if m.get('coordinates') and m.get('coordinates') != 'None'),
                'places_without_coordinates': sum(1 for m in all_data['metadatas'] if not m.get('coordinates') or m.get('coordinates') == 'None')
            }
            
            # Guardar análisis
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, ensure_ascii=False, indent=2)
            
            print(f"📁 Análisis completo exportado a: {filename}")
            
        except Exception as e:
            print(f"❌ Error exportando análisis: {e}")


def main():
    """Función principal"""
    print("🔍 VERIFICADOR COMPLETO DE BASE DE DATOS TURÍSTICA")
    print("=" * 60)
    
    # Crear checker
    checker = DatabaseChecker()
    
    # Conectar a la base de datos
    if not checker.connect():
        return
    
    # Procesar argumentos de línea de comandos
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command in ['stats', 'estadisticas', 'complete']:
            checker.get_complete_stats()
            
        elif command in ['sample', 'muestra']:
            limit = 5
            if len(sys.argv) > 2:
                try:
                    limit = int(sys.argv[2])
                except ValueError:
                    pass
            checker.show_detailed_sample(limit)
            
        elif command in ['search', 'buscar']:
            if len(sys.argv) > 2:
                query = ' '.join(sys.argv[2:])
                checker.perform_comprehensive_search(query)
            else:
                print("❌ Debes proporcionar una consulta de búsqueda")
                
        elif command in ['test', 'pruebas']:
            checker.run_comprehensive_tests()
            
        elif command in ['export', 'analysis']:
            checker.export_complete_analysis()
            
        elif command in ['all', 'todo']:
            # Ejecutar verificación completa
            checker.get_complete_stats()
            checker.show_detailed_sample(3)
            checker.run_comprehensive_tests()
            checker.export_complete_analysis()
            
        elif command in ['help', 'ayuda', '-h', '--help']:
            show_help()
            
        else:
            print(f"❌ Comando desconocido: {command}")
            show_help()
    else:
        # Verificación por defecto
        checker.get_complete_stats()
        checker.show_detailed_sample(3)


def show_help():
    """Muestra la ayuda completa del script"""
    print("\n📖 USO COMPLETO:")
    print("   python check_db.py                      # Estadísticas y muestra básica")
    print("   python check_db.py stats                # Estadísticas completas")
    print("   python check_db.py sample [N]           # Muestra detallada de N lugares")
    print("   python check_db.py search <consulta>    # Búsqueda semántica completa")
    print("   python check_db.py test                 # Ejecutar todas las pruebas")
    print("   python check_db.py export               # Exportar análisis completo")
    print("   python check_db.py all                  # Ejecutar verificación completa")
    print("")
    print("🔍 EJEMPLOS DE USO:")
    print("   python check_db.py search 'museos Madrid'")
    print("   python check_db.py search 'parques Barcelona'")
    print("   python check_db.py sample 10")
    print("   python check_db.py all")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹️  Operación interrumpida por el usuario")
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        sys.exit(1)