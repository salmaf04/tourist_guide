#!/usr/bin/env python3
"""
Script para detectar y analizar información duplicada en la base de datos ChromaDB.
Identifica lugares turísticos repetidos usando diferentes criterios de comparación.
"""

import os
import sys
from pathlib import Path
from collections import defaultdict, Counter
import chromadb
from difflib import SequenceMatcher
import re

# Agregar el directorio src al path para importaciones
current_dir = Path(__file__).parent
src_dir = current_dir.parent
sys.path.insert(0, str(src_dir))

# Evitar importación circular - usar ChromaDB directamente


class DuplicateDetector:
    """Detector de duplicados en la base de datos ChromaDB."""
    
    def __init__(self, db_path="./db/", output_file="duplicate_analysis.txt"):
        """Inicializa el detector con la ruta de la base de datos."""
        self.db_path = db_path
        self.collection = None
        self.output_file = output_file
        self.output_lines = []
        self._setup_database()
    
    def _setup_database(self):
        """Configura la conexión a ChromaDB."""
        try:
            # Verificar si existe la base de datos
            if not os.path.exists(self.db_path):
                print(f"❌ No se encontró la base de datos en: {self.db_path}")
                return False
            
            # Conectar a ChromaDB
            chroma_client = chromadb.PersistentClient(path=self.db_path)
            
            # Obtener la colección
            try:
                self.collection = chroma_client.get_collection(name="tourist_places")
                print(f"✅ Conectado a ChromaDB. Total de documentos: {self.collection.count()}")
                return True
            except Exception as e:
                print(f"❌ No se encontró la colección 'tourist_places': {e}")
                return False
                
        except Exception as e:
            print(f"❌ Error conectando a ChromaDB: {e}")
            return False
    
    def normalize_text(self, text):
        """Normaliza texto para comparación (minúsculas, sin acentos, etc.)."""
        if not text:
            return ""
        
        # Convertir a minúsculas
        text = text.lower().strip()
        
        # Remover acentos
        replacements = {
            'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u', 'ü': 'u',
            'ñ': 'n', 'ç': 'c'
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Remover caracteres especiales y espacios extra
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def calculate_similarity(self, text1, text2):
        """Calcula la similitud entre dos textos (0-1)."""
        if not text1 or not text2:
            return 0.0
        
        normalized1 = self.normalize_text(text1)
        normalized2 = self.normalize_text(text2)
        
        return SequenceMatcher(None, normalized1, normalized2).ratio()
    
    def find_exact_duplicates(self):
        """Encuentra duplicados exactos por nombre y ciudad."""
        print("\n🔍 Buscando duplicados exactos (nombre + ciudad)...")
        
        if not self.collection:
            print("❌ No hay conexión a la base de datos")
            return {}
        
        # Obtener todos los documentos
        all_docs = self.collection.get(include=["metadatas", "documents"])
        
        # Agrupar por nombre normalizado + ciudad normalizada
        groups = defaultdict(list)
        
        for i, metadata in enumerate(all_docs["metadatas"]):
            name = metadata.get("name", "")
            city = metadata.get("city", "")
            
            # Crear clave normalizada
            key = f"{self.normalize_text(name)}|{self.normalize_text(city)}"
            
            groups[key].append({
                'id': all_docs["ids"][i],
                'name': name,
                'city': city,
                'category': metadata.get("category", ""),
                'description': metadata.get("description", "")[:100] + "...",
                'index': i
            })
        
        # Filtrar solo grupos con duplicados
        duplicates = {key: items for key, items in groups.items() if len(items) > 1}
        
        if duplicates:
            print(f"❌ Se encontraron {len(duplicates)} grupos de duplicados exactos:")
            
            total_duplicates = 0
            for key, items in duplicates.items():
                total_duplicates += len(items)
                print(f"\n📍 Grupo: {len(items)} duplicados")
                for item in items:
                    print(f"   • ID: {item['id'][:8]}... | {item['name']} ({item['city']}) | {item['category']}")
                    print(f"     Descripción: {item['description']}")
        else:
            print("✅ No se encontraron duplicados exactos")
        
        return duplicates
    
    def find_similar_duplicates(self, similarity_threshold=0.85):
        """Encuentra duplicados similares usando comparación de texto."""
        print(f"\n🔍 Buscando duplicados similares (umbral: {similarity_threshold})...")
        
        if not self.collection:
            print("❌ No hay conexión a la base de datos")
            return {}
        
        # Obtener todos los documentos
        all_docs = self.collection.get(include=["metadatas", "documents"])
        
        similar_groups = []
        processed_indices = set()
        
        for i, metadata_i in enumerate(all_docs["metadatas"]):
            if i in processed_indices:
                continue
            
            name_i = metadata_i.get("name", "")
            city_i = metadata_i.get("city", "")
            
            # Buscar similares
            similar_items = [{
                'id': all_docs["ids"][i],
                'name': name_i,
                'city': city_i,
                'category': metadata_i.get("category", ""),
                'description': metadata_i.get("description", "")[:100] + "...",
                'index': i
            }]
            
            for j, metadata_j in enumerate(all_docs["metadatas"]):
                if i >= j or j in processed_indices:
                    continue
                
                name_j = metadata_j.get("name", "")
                city_j = metadata_j.get("city", "")
                
                # Calcular similitud del nombre
                name_similarity = self.calculate_similarity(name_i, name_j)
                
                # Si las ciudades son iguales y los nombres son similares
                if (self.normalize_text(city_i) == self.normalize_text(city_j) and 
                    name_similarity >= similarity_threshold):
                    
                    similar_items.append({
                        'id': all_docs["ids"][j],
                        'name': name_j,
                        'city': city_j,
                        'category': metadata_j.get("category", ""),
                        'description': metadata_j.get("description", "")[:100] + "...",
                        'index': j,
                        'similarity': name_similarity
                    })
                    processed_indices.add(j)
            
            if len(similar_items) > 1:
                similar_groups.append(similar_items)
                processed_indices.add(i)
        
        if similar_groups:
            print(f"❌ Se encontraron {len(similar_groups)} grupos de duplicados similares:")
            
            for group_idx, items in enumerate(similar_groups, 1):
                print(f"\n📍 Grupo {group_idx}: {len(items)} elementos similares")
                for item in items:
                    similarity_text = f" (similitud: {item['similarity']:.2f})" if 'similarity' in item else ""
                    print(f"   • ID: {item['id'][:8]}... | {item['name']} ({item['city']}) | {item['category']}{similarity_text}")
                    print(f"     Descripción: {item['description']}")
        else:
            print("✅ No se encontraron duplicados similares")
        
        return similar_groups
    
    def analyze_city_distribution(self):
        """Analiza la distribución de lugares por ciudad."""
        print("\n📊 Analizando distribución por ciudades...")
        
        if not self.collection:
            print("❌ No hay conexión a la base de datos")
            return
        
        # Obtener todos los documentos
        all_docs = self.collection.get(include=["metadatas"])
        
        # Contar por ciudad
        city_counts = Counter()
        category_counts = Counter()
        
        for metadata in all_docs["metadatas"]:
            city = metadata.get("city", "Unknown")
            category = metadata.get("category", "Unknown")
            
            city_counts[city] += 1
            category_counts[category] += 1
        
        print(f"\n🏙️ Distribución por ciudades ({len(city_counts)} ciudades):")
        for city, count in city_counts.most_common():
            print(f"   • {city}: {count} lugares")
        
        print(f"\n🏛️ Distribución por categorías ({len(category_counts)} categorías):")
        for category, count in category_counts.most_common():
            print(f"   • {category}: {count} lugares")
    
    def find_potential_issues(self):
        """Encuentra posibles problemas en los datos."""
        print("\n🔍 Buscando posibles problemas en los datos...")
        
        if not self.collection:
            print("❌ No hay conexión a la base de datos")
            return
        
        # Obtener todos los documentos
        all_docs = self.collection.get(include=["metadatas", "documents"])
        
        issues = {
            'empty_names': [],
            'empty_cities': [],
            'empty_descriptions': [],
            'unknown_cities': [],
            'short_descriptions': [],
            'long_names': []
        }
        
        for i, metadata in enumerate(all_docs["metadatas"]):
            doc_id = all_docs["ids"][i]
            name = metadata.get("name", "")
            city = metadata.get("city", "")
            description = metadata.get("description", "")
            
            # Verificar problemas
            if not name or name.strip() == "":
                issues['empty_names'].append(doc_id)
            
            if not city or city.strip() == "":
                issues['empty_cities'].append(doc_id)
            
            if not description or description.strip() == "":
                issues['empty_descriptions'].append(doc_id)
            
            if city.lower() in ['unknown', 'n/a', 'sin ciudad']:
                issues['unknown_cities'].append((doc_id, name, city))
            
            if description and len(description.strip()) < 20:
                issues['short_descriptions'].append((doc_id, name, len(description)))
            
            if name and len(name) > 100:
                issues['long_names'].append((doc_id, name[:50] + "..."))
        
        # Reportar problemas
        total_issues = sum(len(issue_list) for issue_list in issues.values())
        
        if total_issues > 0:
            print(f"❌ Se encontraron {total_issues} posibles problemas:")
            
            if issues['empty_names']:
                print(f"\n   • Nombres vacíos: {len(issues['empty_names'])} documentos")
            
            if issues['empty_cities']:
                print(f"   • Ciudades vacías: {len(issues['empty_cities'])} documentos")
            
            if issues['empty_descriptions']:
                print(f"   • Descripciones vacías: {len(issues['empty_descriptions'])} documentos")
            
            if issues['unknown_cities']:
                print(f"   • Ciudades desconocidas: {len(issues['unknown_cities'])} documentos")
                for doc_id, name, city in issues['unknown_cities'][:5]:  # Mostrar solo los primeros 5
                    print(f"     - {name} ({city}) [ID: {doc_id[:8]}...]")
            
            if issues['short_descriptions']:
                print(f"   • Descripciones muy cortas: {len(issues['short_descriptions'])} documentos")
            
            if issues['long_names']:
                print(f"   • Nombres muy largos: {len(issues['long_names'])} documentos")
        else:
            print("✅ No se encontraron problemas evidentes en los datos")
    
    def generate_cleanup_suggestions(self, exact_duplicates, similar_duplicates):
        """Genera sugerencias para limpiar duplicados."""
        print("\n💡 Sugerencias de limpieza:")
        
        total_to_remove = 0
        
        if exact_duplicates:
            for key, items in exact_duplicates.items():
                # Sugerir mantener el primero y eliminar el resto
                to_remove = len(items) - 1
                total_to_remove += to_remove
                print(f"\n📍 Grupo exacto: {items[0]['name']} ({items[0]['city']})")
                print(f"   • Mantener: ID {items[0]['id'][:8]}...")
                for item in items[1:]:
                    print(f"   • Eliminar: ID {item['id'][:8]}...")
        
        if similar_duplicates:
            for group in similar_duplicates:
                # Sugerir mantener el que tenga mejor descripción
                best_item = max(group, key=lambda x: len(x.get('description', '')))
                to_remove = len(group) - 1
                total_to_remove += to_remove
                
                print(f"\n📍 Grupo similar: {group[0]['name']} ({group[0]['city']})")
                print(f"   • Mantener: ID {best_item['id'][:8]}... (mejor descripción)")
                for item in group:
                    if item['id'] != best_item['id']:
                        similarity_text = f" (similitud: {item.get('similarity', 0):.2f})" if 'similarity' in item else ""
                        print(f"   • Eliminar: ID {item['id'][:8]}...{similarity_text}")
        
        if total_to_remove > 0:
            print(f"\n📊 Resumen: Se sugiere eliminar {total_to_remove} documentos duplicados")
            print("⚠️  IMPORTANTE: Revisa manualmente antes de eliminar cualquier documento")
        else:
            print("\n✅ No se requiere limpieza de duplicados")
    
    def run_full_analysis(self):
        """Ejecuta el análisis completo de duplicados."""
        print("🚀 Iniciando análisis completo de duplicados en ChromaDB")
        print("=" * 60)
        
        if not self.collection:
            print("❌ No se pudo conectar a la base de datos")
            return
        
        # Análisis de distribución
        self.analyze_city_distribution()
        
        # Buscar duplicados exactos
        exact_duplicates = self.find_exact_duplicates()
        
        # Buscar duplicados similares
        similar_duplicates = self.find_similar_duplicates()
        
        # Buscar problemas generales
        self.find_potential_issues()
        
        # Generar sugerencias
        self.generate_cleanup_suggestions(exact_duplicates, similar_duplicates)
        
        print("\n" + "=" * 60)
        print("✅ Análisis completado")


def main():
    """Función principal del script."""
    print("🔍 Detector de Duplicados - ChromaDB Tourist Places")
    print("=" * 50)
    
    # Crear detector
    detector = DuplicateDetector()
    
    # Ejecutar análisis completo
    detector.run_full_analysis()


if __name__ == "__main__":
    main()