#!/usr/bin/env python3
"""
Script simplificado para encontrar duplicados exactos en ChromaDB y guardarlos en un archivo de texto.
"""

import os
import sys
from pathlib import Path
from collections import defaultdict
import chromadb
import re
from datetime import datetime

# Agregar el directorio src al path para importaciones
current_dir = Path(__file__).parent
src_dir = current_dir.parent
sys.path.insert(0, str(src_dir))


class ExactDuplicateFinder:
    """Detector de duplicados exactos en ChromaDB."""
    
    def __init__(self, db_path="./db/", output_file="duplicados_exactos.txt"):
        """Inicializa el detector."""
        self.db_path = db_path
        self.collection = None
        self.output_file = output_file
        self.output_lines = []
        
    def log(self, message):
        """Añade mensaje al log de salida."""
        self.output_lines.append(message)
        print(message)  # Tambi��n mostrar en consola
    
    def normalize_text(self, text):
        """Normaliza texto para comparación."""
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
    
    def connect_to_database(self):
        """Conecta a la base de datos ChromaDB."""
        try:
            if not os.path.exists(self.db_path):
                self.log(f"❌ No se encontró la base de datos en: {self.db_path}")
                return False
            
            # Conectar a ChromaDB
            chroma_client = chromadb.PersistentClient(path=self.db_path)
            
            # Obtener la colección
            try:
                self.collection = chroma_client.get_collection(name="tourist_places")
                self.log(f"✅ Conectado a ChromaDB. Total de documentos: {self.collection.count()}")
                return True
            except Exception as e:
                self.log(f"❌ No se encontró la colección 'tourist_places': {e}")
                return False
                
        except Exception as e:
            self.log(f"❌ Error conectando a ChromaDB: {e}")
            return False
    
    def find_exact_duplicates(self):
        """Encuentra duplicados exactos por nombre y ciudad."""
        self.log("\n🔍 BUSCANDO DUPLICADOS EXACTOS (nombre + ciudad)...")
        self.log("=" * 60)
        
        if not self.collection:
            self.log("❌ No hay conexión a la base de datos")
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
                'description': metadata.get("description", ""),
                'index': i
            })
        
        # Filtrar solo grupos con duplicados
        duplicates = {key: items for key, items in groups.items() if len(items) > 1}
        
        if duplicates:
            total_duplicates = sum(len(items) for items in duplicates.values())
            total_to_remove = sum(len(items) - 1 for items in duplicates.values())
            
            self.log(f"❌ DUPLICADOS EXACTOS ENCONTRADOS:")
            self.log(f"   • Grupos de duplicados: {len(duplicates)}")
            self.log(f"   • Total de documentos duplicados: {total_duplicates}")
            self.log(f"   • Documentos que se pueden eliminar: {total_to_remove}")
            self.log("")
            
            for group_num, (key, items) in enumerate(duplicates.items(), 1):
                self.log(f"📍 GRUPO {group_num}: {len(items)} duplicados")
                self.log(f"   Lugar: {items[0]['name']}")
                self.log(f"   Ciudad: {items[0]['city']}")
                self.log("")
                
                # Mostrar todos los duplicados del grupo
                for item_num, item in enumerate(items, 1):
                    self.log(f"   {item_num}. ID: {item['id']}")
                    self.log(f"      Nombre: {item['name']}")
                    self.log(f"      Ciudad: {item['city']}")
                    self.log(f"      Categoría: {item['category']}")
                    
                    # Mostrar descripción truncada
                    description = item['description']
                    if len(description) > 100:
                        description = description[:100] + "..."
                    self.log(f"      Descripción: {description}")
                    self.log("")
                
                # Sugerencia de limpieza
                self.log(f"   💡 SUGERENCIA: Mantener el ID {items[0]['id'][:8]}... y eliminar los otros {len(items)-1}")
                self.log(f"   🗑️  IDs a eliminar: {', '.join([item['id'][:8] + '...' for item in items[1:]])}")
                self.log("")
                self.log("-" * 50)
                self.log("")
        else:
            self.log("✅ No se encontraron duplicados exactos")
        
        return duplicates
    
    def save_to_file(self):
        """Guarda los resultados en un archivo de texto."""
        try:
            with open(self.output_file, 'w', encoding='utf-8') as f:
                # Escribir encabezado
                f.write("ANÁLISIS DE DUPLICADOS EXACTOS - ChromaDB Tourist Places\n")
                f.write("=" * 60 + "\n")
                f.write(f"Fecha y hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Base de datos: {self.db_path}\n")
                f.write("\n")
                
                # Escribir todas las líneas del log
                for line in self.output_lines:
                    f.write(line + "\n")
            
            print(f"\n💾 Resultados guardados en: {self.output_file}")
            
        except Exception as e:
            print(f"❌ Error guardando archivo: {e}")
    
    def run_analysis(self):
        """Ejecuta el análisis completo."""
        self.log("🔍 DETECTOR DE DUPLICADOS EXACTOS")
        self.log("=" * 40)
        self.log(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log("")
        
        # Conectar a la base de datos
        if not self.connect_to_database():
            self.save_to_file()
            return
        
        # Buscar duplicados exactos
        duplicates = self.find_exact_duplicates()
        
        # Resumen final
        if duplicates:
            total_groups = len(duplicates)
            total_duplicates = sum(len(items) - 1 for items in duplicates.values())
            
            self.log("📊 RESUMEN FINAL:")
            self.log(f"   • Total de grupos con duplicados: {total_groups}")
            self.log(f"   • Total de documentos que se pueden eliminar: {total_duplicates}")
            self.log("")
            self.log("⚠️  IMPORTANTE:")
            self.log("   • Revisa manualmente cada grupo antes de eliminar")
            self.log("   • Usa el script remove_duplicates.py para eliminar de forma segura")
            self.log("   • Siempre haz un backup antes de eliminar datos")
        else:
            self.log("✅ BASE DE DATOS LIMPIA - No hay duplicados exactos")
        
        # Guardar resultados
        self.save_to_file()


def main():
    """Función principal del script."""
    print("🔍 Detector de Duplicados Exactos - ChromaDB")
    print("Guardando resultados en archivo de texto...")
    print("-" * 50)
    
    # Crear detector y ejecutar análisis
    detector = ExactDuplicateFinder()
    detector.run_analysis()


if __name__ == "__main__":
    main()