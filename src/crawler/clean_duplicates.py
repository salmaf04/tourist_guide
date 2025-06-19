#!/usr/bin/env python3
"""
Script para eliminar duplicados exactos de ChromaDB de forma segura.
Incluye backup automático y confirmación del usuario.
"""

import os
import sys
import shutil
from pathlib import Path
from collections import defaultdict
import chromadb
import re
from datetime import datetime

# Agregar el directorio src al path para importaciones
current_dir = Path(__file__).parent
src_dir = current_dir.parent
sys.path.insert(0, str(src_dir))


class DuplicateCleaner:
    """Limpiador de duplicados para ChromaDB."""
    
    def __init__(self, db_path="./db/"):
        """Inicializa el limpiador."""
        self.db_path = db_path
        self.collection = None
        self.backup_path = None
        
    def normalize_text(self, text):
        """Normaliza texto para comparación."""
        if not text:
            return ""
        
        text = text.lower().strip()
        
        # Remover acentos
        replacements = {
            'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u', 'ü': 'u',
            'ñ': 'n', 'ç': 'c'
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def connect_to_database(self):
        """Conecta a la base de datos ChromaDB."""
        try:
            if not os.path.exists(self.db_path):
                print(f"❌ No se encontró la base de datos en: {self.db_path}")
                return False
            
            chroma_client = chromadb.PersistentClient(path=self.db_path)
            
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
    
    def create_backup(self):
        """Crea una copia de seguridad de la base de datos."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.backup_path = f"{self.db_path}_backup_{timestamp}"
            
            if os.path.exists(self.backup_path):
                shutil.rmtree(self.backup_path)
            
            shutil.copytree(self.db_path, self.backup_path)
            print(f"✅ Backup creado en: {self.backup_path}")
            return True
        except Exception as e:
            print(f"❌ Error creando backup: {e}")
            return False
    
    def find_duplicates(self):
        """Encuentra duplicados exactos y determina cuáles eliminar."""
        if not self.collection:
            return []
        
        print("🔍 Analizando duplicados...")
        
        all_docs = self.collection.get(include=["metadatas", "documents"])
        
        # Agrupar por nombre normalizado + ciudad normalizada
        groups = defaultdict(list)
        
        for i, metadata in enumerate(all_docs["metadatas"]):
            name = metadata.get("name", "")
            city = metadata.get("city", "")
            
            key = f"{self.normalize_text(name)}|{self.normalize_text(city)}"
            
            groups[key].append({
                'id': all_docs["ids"][i],
                'name': name,
                'city': city,
                'category': metadata.get("category", ""),
                'description': metadata.get("description", ""),
                'description_length': len(metadata.get("description", "")),
                'index': i
            })
        
        # Determinar qué eliminar (mantener el que tenga mejor descripción)
        ids_to_remove = []
        duplicate_groups = []
        
        for key, items in groups.items():
            if len(items) > 1:
                # Ordenar por longitud de descripción (descendente)
                items.sort(key=lambda x: x['description_length'], reverse=True)
                
                duplicate_groups.append({
                    'keep': items[0],
                    'remove': items[1:],
                    'total': len(items)
                })
                
                # Añadir IDs a eliminar
                for item in items[1:]:
                    ids_to_remove.append(item['id'])
        
        return duplicate_groups, ids_to_remove
    
    def show_duplicates_summary(self, duplicate_groups):
        """Muestra un resumen de los duplicados encontrados."""
        if not duplicate_groups:
            print("✅ No se encontraron duplicados exactos")
            return
        
        print(f"\n❌ DUPLICADOS ENCONTRADOS: {len(duplicate_groups)} grupos")
        print("=" * 60)
        
        total_to_remove = sum(len(group['remove']) for group in duplicate_groups)
        print(f"📊 Total de documentos a eliminar: {total_to_remove}")
        print()
        
        for i, group in enumerate(duplicate_groups, 1):
            print(f"📍 GRUPO {i}: {group['keep']['name']} ({group['keep']['city']})")
            print(f"   Total duplicados: {group['total']}")
            print(f"   ✅ MANTENER: ID {group['keep']['id'][:12]}... (desc: {group['keep']['description_length']} chars)")
            
            for item in group['remove']:
                print(f"   🗑️  ELIMINAR: ID {item['id'][:12]}... (desc: {item['description_length']} chars)")
            print()
    
    def confirm_deletion(self, total_to_remove):
        """Solicita confirmación del usuario para eliminar."""
        print("⚠️  ADVERTENCIA: Esta operación eliminará datos permanentemente")
        print(f"Se eliminarán {total_to_remove} documentos duplicados")
        print()
        
        response = input("¿Deseas continuar? Escribe 'ELIMINAR' para confirmar: ").strip()
        
        return response == "ELIMINAR"
    
    def remove_duplicates(self, ids_to_remove):
        """Elimina los documentos duplicados."""
        if not ids_to_remove:
            print("✅ No hay duplicados para eliminar")
            return True
        
        try:
            print(f"🗑️  Eliminando {len(ids_to_remove)} documentos duplicados...")
            
            # Eliminar en lotes para evitar problemas de memoria
            batch_size = 50
            for i in range(0, len(ids_to_remove), batch_size):
                batch = ids_to_remove[i:i + batch_size]
                self.collection.delete(ids=batch)
                print(f"   Eliminados {min(i + batch_size, len(ids_to_remove))}/{len(ids_to_remove)} documentos")
            
            print(f"✅ Eliminación completada")
            print(f"📊 Documentos restantes: {self.collection.count()}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error eliminando duplicados: {e}")
            return False
    
    def save_removal_log(self, duplicate_groups, ids_to_remove):
        """Guarda un log de los elementos eliminados."""
        try:
            log_file = f"duplicates_removed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write(f"LOG DE ELIMINACIÓN DE DUPLICADOS\n")
                f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Base de datos: {self.db_path}\n")
                f.write(f"Backup: {self.backup_path}\n")
                f.write(f"Total eliminados: {len(ids_to_remove)}\n")
                f.write("=" * 60 + "\n\n")
                
                for i, group in enumerate(duplicate_groups, 1):
                    f.write(f"GRUPO {i}: {group['keep']['name']} ({group['keep']['city']})\n")
                    f.write(f"MANTENIDO: {group['keep']['id']}\n")
                    f.write("ELIMINADOS:\n")
                    for item in group['remove']:
                        f.write(f"  - {item['id']} | {item['name']} | {item['category']}\n")
                    f.write("\n")
            
            print(f"📝 Log guardado en: {log_file}")
            
        except Exception as e:
            print(f"⚠️  Error guardando log: {e}")
    
    def run_cleanup(self):
        """Ejecuta el proceso completo de limpieza."""
        print("🧹 LIMPIADOR DE DUPLICADOS - ChromaDB")
        print("=" * 50)
        
        # Conectar a la base de datos
        if not self.connect_to_database():
            return False
        
        # Encontrar duplicados
        duplicate_groups, ids_to_remove = self.find_duplicates()
        
        # Mostrar resumen
        self.show_duplicates_summary(duplicate_groups)
        
        if not duplicate_groups:
            return True
        
        # Confirmar eliminación
        if not self.confirm_deletion(len(ids_to_remove)):
            print("❌ Operación cancelada por el usuario")
            return False
        
        # Crear backup
        if not self.create_backup():
            print("❌ No se pudo crear backup. Abortando eliminación.")
            return False
        
        # Eliminar duplicados
        if self.remove_duplicates(ids_to_remove):
            # Guardar log
            self.save_removal_log(duplicate_groups, ids_to_remove)
            
            print("\n✅ LIMPIEZA COMPLETADA EXITOSAMENTE")
            print(f"📁 Backup disponible en: {self.backup_path}")
            return True
        else:
            print("\n❌ Error durante la eliminación")
            return False


def main():
    """Función principal del script."""
    print("🧹 Script de Limpieza de Duplicados")
    print("Este script eliminará duplicados exactos de ChromaDB")
    print("-" * 50)
    
    # Crear limpiador y ejecutar
    cleaner = DuplicateCleaner()
    success = cleaner.run_cleanup()
    
    if success:
        print("\n🎉 Proceso completado exitosamente")
    else:
        print("\n💥 El proceso falló o fue cancelado")


if __name__ == "__main__":
    main()