#!/usr/bin/env python3
"""
Script para limpiar completamente la base de datos ChromaDB
"""

import sys
import os
from pathlib import Path

# Agregar el directorio src al path
sys.path.append(str(Path(__file__).parent.parent))

import chromadb
import shutil
import time
import gc


def clear_database(db_path="../db"):
    """Limpia completamente la base de datos ChromaDB"""
    
    print("🗑️  LIMPIANDO BASE DE DATOS TURÍSTICA")
    print("=" * 50)
    
    try:
        # Verificar si existe la base de datos
        if os.path.exists(db_path):
            print(f"📁 Base de datos encontrada en: {db_path}")
            
            # Conectar y obtener estadísticas antes de limpiar
            chroma_client = None
            collection = None
            count_before = 0
            
            try:
                chroma_client = chromadb.PersistentClient(path=db_path)
                collection = chroma_client.get_or_create_collection(name="tourist_places")
                count_before = collection.count()
                print(f"📊 Lugares actuales en la base de datos: {count_before}")
                
                # Eliminar todos los elementos de la colección en lugar de eliminar la colección
                if count_before > 0:
                    # Obtener todos los IDs
                    all_data = collection.get()
                    if all_data['ids']:
                        collection.delete(ids=all_data['ids'])
                        print("✅ Todos los elementos eliminados de la colección")
                
                # Cerrar conexiones explícitamente
                del collection
                del chroma_client
                gc.collect()
                time.sleep(2)  # Dar tiempo para que se liberen los recursos
                
            except Exception as e:
                print(f"⚠️  Error accediendo a la colección: {e}")
            
            # Intentar eliminar el directorio con reintentos
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    if os.path.exists(db_path):
                        print(f"🔄 Intento {attempt + 1} de eliminar directorio...")
                        shutil.rmtree(db_path)
                        print(f"✅ Directorio de base de datos eliminado: {db_path}")
                        break
                except PermissionError as e:
                    if attempt < max_retries - 1:
                        print(f"⚠️  Archivo en uso, esperando 3 segundos...")
                        time.sleep(3)
                    else:
                        print(f"❌ No se pudo eliminar el directorio después de {max_retries} intentos")
                        print("💡 Solución alternativa: La colección fue vaciada, pero el directorio permanece")
                        break
            
            # Recrear el directorio vacío
            os.makedirs(db_path, exist_ok=True)
            print(f"✅ Directorio de base de datos recreado: {db_path}")
            
        else:
            print(f"⚠️  No se encontró base de datos en: {db_path}")
        
        # Verificar que la limpieza fue exitosa
        time.sleep(1)  # Pequeña pausa antes de verificar
        chroma_client = chromadb.PersistentClient(path=db_path)
        collection = chroma_client.get_or_create_collection(name="tourist_places")
        count_after = collection.count()
        
        print(f"\n✅ LIMPIEZA COMPLETADA")
        print(f"📊 Lugares en la base de datos después de limpiar: {count_after}")
        
        if count_after == 0:
            print("🎉 Base de datos limpiada exitosamente")
            print("\n💡 Ahora puedes ejecutar el crawler para todas las ciudades:")
            print("   py main_fixed.py")
        else:
            print("❌ La limpieza no fue completamente exitosa")
            
        # Cerrar conexiones
        del collection
        del chroma_client
        gc.collect()
            
        return count_after == 0
        
    except Exception as e:
        print(f"❌ Error limpiando la base de datos: {e}")
        return False


if __name__ == "__main__":
    try:
        success = clear_database()
        if success:
            print("\n🚀 ¡Listo para ejecutar el crawler con todas las ciudades!")
        else:
            print("\n⚠️  Hubo problemas durante la limpieza")
    except KeyboardInterrupt:
        print("\n⏹️  Operación interrumpida por el usuario")
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        sys.exit(1)