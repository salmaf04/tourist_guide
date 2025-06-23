#!/usr/bin/env python3
"""
Utilidades para el sistema de guía turística.
Incluye funciones para extraer ciudades de la base de datos y obtener sus coordenadas.
"""

import chromadb
import os
from geopy.geocoders import Nominatim
from typing import Dict, Tuple, Optional
import logging
import time

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _find_database_path() -> str:
    """
    Encuentra automáticamente la ruta correcta de la base de datos ChromaDB.
    Prueba diferentes rutas posibles desde el directorio actual.
    
    Returns:
        str: Ruta válida a la base de datos
        
    Raises:
        RuntimeError: Si no se encuentra ninguna ruta válida
    """
    possible_paths = [
        "src/crawler/db",           # Desde raíz del proyecto
        "crawler/db",               # Desde src/
        "../crawler/db",            # Desde src/utils/
        "../../src/crawler/db",     # Desde subdirectorios
        "./src/crawler/db",         # Variante con ./
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            logger.info(f"Base de datos encontrada en: {path}")
            return path
    
    # Si no se encuentra, intentar crear la ruta absoluta
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Subir dos niveles desde src/utils/ hasta la raíz del proyecto
    project_root = os.path.dirname(os.path.dirname(current_dir))
    absolute_db_path = os.path.join(project_root, "src", "crawler", "db")
    
    if os.path.exists(absolute_db_path):
        logger.info(f"Base de datos encontrada en ruta absoluta: {absolute_db_path}")
        return absolute_db_path
    
    # Listar rutas intentadas para debugging
    logger.error("No se encontró la base de datos en ninguna de estas rutas:")
    for path in possible_paths + [absolute_db_path]:
        logger.error(f"  - {path} (existe: {os.path.exists(path)})")
    
    raise RuntimeError(f"No se pudo encontrar la base de datos ChromaDB en ninguna ubicación conocida")

def get_cities_with_coordinates(db_path: str = None, 
                              user_agent: str = "tourist_guide_app",
                              specific_city: Optional[str] = None) -> Dict[str, Tuple[float, float]]:
    """
    Consulta la base de datos ChromaDB para extraer todas las ciudades únicas
    y devuelve un diccionario con las ciudades y sus coordenadas del centro.
    
    Args:
        db_path (str, optional): Ruta a la base de datos ChromaDB. Si es None, se detecta automáticamente.
        user_agent (str): User agent para el geocodificador
        specific_city (Optional[str]): Si se proporciona, solo devuelve las coordenadas de esta ciudad específica
        
    Returns:
        Dict[str, Tuple[float, float]]: Diccionario con formato {ciudad: (latitud, longitud)}
        Si specific_city se proporciona, devuelve un diccionario con solo esa ciudad
        
    Raises:
        RuntimeError: Si no se puede conectar a la base de datos
        Exception: Si hay errores en la geocodificación
    """
    
    # Si no se proporciona db_path o no existe, detectar automáticamente
    if db_path is None or not os.path.exists(db_path):
        if db_path is not None:
            logger.warning(f"Ruta proporcionada no existe: {db_path}. Detectando automáticamente...")
        db_path = _find_database_path()
    
    cities_coordinates = {}
    
    try:
        # Conectar a ChromaDB
        logger.info(f"Conectando a ChromaDB en: {db_path}")
        chroma_client = chromadb.PersistentClient(path=db_path)
        collection = chroma_client.get_or_create_collection(name="tourist_places")
        
        total_docs = collection.count()
        logger.info(f"Total de documentos en la base de datos: {total_docs}")
        
        if total_docs == 0:
            logger.warning("La base de datos está vacía")
            return cities_coordinates
        
        # Obtener todos los metadatos para extraer las ciudades
        logger.info("Extrayendo metadatos de todos los documentos...")
        all_docs = collection.get(include=["metadatas"])
        
        # Extraer ciudades únicas
        unique_cities = set()
        for metadata in all_docs["metadatas"]:
            city = metadata.get("city")
            if city and city.strip():  # Verificar que la ciudad no esté vacía
                unique_cities.add(city.strip())
        
        # Si se especifica una ciudad específica, filtrar solo esa ciudad
        if specific_city:
            if specific_city in unique_cities:
                unique_cities = {specific_city}
                logger.info(f"Buscando coordenadas específicamente para: {specific_city}")
            else:
                logger.warning(f"Ciudad específica '{specific_city}' no encontrada en la base de datos")
                # Intentar geocodificar directamente aunque no esté en la BD
                try:
                    geolocator = Nominatim(user_agent=user_agent)
                    location = geolocator.geocode(specific_city, timeout=10)
                    if location:
                        coordinates = (location.latitude, location.longitude)
                        cities_coordinates[specific_city] = coordinates
                        logger.info(f"✅ {specific_city} (geocodificación directa): {coordinates}")
                    else:
                        logger.warning(f"⚠️ No se pudieron obtener coordenadas para: {specific_city}")
                except Exception as e:
                    logger.error(f"❌ Error geocodificando {specific_city}: {e}")
                return cities_coordinates
        else:
            logger.info(f"Ciudades únicas encontradas: {len(unique_cities)}")
            logger.info(f"Lista de ciudades: {sorted(unique_cities)}")
        
        # Inicializar geocodificador
        geolocator = Nominatim(user_agent=user_agent)
        
        # Obtener coordenadas para cada ciudad
        for city in sorted(unique_cities):
            try:
                logger.info(f"Geocodificando: {city}")
                
                # Intentar geocodificar la ciudad
                location = geolocator.geocode(city, timeout=10)
                
                if location:
                    coordinates = (location.latitude, location.longitude)
                    cities_coordinates[city] = coordinates
                    logger.info(f"✅ {city}: {coordinates}")
                else:
                    logger.warning(f"⚠️ No se pudieron obtener coordenadas para: {city}")
                
                # Pausa para respetar los límites de la API de Nominatim
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"❌ Error geocodificando {city}: {e}")
                continue
        
        logger.info(f"Proceso completado. Ciudades geocodificadas: {len(cities_coordinates)}")
        
    except Exception as e:
        logger.error(f"Error conectando a ChromaDB: {e}")
        raise RuntimeError(f"No se pudo conectar a la base de datos: {e}")
    
    return cities_coordinates


def get_city_coordinates(city_name: str, user_agent: str = "tourist_guide_app", 
                        db_path: str = None) -> Optional[Tuple[float, float]]:
    """
    Obtiene las coordenadas del centro de una ciudad específica.
    Primero intenta usar la función optimizada con base de datos, luego geocodificación directa.
    
    Args:
        city_name (str): Nombre de la ciudad
        user_agent (str): User agent para el geocodificador
        db_path (str, optional): Ruta a la base de datos ChromaDB. Si es None, se detecta automáticamente.
        
    Returns:
        Optional[Tuple[float, float]]: Tupla con (latitud, longitud) o None si no se encuentra
    """
    try:
        # Primero intentar con la función optimizada que usa la base de datos
        cities_coords = get_cities_with_coordinates(db_path, user_agent, specific_city=city_name)
        if city_name in cities_coords:
            return cities_coords[city_name]
        
        # Si no se encuentra, usar geocodificación directa como fallback
        logger.info(f"Usando geocodificación directa para {city_name}")
        geolocator = Nominatim(user_agent=user_agent)
        location = geolocator.geocode(city_name, timeout=10)
        
        if location:
            return (location.latitude, location.longitude)
        else:
            logger.warning(f"No se encontraron coordenadas para: {city_name}")
            return None
            
    except Exception as e:
        logger.error(f"Error geocodificando {city_name}: {e}")
        return None


def list_cities_in_database(db_path: str = None) -> Dict[str, int]:
    """
    Lista todas las ciudades en la base de datos con el número de lugares por ciudad.
    
    Args:
        db_path (str, optional): Ruta a la base de datos ChromaDB. Si es None, se detecta automáticamente.
        
    Returns:
        Dict[str, int]: Diccionario con formato {ciudad: número_de_lugares}
    """
    
    # Si no se proporciona db_path o no existe, detectar automáticamente
    if db_path is None or not os.path.exists(db_path):
        if db_path is not None:
            logger.warning(f"Ruta proporcionada no existe: {db_path}. Detectando automáticamente...")
        db_path = _find_database_path()
    
    try:
        # Conectar a ChromaDB
        chroma_client = chromadb.PersistentClient(path=db_path)
        collection = chroma_client.get_or_create_collection(name="tourist_places")
        
        # Obtener todos los metadatos
        all_docs = collection.get(include=["metadatas"])
        
        # Contar lugares por ciudad
        cities_count = {}
        for metadata in all_docs["metadatas"]:
            city = metadata.get("city", "Unknown")
            cities_count[city] = cities_count.get(city, 0) + 1
        
        return cities_count
        
    except Exception as e:
        logger.error(f"Error accediendo a la base de datos: {e}")
        raise RuntimeError(f"No se pudo acceder a la base de datos: {e}")


def print_cities_summary(db_path: str = None):
    """
    Imprime un resumen de las ciudades en la base de datos con sus coordenadas.
    
    Args:
        db_path (str, optional): Ruta a la base de datos ChromaDB. Si es None, se detecta automáticamente.
    """
    try:
        # Obtener ciudades con conteo
        cities_count = list_cities_in_database(db_path)
        
        print("🏙️ RESUMEN DE CIUDADES EN LA BASE DE DATOS")
        print("=" * 50)
        print(f"Total de ciudades: {len(cities_count)}")
        print(f"Total de lugares: {sum(cities_count.values())}")
        print()
        
        # Obtener coordenadas
        cities_coordinates = get_cities_with_coordinates(db_path)
        
        print("📍 CIUDADES CON COORDENADAS:")
        print("-" * 50)
        
        for city in sorted(cities_count.keys()):
            count = cities_count[city]
            coords = cities_coordinates.get(city, "No disponible")
            
            if isinstance(coords, tuple):
                coords_str = f"({coords[0]:.4f}, {coords[1]:.4f})"
            else:
                coords_str = str(coords)
            
            print(f"{city:20} | {count:3d} lugares | {coords_str}")
        
        print()
        print(f"✅ Ciudades geocodificadas exitosamente: {len(cities_coordinates)}")
        print(f"⚠️ Ciudades sin coordenadas: {len(cities_count) - len(cities_coordinates)}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    """
    Ejemplo de uso del módulo
    """
    print("🌍 UTILIDADES DE GEOCODIFICACIÓN - GUÍA TURÍSTICA")
    print("=" * 60)
    
    try:
        # Mostrar resumen completo
        print_cities_summary()
        
        print("\n" + "=" * 60)
        print("📋 DICCIONARIO DE CIUDADES Y COORDENADAS:")
        print("=" * 60)
        
        # Obtener y mostrar el diccionario de ciudades con coordenadas
        cities_dict = get_cities_with_coordinates()
        
        for city, coords in sorted(cities_dict.items()):
            print(f"'{city}': {coords},")
            
    except Exception as e:
        print(f"❌ Error ejecutando el ejemplo: {e}")