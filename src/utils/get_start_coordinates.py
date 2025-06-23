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

def get_cities_with_coordinates(db_path: str = "../crawler/db", 
                              user_agent: str = "tourist_guide_app") -> Dict[str, Tuple[float, float]]:
    """
    Consulta la base de datos ChromaDB para extraer todas las ciudades únicas
    y devuelve un diccionario con las ciudades y sus coordenadas del centro.
    
    Args:
        db_path (str): Ruta a la base de datos ChromaDB
        user_agent (str): User agent para el geocodificador
        
    Returns:
        Dict[str, Tuple[float, float]]: Diccionario con formato {ciudad: (latitud, longitud)}
        
    Raises:
        RuntimeError: Si no se puede conectar a la base de datos
        Exception: Si hay errores en la geocodificación
    """
    
    # Verificar que la ruta de la base de datos existe
    if not os.path.exists(db_path):
        raise RuntimeError(f"La ruta de la base de datos no existe: {db_path}")
    
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


def get_city_coordinates(city_name: str, user_agent: str = "tourist_guide_app") -> Optional[Tuple[float, float]]:
    """
    Obtiene las coordenadas del centro de una ciudad específica.
    
    Args:
        city_name (str): Nombre de la ciudad
        user_agent (str): User agent para el geocodificador
        
    Returns:
        Optional[Tuple[float, float]]: Tupla con (latitud, longitud) o None si no se encuentra
    """
    try:
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


def list_cities_in_database(db_path: str = "src/crawler/db") -> Dict[str, int]:
    """
    Lista todas las ciudades en la base de datos con el número de lugares por ciudad.
    
    Args:
        db_path (str): Ruta a la base de datos ChromaDB
        
    Returns:
        Dict[str, int]: Diccionario con formato {ciudad: número_de_lugares}
    """
    
    if not os.path.exists(db_path):
        raise RuntimeError(f"La ruta de la base de datos no existe: {db_path}")
    
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


def print_cities_summary(db_path: str = "src/crawler/db"):
    """
    Imprime un resumen de las ciudades en la base de datos con sus coordenadas.
    
    Args:
        db_path (str): Ruta a la base de datos ChromaDB
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