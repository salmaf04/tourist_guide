"""
Geocoding Service for location-based operations
Handles coordinate retrieval and caching for places and cities
"""

import logging
import time
import requests
import numpy as np
from typing import Dict, List, Tuple, Optional
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import os

logger = logging.getLogger(__name__)

class GeocodingService:
    """Handles geocoding operations and coordinate management"""
    
    def __init__(self):
        """Initialize geocoding service"""
        self.geocoder = Nominatim(user_agent="tourist_guide_app")
        self.coordinate_cache = {}
        self.openrouteservice_api_key = os.getenv('OPENROUTESERVICE_API_KEY')
        
        if not self.openrouteservice_api_key:
            raise ValueError("OPENROUTESERVICE_API_KEY not found in environment variables.")
    
    def get_coordinates_from_city(self, city_name: str) -> Tuple[Optional[float], Optional[float]]:
        """Get coordinates for a city using geopy."""
        cache_key = f"city:{city_name}"
        if cache_key in self.coordinate_cache:
            return self.coordinate_cache[cache_key]

        try:
            location = self.geocoder.geocode(city_name, timeout=10)
            if location:
                coordinates = (location.latitude, location.longitude)
                self.coordinate_cache[cache_key] = coordinates
                return coordinates
            else:
                self.coordinate_cache[cache_key] = (None, None)
                return None, None
        except Exception as e:
            logger.error(f"Error geocoding city {city_name}: {e}")
            self.coordinate_cache[cache_key] = (None, None)
            return None, None

    def get_coordinates_from_place(self, place: Dict) -> Tuple[Optional[float], Optional[float]]:
        """Get coordinates for a specific place."""
        if place.get('coordinates'):
            return place['coordinates']

        place_name = place.get('name', '')
        city = place.get('location', {}).get('city', '')
        cache_key = f"place:{place_name}:{city}"

        if cache_key in self.coordinate_cache:
            return self.coordinate_cache[cache_key]

        try:
            search_query = f"{place_name}, {city}"
            location = self.geocoder.geocode(search_query, timeout=10)
            if location:
                coordinates = (location.latitude, location.longitude)
                self.coordinate_cache[cache_key] = coordinates
                return coordinates
            return self.get_coordinates_from_city(city)
        except Exception as e:
            logger.error(f"Error geocoding place {place_name}: {e}")
            self.coordinate_cache[cache_key] = (None, None)
            return None, None

    def filter_places_by_distance(self, places: List[Dict], max_distance_km: float, 
                                 user_lat: float, user_lon: float) -> Tuple[List[Dict], List[int]]:
        """Filter places by maximum distance."""
        filtered_places = []
        filtered_indices = []

        for i, place in enumerate(places):
            place_lat, place_lon = self.get_coordinates_from_place(place)
            if place_lat is not None and place_lon is not None:
                distance_km = geodesic((user_lat, user_lon), (place_lat, place_lon)).kilometers
                if distance_km <= max_distance_km:
                    filtered_places.append(place)
                    filtered_indices.append(i + 1)
        return filtered_places, filtered_indices

    def map_transport_mode_to_profile(self, transport_mode: str) -> str:
        """Map Spanish transport modes to OpenRouteService profiles."""
        transport_map = {
            "Caminar": "foot-walking",
            "Bicicleta": "cycling-regular",
            "Bus": "bus",
            "Coche/taxi": "driving-car",
            "Otro": "foot-walking"
        }
        return transport_map.get(transport_mode, "foot-walking")

    def calculate_time_matrix_ors(self, places: List[Dict], user_lat: float, 
                                 user_lon: float, transport_mode: str) -> np.ndarray:
        """Calculate time matrix using OpenRouteService API with retry logic."""
        api_key = self.openrouteservice_api_key
        coordinates = [[user_lon, user_lat]]

        for place in places:
            lat, lon = self.get_coordinates_from_place(place)
            if lat is not None and lon is not None:
                coordinates.append([lon, lat])
            else:
                raise RuntimeError(f"Could not get coordinates for place: {place.get('name', 'Unknown')}")

        profile = self.map_transport_mode_to_profile(transport_mode)
        url = f"https://api.openrouteservice.org/v2/matrix/{profile}"
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json; charset=utf-8'
        }
        body = {
            "locations": coordinates,
            "metrics": ["duration"],
            "units": "m"
        }

        # Retry logic with exponential backoff
        max_retries = 3
        base_delay = 2  # seconds
        request_timeout = 30  # seconds
        
        last_error = None
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempting OpenRouteService API call (attempt {attempt + 1}/{max_retries})")
                
                # Set timeout to prevent hanging
                response = requests.post(url, json=body, headers=headers, timeout=request_timeout)
                
                if response.status_code == 200:
                    data = response.json()
                    logger.info("OpenRouteService API call successful")
                    return np.array(data['durations']) / 60  # Convertir a minutos
                elif response.status_code == 504:
                    error_msg = f"OpenRouteService API timeout (504) on attempt {attempt + 1}"
                    logger.warning(error_msg)
                    last_error = f"504 Gateway Timeout: {response.text}"
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)  # Exponential backoff
                        logger.info(f"Waiting {delay} seconds before retry...")
                        time.sleep(delay)
                        continue
                else:
                    error_msg = f"OpenRouteService API failed with status {response.status_code}: {response.text}"
                    logger.error(error_msg)
                    last_error = f"HTTP {response.status_code}: {response.text}"
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        logger.info(f"Waiting {delay} seconds before retry...")
                        time.sleep(delay)
                        continue
                        
            except requests.exceptions.Timeout:
                error_msg = f"OpenRouteService API request timeout on attempt {attempt + 1}"
                logger.warning(error_msg)
                last_error = "Request timeout after 30 seconds"
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    logger.info(f"Waiting {delay} seconds before retry...")
                    time.sleep(delay)
                    continue
                    
            except Exception as e:
                error_msg = f"Error calling OpenRouteService API on attempt {attempt + 1}: {e}"
                logger.error(error_msg)
                last_error = str(e)
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    logger.info(f"Waiting {delay} seconds before retry...")
                    time.sleep(delay)
                    continue
        
        # Si llegamos aquí, todos los intentos fallaron
        logger.error("All OpenRouteService API attempts failed")
        raise RuntimeError(f"Failed to call OpenRouteService API after {max_retries} attempts. Last error: {last_error}")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get statistics about the coordinate cache"""
        return {
            'total_cached': len(self.coordinate_cache),
            'cities_cached': len([k for k in self.coordinate_cache.keys() if k.startswith('city:')]),
            'places_cached': len([k for k in self.coordinate_cache.keys() if k.startswith('place:')])
        }