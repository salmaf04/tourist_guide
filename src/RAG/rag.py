from typing import List, Dict, Tuple
import numpy as np
from typing import List, Dict
from .mock import tourist_collection

def vectorize_preferences(preferences: dict) -> np.ndarray:
    """
    Vectoriza las preferencias para comparación con atracciones.
    """
    categories = ["history", "engineering", "culture", "beach", "nature", "shopping", "food"]
    vector = np.array([preferences.get(cat, 0) for cat in categories], dtype=float)
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector

def retrieve_attractions(location: str, user_prefs: dict, top_k: int = 10) -> List[dict]:
    """
    Recupera atracciones turísticas relevantes basadas en ubicación y preferencias usando dot product.
    """
    # Filtrar por ubicación
    location_results = [doc for doc in tourist_collection if doc["location"].lower() == location.lower()]
    if len(location_results) < top_k:
        location_results = tourist_collection  # Si no hay suficientes, usar todos

    prefs_vector = vectorize_preferences(user_prefs)
    scored_attractions = []
    for doc in location_results:
        # Vectorizar la categoría y tags del sitio turístico
        attraction_vector = vectorize_preferences(
            {doc.get("category", ""): 1.0, **{tag: 0.3 for tag in doc.get("tags", [])}}
        )
        similarity = np.dot(prefs_vector, attraction_vector)
        scored_attractions.append((similarity, doc, attraction_vector))

    scored_attractions.sort(reverse=True, key=lambda x: x[0])
    # Retorna los top_k lugares, junto con sus vectores
    return [(att[1], att[2]) for att in scored_attractions[:top_k]]

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calcula la distancia en kilómetros entre dos puntos geográficos usando la fórmula de Haversine.
    """
    R = 6371  # Radio de la Tierra en km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def compute_distance_matrix(sites: List[Dict]) -> np.ndarray:
    """
    Calcula la matriz de distancias reales (en km) entre todos los sitios turísticos.
    Cada sitio debe tener 'lat' y 'lon' en su diccionario.
    """
    n = len(sites)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            lat1, lon1 = sites[i]['lat'], sites[i]['lon']
            lat2, lon2 = sites[j]['lat'], sites[j]['lon']
            dist = haversine_distance(lat1, lon1, lat2, lon2)
            mat[i, j] = mat[j, i] = dist
    return mat

def get_interest_places_and_vectors(
    location: str, user_prefs: dict, top_k: int = 10
) -> Tuple[List[dict], np.ndarray, np.ndarray, List[np.ndarray]]:
    """
    Devuelve:
    - Lista de lugares de interés (dicts)
    - Matriz de distancias entre lugares (np.ndarray)
    - Vector unitario de preferencias del usuario (np.ndarray)
    - Lista de vectores unitarios de preferencias de cada sitio (List[np.ndarray])
    """
    # Recuperar lugares y sus vectores
    places_and_vectors = retrieve_attractions(location, user_prefs, top_k=top_k)
    places = [p[0] for p in places_and_vectors]
    site_vectors = [p[1] for p in places_and_vectors]
    user_vector = vectorize_preferences(user_prefs)
    distance_matrix = compute_distance_matrix(places)
    return places, distance_matrix, user_vector, site_vectors

def filter_places_by_distance(
    places: List[dict], user_lat: float, user_lon: float, max_distance_km: float
) -> List[dict]:
    """
    Filtra los lugares para que solo se incluyan aquellos cuya distancia desde el punto de partida del usuario
    no exceda max_distance_km.
    """
    filtered = []
    for place in places:
        lat = place.get("lat")
        lon = place.get("lon")
        if lat is not None and lon is not None:
            dist = haversine_distance(user_lat, user_lon, lat, lon)
            if dist <= max_distance_km:
                place = place.copy()
                place["distance_from_user"] = dist
                filtered.append(place)
                
    return filtered
