"""
Place Filter for filtering and processing tourist places
Handles place filtering by city, distance, and other criteria
"""

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

class PlaceFilter:
    """Handles filtering operations for tourist places"""
    
    def __init__(self):
        """Initialize place filter"""
        pass
    
    def is_city_or_region_name(self, place_name: str, target_city: str) -> bool:
        """Check if a place name is actually a city or region name."""
        # Lista de nombres de ciudades y regiones españolas comunes
        cities_regions = {
            'madrid', 'barcelona', 'valencia', 'sevilla', 'bilbao', 'granada', 'toledo', 'salamanca',
            'cataluña', 'catalunya', 'andalucía', 'andalucia', 'país vasco', 'pais vasco', 'euskadi',
            'castilla y león', 'castilla y leon', 'castilla-la mancha', 'comunidad de madrid',
            'comunidad valenciana', 'galicia', 'asturias', 'cantabria', 'aragón', 'aragon',
            'navarra', 'la rioja', 'extremadura', 'murcia', 'islas baleares', 'baleares',
            'islas canarias', 'canarias', 'ceuta', 'melilla', 'españa', 'spain'
        }
        
        place_name_lower = place_name.lower().strip()
        target_city_lower = target_city.lower().strip()
        
        # Si el nombre del lugar es exactamente igual al nombre de la ciudad objetivo
        if place_name_lower == target_city_lower:
            return True
            
        # Si el nombre del lugar está en la lista de ciudades/regiones
        if place_name_lower in cities_regions:
            return True
            
        # Si el nombre del lugar es muy corto (probablemente una ciudad)
        if len(place_name_lower) <= 3:
            return True
            
        # Si el nombre contiene palabras como "ciudad de", "provincia de", etc.
        generic_terms = ['ciudad de', 'provincia de', 'comunidad de', 'región de', 'area de']
        for term in generic_terms:
            if term in place_name_lower:
                return True
                
        return False

    def filter_places_by_city(self, places: List[Dict], target_city: str) -> List[Dict]:
        """Filter places by city and remove city/region names and duplicates."""
        # Primero filtrar por ciudad
        city_places = [place for place in places if place['location']['city'].lower() == target_city.lower()]
        
        # Filtrar lugares que son nombres de ciudades/regiones
        filtered_places = []
        for place in city_places:
            place_name = place.get('name', '')
            if not self.is_city_or_region_name(place_name, target_city):
                filtered_places.append(place)
            else:
                logger.info(f"Filtered out city/region name: {place_name}")
        return filtered_places
    
    def get_available_cities(self, places_data: List[Dict]) -> List[str]:
        """Get list of available cities from places data"""
        available_cities = set()
        for place in places_data:
            available_cities.add(place.get('location', {}).get('city', 'Unknown'))
        return sorted(list(available_cities))
    
    def filter_and_sort_places(self, places: List[Dict], enhanced_scores: List[float], 
                              max_places: int = 50) -> List[Dict]:
        """Filter and sort places by enhanced scores"""
        # Ordenar lugares por puntuación mejorada
        sorted_places = sorted(zip(places, enhanced_scores), 
                             key=lambda x: x[1], reverse=True)
        
        # Limitar a los mejores lugares
        sorted_places = sorted_places[:max_places]
        
        return [place for place, score in sorted_places]
    
    def print_similarity_ranking(self, places_with_similarity: List[tuple], user_preferences: Dict):
        """Print places ordered by similarity for debugging"""
        print("\n" + "="*80)
        print("🎯 LUGARES ORDENADOS POR SIMILITUD COSENO")
        print("="*80)
        
        # Mostrar contexto de preferencias del usuario
        print("👤 PREFERENCIAS DEL USUARIO:")
        if 'category_interest' in user_preferences:
            high_prefs = [(cat, score) for cat, score in user_preferences['category_interest'].items() if score >= 4]
            if high_prefs:
                print(f"   🔥 Intereses altos: {', '.join([f'{cat}({score}/5)' for cat, score in high_prefs])}")
        if user_preferences.get('user_notes'):
            print(f"   📝 Notas: {user_preferences['user_notes'][:60]}...")
        print(f"   🏙️ Ciudad: {user_preferences.get('city', 'N/A')}")
        print("-" * 80)
        
        # Mostrar lugares ordenados
        for i, (place, embedding, similarity) in enumerate(places_with_similarity, 1):
            place_name = place.get('name', 'Lugar desconocido')
            place_category = place.get('category', 'general')
            
            # Agregar indicador visual para similitudes altas
            if similarity >= 0.7:
                indicator = "🔥"
            elif similarity >= 0.5:
                indicator = "⭐"
            elif similarity >= 0.3:
                indicator = "✅"
            else:
                indicator = "📍"
            
            print(f"{i:2d}. {similarity:.4f} {indicator} | {place_name} ({place_category})")
        
        print("="*80)
        print(f"📊 Total de lugares ordenados: {len(places_with_similarity)}")
        if places_with_similarity:
            similarities = [item[2] for item in places_with_similarity]
            print(f"📊 Rango de similitud: {min(similarities):.4f} - {max(similarities):.4f}")
            avg_similarity = sum(similarities) / len(similarities)
            print(f"📊 Similitud promedio: {avg_similarity:.4f}")
        print("="*80 + "\n")