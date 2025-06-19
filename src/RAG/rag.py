import numpy as np
import requests
import re
import time
from typing import Dict, List, Tuple, Any, Optional, Set
from sentence_transformers import SentenceTransformer
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter, defaultdict
import logging
import os
from dotenv import load_dotenv
import chromadb
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from agent_generator.client import GeminiClient


# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGPlanner:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", chroma_db_path: str = "src/crawler/db/"):
        """
        Initialize the RAG Planner with NLP capabilities
        """
        self.model = SentenceTransformer(model_name)
        self.chroma_db_path = chroma_db_path

        # Initialize ChromaDB
        self._init_chromadb()
        self.places_data = self._load_places_from_chroma()

        # Initialize geocoder
        self.geocoder = Nominatim(user_agent="tourist_guide_app")
        self.coordinate_cache = {}

        # Initialize Gemini client
        self.gemini_client = GeminiClient()

        # Get API keys
        self.openrouteservice_api_key = os.getenv('OPENROUTESERVICE_API_KEY')

        # Initialize NLP components
        self._init_nlp_components()

        # Validate API keys
        if not self.openrouteservice_api_key:
            raise ValueError("OPENROUTESERVICE_API_KEY not found in environment variables.")

        logger.info("RAG Planner initialized with Gemini client, API keys, and NLP components successfully.")

    def _init_nlp_components(self):
        """Initialize essential NLP components for query processing."""
        try:
            # Download required NLTK data
            self._download_nltk_data()
            
            # Get stopwords using NLTK
            self.spanish_stopwords = self._get_nltk_stopwords()

            # Initialize NLTK stemmer for Spanish
            self.stemmer = SnowballStemmer('spanish')
            
            # Tourism-related keywords for enhanced processing
            self.tourism_keywords = {
                'attractions': ['museo', 'catedral', 'iglesia', 'palacio', 'castillo', 'parque', 'plaza', 'mercado'],
                'activities': ['visitar', 'ver', 'conocer', 'explorar', 'caminar', 'disfrutar', 'admirar'],
                'sentiments': ['hermoso', 'bonito', 'increíble', 'impresionante', 'maravilloso', 'espectacular'],
                'time_expressions': ['mañana', 'tarde', 'noche', 'día', 'hora', 'tiempo', 'duración'],
                'locations': ['centro', 'histórico', 'antiguo', 'moderno', 'tradicional', 'típico']
            }

            logger.info("NLP components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing NLP components: {e}")
            self.stemmer = None

    def _download_nltk_data(self):
        """Download required NLTK data packages."""
        nltk_downloads = ['stopwords', 'punkt']
        
        for package in nltk_downloads:
            try:
                nltk.data.find(f'tokenizers/{package}')
            except LookupError:
                try:
                    nltk.download(package, quiet=True)
                    logger.info(f"Downloaded NLTK package: {package}")
                except Exception as e:
                    logger.warning(f"Failed to download NLTK package {package}: {e}")

    def _get_nltk_stopwords(self) -> Set[str]:
        """Get Spanish and English stopwords using NLTK."""
        try:
            # Get Spanish stopwords
            spanish_stops = set(stopwords.words('spanish'))
            
            # Get English stopwords as fallback
            try:
                english_stops = set(stopwords.words('english'))
            except:
                english_stops = set()
            
            # Combine both sets
            combined_stops = spanish_stops.union(english_stops)
            
            # Add some custom tourism-related stopwords
            custom_stops = {
                'www', 'http', 'https', 'com', 'org', 'es', 'html', 'php',
                'turismo', 'tourism', 'tourist', 'guide', 'guía', 'información',
                'info', 'página', 'page', 'web', 'site', 'sitio'
            }
            
            combined_stops.update(custom_stops)
            
            logger.info(f"Loaded {len(combined_stops)} stopwords from NLTK")
            return combined_stops
            
        except Exception as e:
            logger.error(f"Error loading NLTK stopwords: {e}")
            # Fallback to a minimal set
            return {'de', 'la', 'el', 'en', 'y', 'a', 'que', 'es', 'se', 'no', 'te', 'lo', 'le', 'da', 'su', 'por', 'son', 'con', 'para', 'al', 'del', 'los', 'las', 'un', 'una'}

    def _init_chromadb(self):
        """Initialize ChromaDB client and collection."""
        try:
            self.chroma_client = chromadb.PersistentClient(path=self.chroma_db_path)
            self.chroma_collection = self.chroma_client.get_or_create_collection(name="tourist_places")
            logger.info(f"ChromaDB initialized. Collection has {self.chroma_collection.count()} documents.")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise RuntimeError(f"ChromaDB initialization failed: {e}")

    def _safe_eval_coordinates(self, coord_str: str) -> Optional[Tuple[float, float]]:
        """Safely evaluate coordinate strings from metadata."""
        if not coord_str or coord_str in ["None", "Unknown", ""]:
            return None
        
        try:
            # Try to evaluate as Python literal
            result = eval(coord_str)
            if isinstance(result, (list, tuple)) and len(result) == 2:
                lat, lon = result
                if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
                    return (float(lat), float(lon))
            return None
        except:
            return None

    def _load_places_from_chroma(self) -> List[Dict]:
        """Load places data from ChromaDB."""
        try:
            all_docs = self.chroma_collection.get(include=["metadatas", "documents"])

            if not all_docs["documents"]:
                raise RuntimeError("No documents found in ChromaDB. Run the crawler first.")

            places_data = []
            processed_urls = set()

            for doc_id, doc_text, metadata in zip(all_docs["ids"], all_docs["documents"], all_docs["metadatas"]):
                place_name = metadata.get("name", "Tourist Site")
                place_city = metadata.get("city", "Unknown")
                place_key = f"{place_name}_{place_city}"
                
                if place_key in processed_urls:
                    continue
                processed_urls.add(place_key)

                place = {
                    "name": place_name,
                    "type": "tourist_site",
                    "description": doc_text[:500] + "..." if len(doc_text) > 500 else doc_text,
                    "category": metadata.get("category", "general"),
                    "bestTimeToVisit": "Year-round",
                    "estimatedVisitDuration": "2 hours",
                    "location": {
                        "city": place_city,
                        "country": "Spain"
                    },
                    "chroma_doc_id": doc_id,
                    "coordinates": self._safe_eval_coordinates(metadata.get("coordinates", "None"))
                }
                places_data.append(place)

            logger.info(f"Loaded {len(places_data)} places from ChromaDB")
            return places_data

        except Exception as e:
            logger.error(f"Failed to load places from ChromaDB: {e}")
            raise RuntimeError(f"ChromaDB is required but failed to load data: {e}")

    def _get_coordinates_from_city(self, city_name: str) -> Tuple[Optional[float], Optional[float]]:
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

    def _get_coordinates_from_place(self, place: Dict) -> Tuple[Optional[float], Optional[float]]:
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
            return self._get_coordinates_from_city(city)
        except Exception as e:
            logger.error(f"Error geocoding place {place_name}: {e}")
            self.coordinate_cache[cache_key] = (None, None)
            return None, None

    def _map_transport_mode_to_profile(self, transport_mode: str) -> str:
        """Map Spanish transport modes to OpenRouteService profiles."""
        transport_map = {
            "Caminar": "foot-walking",
            "Bicicleta": "cycling-regular",
            "Bus": "bus",
            "Coche/taxi": "driving-car",
            "Otro": "foot-walking"
        }
        return transport_map.get(transport_mode, "foot-walking")

    def _calculate_time_matrix_ors(self, places: List[Dict], user_lat: float, user_lon: float, transport_mode: str) -> np.ndarray:
        """Calculate time matrix using OpenRouteService API with retry logic."""
        api_key = self.openrouteservice_api_key
        coordinates = [[user_lon, user_lat]]

        for place in places:
            lat, lon = self._get_coordinates_from_place(place)
            if lat is not None and lon is not None:
                coordinates.append([lon, lat])
            else:
                raise RuntimeError(f"Could not get coordinates for place: {place.get('name', 'Unknown')}")

        profile = self._map_transport_mode_to_profile(transport_mode)
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

    def _generate_place_embeddings(self, places: List[Dict]) -> List[np.ndarray]:
        """Generate embeddings for places."""
        embeddings = []
        for place in places:
            text_representation = f"""
Name: {place.get('name', '')}
Type: {place.get('type', '')}
Description: {place.get('description', '')}
Category: {place.get('category', '')}
Best Time to Visit: {place.get('bestTimeToVisit', '')}
Location: {place.get('location', {}).get('city', '')} {place.get('location', {}).get('country', '')}
            """.strip()
            embedding = self.model.encode(text_representation)
            embeddings.append(embedding)
        return embeddings

    def _generate_user_embedding(self, user_preferences: Dict) -> np.ndarray:
        """Generate embedding for user preferences."""
        preferences_text = []
        if 'category_interest' in user_preferences:
            for category, score in user_preferences['category_interest'].items():
                if score == 5:
                    preferences_text.append(f"Absolutely loves {category}")
                elif score == 4:
                    preferences_text.append(f"Really enjoys {category}")
                elif score == 3:
                    preferences_text.append(f"Moderately interested in {category}")
                elif score == 2:
                    preferences_text.append(f"Limited interest in {category}")
                elif score == 1:
                    preferences_text.append(f"Dislikes {category}")
        if 'transport_modes' in user_preferences:
            preferences_text.append(f"Prefers: {', '.join(user_preferences['transport_modes'])}")
        if 'user_notes' in user_preferences and user_preferences['user_notes']:
            preferences_text.append(f"Notes: {user_preferences['user_notes']}")
        if 'city' in user_preferences:
            preferences_text.append(f"Visiting {user_preferences['city']}")
        if 'available_hours' in user_preferences:
            preferences_text.append(f"Has {user_preferences['available_hours']} hours")
        full_text = ". ".join(preferences_text)
        return self.model.encode(full_text)

    def _filter_places_by_distance(self, places: List[Dict], max_distance_km: float, user_lat: float, user_lon: float) -> Tuple[List[Dict], List[int]]:
        """Filter places by maximum distance."""
        filtered_places = []
        filtered_indices = []

        for i, place in enumerate(places):
            place_lat, place_lon = self._get_coordinates_from_place(place)
            if place_lat is not None and place_lon is not None:
                distance_km = geodesic((user_lat, user_lon), (place_lat, place_lon)).kilometers
                if distance_km <= max_distance_km:
                    filtered_places.append(place)
                    filtered_indices.append(i + 1)
        return filtered_places, filtered_indices

    def _is_city_or_region_name(self, place_name: str, target_city: str) -> bool:
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

    def _remove_duplicate_places(self, places: List[Dict]) -> List[Dict]:
        """Remove duplicate places based on name similarity and location."""
        if not places:
            return places
            
        unique_places = []
        seen_names = set()
        
        for place in places:
            place_name = place.get('name', '').strip()
            place_name_lower = place_name.lower()
            
            # Normalizar el nombre (quitar artículos, espacios extra, etc.)
            normalized_name = re.sub(r'\b(el|la|los|las|de|del|de la)\b', '', place_name_lower)
            normalized_name = re.sub(r'\s+', ' ', normalized_name).strip()
            
            # Verificar si ya hemos visto este nombre o uno muy similar
            is_duplicate = False
            for seen_name in seen_names:
                # Calcular similitud simple
                if normalized_name == seen_name or normalized_name in seen_name or seen_name in normalized_name:
                    is_duplicate = True
                    break
                    
                # Verificar si los nombres son muy similares (diferencia de 1-2 caracteres)
                if len(normalized_name) > 5 and len(seen_name) > 5:
                    # Calcular distancia de Levenshtein simple
                    if abs(len(normalized_name) - len(seen_name)) <= 2:
                        common_chars = sum(1 for a, b in zip(normalized_name, seen_name) if a == b)
                        similarity = common_chars / max(len(normalized_name), len(seen_name))
                        if similarity > 0.8:
                            is_duplicate = True
                            break
            
            if not is_duplicate:
                seen_names.add(normalized_name)
                unique_places.append(place)
                
        logger.info(f"Removed {len(places) - len(unique_places)} duplicate places")
        return unique_places

    def _filter_places_by_city(self, places: List[Dict], target_city: str) -> List[Dict]:
        """Filter places by city and remove city/region names and duplicates."""
        # Primero filtrar por ciudad
        city_places = [place for place in places if place['location']['city'].lower() == target_city.lower()]
        
        # Filtrar lugares que son nombres de ciudades/regiones
        filtered_places = []
        for place in city_places:
            place_name = place.get('name', '')
            if not self._is_city_or_region_name(place_name, target_city):
                filtered_places.append(place)
            else:
                logger.info(f"Filtered out city/region name: {place_name}")
        
        # Remover duplicados
        unique_places = self._remove_duplicate_places(filtered_places)
        
        logger.info(f"City filtering: {len(places)} -> {len(city_places)} -> {len(filtered_places)} -> {len(unique_places)} places")
        return unique_places

    def _semantic_search_chroma(self, query: str, n_results: int = 50) -> List[Dict]:
        """Perform semantic search in ChromaDB with deduplication."""
        try:
            search_results = self.chroma_collection.query(
                query_texts=[query],
                n_results=n_results,
                include=["metadatas", "documents", "distances"]
            )

            if not search_results["documents"] or not search_results["documents"][0]:
                return []

            places_data = []
            processed_urls = set()

            for i, (doc_text, metadata, distance) in enumerate(zip(
                search_results["documents"][0],
                search_results["metadatas"][0],
                search_results["distances"][0]
            )):
                place_name = metadata.get("name", "Tourist Site")
                place_city = metadata.get("city", "Unknown")
                place_key = f"{place_name}_{place_city}"
                
                if place_key in processed_urls:
                    continue
                processed_urls.add(place_key)

                place = {
                    "name": place_name,
                    "type": "tourist_site",
                    "description": doc_text[:500] + "..." if len(doc_text) > 500 else doc_text,
                    "category": metadata.get("category", "general"),
                    "bestTimeToVisit": "Year-round",
                    "estimatedVisitDuration": "2 hours",
                    "location": {
                        "city": place_city,
                        "country": "Spain"
                    },
                    "semantic_distance": distance,
                    "chroma_doc_id": search_results["ids"][0][i],
                    "coordinates": self._safe_eval_coordinates(metadata.get("coordinates", "None"))
                }
                places_data.append(place)

            # Aplicar filtrado de duplicados también aquí
            unique_places = self._remove_duplicate_places(places_data)
            logger.info(f"Semantic search: {len(places_data)} -> {len(unique_places)} unique places")
            
            return unique_places
        except Exception as e:
            logger.error(f"Error performing semantic search: {e}")
            return []

    def _create_search_query_from_preferences(self, user_preferences: Dict) -> str:
        """Create a search query from user preferences using NLP processing with sentiment analysis."""
        query_parts = []
        
        if 'city' in user_preferences:
            query_parts.append(f"tourist attractions in {user_preferences['city']}")
            
        if 'category_interest' in user_preferences:
            high_interest_categories = [
                category for category, score in user_preferences['category_interest'].items()
                if score >= 4
            ]
            if high_interest_categories:
                query_parts.append(f"interested in {', '.join(high_interest_categories)}")
                
        if 'user_notes' in user_preferences and user_preferences['user_notes']:
            # Process user notes with NLP to extract relevant keywords and sentiment
            processed_notes = self.preprocess_text(user_preferences['user_notes'])
            keywords = self.extract_query_keywords(processed_notes, max_keywords=5)
            sentiment = self.analyze_query_sentiment(user_preferences['user_notes'])
            
            # Build query based on keywords and sentiment
            if keywords:
                keyword_query = ' '.join(keywords)
                
                # Enhance query based on sentiment
                if sentiment['sentiment'] == 'positive' and sentiment['confidence'] > 0.3:
                    # User is enthusiastic, boost positive descriptors
                    keyword_query += " amazing wonderful excellent"
                elif sentiment['sentiment'] == 'negative' and sentiment['confidence'] > 0.3:
                    # User has concerns, focus on quality and avoid negative aspects
                    keyword_query += " quality recommended peaceful"
                
                query_parts.append(keyword_query)
            else:
                query_parts.append(processed_notes)
                
        return ". ".join(query_parts)

    def _calculate_cosine_similarity(self, user_embedding: np.ndarray, place_embeddings: List[np.ndarray]) -> List[float]:
        """Calculate cosine similarity between user and place embeddings."""
        user_embedding_2d = user_embedding.reshape(1, -1)
        similarities = []
        for place_embedding in place_embeddings:
            place_embedding_2d = place_embedding.reshape(1, -1)
            similarity = cosine_similarity(user_embedding_2d, place_embedding_2d)[0][0]
            similarities.append(similarity)
        return similarities

    def _filter_places_by_similarity(self, places: List[Dict], place_embeddings: List[np.ndarray],
                                   user_embedding: np.ndarray, similarity_threshold: float = 0.1) -> Tuple[List[Dict], List[np.ndarray], List[float]]:
        """Filter places by cosine similarity."""
        similarities = self._calculate_cosine_similarity(user_embedding, place_embeddings)
        filtered_places = []
        filtered_embeddings = []
        filtered_similarities = []

        for place, embedding, similarity in zip(places, place_embeddings, similarities):
            if similarity >= similarity_threshold:
                filtered_places.append(place)
                filtered_embeddings.append(embedding)
                filtered_similarities.append(similarity)

        return filtered_places, filtered_embeddings, filtered_similarities

    def _call_gemini_llm(self, prompt: str, system: Optional[str] = None) -> str:
        """Call Gemini API for LLM response using the Gemini client."""
        try:
            response = self.gemini_client.generate(prompt, system)
            if response and response != "[Error de generación]" and response != "[Límite de solicitudes excedido]":
                return response
            else:
                raise RuntimeError(f"Gemini client failed to generate response: {response}")
        except Exception as e:
            logger.error(f"Error calling Gemini client: {e}")
            raise RuntimeError(f"Failed to call Gemini LLM: {e}")

    def _get_llm_recommendation_prompt(self, places: List[Dict], user_preferences: Dict) -> str:
        """Generate a prompt for the LLM to estimate time and classify categories."""
        if not places:
            return "No places found matching the criteria."

        prompt = f"""
Based on the following tourist preferences and available places, perform two tasks:
1. Estimate how much time (in hours) the tourist would be interested in spending at each location.
2. Classify each place into one of the following categories: engineering, history, food, culture, beach, shopping, nature, or general.

TOURIST PROFILE:
- Visiting: {user_preferences.get('city', 'Unknown city')}
- Available hours for tourism: {user_preferences.get('available_hours', 'Unknown')} hours
- Maximum travel distance: {user_preferences.get('max_distance', 'Unknown')} km
- Transportation modes: {', '.join(user_preferences.get('transport_modes', []))}

CATEGORY INTERESTS (1-5 scale):
"""
        category_interest = user_preferences.get('category_interest', {})
        for category, score in category_interest.items():
            prompt += f"- {category.capitalize()}: {score}/5\n"

        if user_preferences.get('user_notes'):
            prompt += f"\nADDITIONAL NOTES: {user_preferences['user_notes']}\n"

        prompt += "\nAVAILABLE PLACES:\n"
        for i, place in enumerate(places):
            prompt += f"""
{i+1}. {place['name']}
   - Type: {place.get('type', 'Unknown')}
   - Description: {place.get('description', 'No description')}
   - Category: {place.get('category', 'Unknown')}
   - Estimated Visit Duration: {place.get('estimatedVisitDuration', 'Unknown')}
"""

        prompt += """
TASK: For each place listed above, provide:
1. Estimated time (in hours) the tourist would be interested in spending there based on their preferences.
2. Category classification (choose one: engineering, history, food, culture, beach, shopping, nature, or general).

Consider:
- The tourist's category interests and ratings
- The type and appeal of each place
- The tourist's additional notes and preferences
- The suggested visit duration for each place

Provide your response in the following format:
Place 1: X.X hours - Brief reasoning | Category: <category> - Brief reasoning
Place 2: X.X hours - Brief reasoning | Category: <category> - Brief reasoning
...

Be realistic with time estimates and precise with category classification.
"""
        return prompt

    def _parse_llm_time_estimates(self, llm_response: str, places: List[Dict]) -> Tuple[List[float], List[str]]:
        """Parse LLM response to extract time estimates and categories."""
        time_estimates = []
        categories = []
        lines = llm_response.split('\n')

        for i, place in enumerate(places):
            found_time = None
            found_category = "general"

            for line in lines:
                if f"Place {i+1}:" in line or f"{i+1}." in line or place['name'] in line:
                    time_match = re.search(r'(\d+\.?\d*)\s*(?:hours?|h)', line, re.IGNORECASE)
                    if time_match:
                        found_time = float(time_match.group(1))
                    category_match = re.search(r'Category:\s*(\w+)', line, re.IGNORECASE)
                    if category_match:
                        found_category = category_match.group(1).lower()
                        if found_category not in ["engineering", "history", "food", "culture", "beach", "shopping", "nature"]:
                            found_category = "general"
                    break

            if found_time is None:
                estimated_duration = place.get('estimatedVisitDuration', '2 hours')
                duration_match = re.search(r'(\d+\.?\d*)', estimated_duration)
                found_time = float(duration_match.group(1)) if duration_match else 2.0

            time_estimates.append(found_time)
            categories.append(found_category)

        return time_estimates, categories

    def process_user_request(self, user_preferences: Dict, user_lat: float, user_lon: float, transport_mode: str) -> Dict[str, Any]:
        """Main RAG processing function."""
        search_query = self._create_search_query_from_preferences(user_preferences)
        semantic_places = self._semantic_search_chroma(search_query, n_results=100)  # Buscar más lugares

        city = user_preferences.get('city', '')
        if semantic_places and city:
            city_places = self._filter_places_by_city(semantic_places, city)
            if not city_places:
                city_places = self._filter_places_by_city(self.places_data, city)
        else:
            city_places = self._filter_places_by_city(self.places_data, city)

        if not city_places:
            raise RuntimeError(f"No places found for city: {city}")

        max_distance_km = user_preferences.get('max_distance', 10)
        distance_filtered_places, _ = self._filter_places_by_distance(
            city_places, max_distance_km, user_lat, user_lon
        )

        if not distance_filtered_places:
            raise RuntimeError(f"No places found within {max_distance_km}km.")

        # Generar embeddings para todos los lugares encontrados
        distance_place_embeddings = self._generate_place_embeddings(distance_filtered_places)
        user_embedding = self._generate_user_embedding(user_preferences)
        
        # Calcular similitudes para todos los lugares
        all_similarities = self._calculate_cosine_similarity(user_embedding, distance_place_embeddings)
        
        # En lugar de filtrar por similitud, usar todos los lugares encontrados
        # pero ordenarlos por similitud para dar prioridad a los más relevantes
        places_with_similarity = list(zip(distance_filtered_places, distance_place_embeddings, all_similarities))
        places_with_similarity.sort(key=lambda x: x[2], reverse=True)  # Ordenar por similitud descendente
        
        # Separar los datos ordenados
        filtered_places = [item[0] for item in places_with_similarity]
        place_embeddings = [item[1] for item in places_with_similarity]
        similarity_scores = [item[2] for item in places_with_similarity]
        
        logger.info(f"Using all {len(filtered_places)} places found (sorted by relevance)")

        # Calcular matriz de tiempos para todos los lugares
        time_matrix = self._calculate_time_matrix_ors(filtered_places, user_lat, user_lon, transport_mode)

        # Procesar con LLM todos los lugares encontrados
        llm_prompt = self._get_llm_recommendation_prompt(filtered_places, user_preferences)
        llm_response = self._call_gemini_llm(llm_prompt)
        llm_time_estimates, llm_categories = self._parse_llm_time_estimates(llm_response, filtered_places)

        # Actualizar categorías
        for place, category in zip(filtered_places, llm_categories):
            place['category'] = category

        return {
            'filtered_places': filtered_places,
            'time_matrix': time_matrix,
            'place_embeddings': place_embeddings,
            'user_embedding': user_embedding,
            'llm_time_estimates': llm_time_estimates,
            'llm_response': llm_response,
            'similarity_scores': similarity_scores,
            'user_preferences': user_preferences,
            'transport_mode': transport_mode,
            'data_source': "ChromaDB semantic search (all unique places)",
            'llm_categories': llm_categories
        }
