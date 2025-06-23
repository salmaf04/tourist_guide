import numpy as np
import requests
import re
import time
from typing import Dict, List, Tuple, Any, Optional, Set
from sentence_transformers import SentenceTransformer
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
from collections import Counter, defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import logging
import os
from dotenv import load_dotenv
import chromadb
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import wordnet
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk

from pysentimiento import create_analyzer
from keybert import KeyBERT

from agent_generator.mistral_client import MistralClient 

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGPlanner:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", chroma_db_path: str = None):
        """
        Initialize the RAG Planner with NLP capabilities
        """
        self.model = SentenceTransformer(model_name)
        self.keybert_model = KeyBERT()
        self.sentiment_analyzer = create_analyzer(task="sentiment", lang="es")
        
        # Set default ChromaDB path if none provided
        if chroma_db_path is None:
            # Get the absolute path to the project root
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            self.chroma_db_path = os.path.join(project_root, "src", "crawler", "db")
        else:
            self.chroma_db_path = chroma_db_path

        # Initialize ChromaDB
        self._init_chromadb()
        self.places_data = self._load_places_from_chroma()

        # Initialize geocoder
        self.geocoder = Nominatim(user_agent="tourist_guide_app")
        self.coordinate_cache = {}

        # Initialize Mistral client
        self.mistral_client = MistralClient()

        # Get API keys
        self.openrouteservice_api_key = os.getenv('OPENROUTESERVICE_API_KEY')

        # Initialize NLP components
        self._init_nlp_components()

        # Validate API keys
        if not self.openrouteservice_api_key:
            raise ValueError("OPENROUTESERVICE_API_KEY not found in environment variables.")

        logger.info("RAG Planner initialized with Mistral client, API keys, and NLP components successfully.")

    def _init_nlp_components(self):
        """Initialize essential NLP components for query processing."""
        try:
            # Download required NLTK data
            self._download_nltk_data()
            
            # Get stopwords using NLTK
            self.spanish_stopwords = self._get_nltk_stopwords()

            # Initialize NLTK stemmer for Spanish
            self.stemmer = SnowballStemmer('spanish')
            
            # pysentimiento analyzer is already initialized in the constructor.
            
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
        nltk_downloads = ['stopwords', 'punkt', 'vader_lexicon', 'wordnet', 'averaged_perceptron_tagger']
        
        for package in nltk_downloads:
            try:
                if package == 'vader_lexicon':
                    nltk.data.find('vader_lexicon')
                elif package in ['stopwords', 'punkt']:
                    nltk.data.find(f'tokenizers/{package}')
                else:
                    nltk.data.find(f'corpora/{package}')
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
    def _extract_aspects(self, text: str, n_aspects: int = 3) -> List[Tuple[str, float]]:
        """Extrae los aspectos clave de un texto con sus puntuaciones"""
        try:
            aspects = self.keybert_model.extract_keywords(
                text,
                keyphrase_ngram_range=(1, 2),
                top_n=n_aspects,
                stop_words=list(self.spanish_stopwords))
            return aspects
        except Exception as e:
            logger.error(f"Error extracting aspects: {e}")
            return []

    def _analyze_aspect_sentiment(self, aspect: str, context: str) -> Dict[str, float]:
        """Analiza el sentimiento hacia un aspecto específico en un contexto usando el método estándar"""
        # Usar el método estándar que ya tiene validaciones y manejo de errores
        combined_text = f"{aspect} : {context}"
        sentiment_result = self.analyze_query_sentiment(combined_text)
        
        # Convertir al formato esperado por el método que lo llama
        return {
            'sentiment': sentiment_result['sentiment'].upper()[:3],  # POS, NEG, NEU
            'probas': {
                'POS': sentiment_result['scores']['pos'],
                'NEG': sentiment_result['scores']['neg'],
                'NEU': sentiment_result['scores']['neu']
            },
            'score': sentiment_result['scores']['compound']
        }

    def _calculate_category_penalty(self, user_prefs: Dict, place: Dict) -> float:
        """
        Calcula una penalización basada en las categorías que el usuario no le gustan
        """
        if 'category_interest' not in user_prefs:
            return 1.0
        
        place_category = place.get('category', '').lower()
        if not place_category:
            return 1.0
        
        # Mapear categorías de lugares a categorías de preferencias
        category_mapping = {
            'history': ['history', 'sitio histórico', 'histórico', 'museo', 'catedral', 'iglesia', 'castillo'],
            'religion': ['religion', 'religioso', 'catedral', 'iglesia', 'basílica', 'monasterio', 'convento'],
            'culture': ['culture', 'cultural', 'museo', 'catedral', 'iglesia', 'arte', 'teatro'],
            'urban': ['urban', 'urbano', 'ciudad', 'plaza', 'calle', 'barrio', 'centro'],
            'nature': ['nature', 'natural', 'sitio natural', 'parque', 'meadows', 'jardín', 'bosque'],
            'beach': ['beach', 'playa', 'sea', 'beaches', 'costa', 'mar'],
            'food': ['food', 'mercado', 'market', 'restaurante', 'gastronomía', 'comida'],
            'shopping': ['shopping', 'mercado', 'market', 'tienda', 'comercio', 'compras'],
            'entertainment': ['entertainment', 'entretenimiento', 'diversión', 'espectáculo', 'ocio']
        }
        
        # Encontrar qué categoría de preferencia corresponde al lugar
        place_pref_category = None
        for pref_cat, keywords in category_mapping.items():
            if any(keyword in place_category for keyword in keywords):
                place_pref_category = pref_cat
                break
        
        if place_pref_category and place_pref_category in user_prefs['category_interest']:
            score = user_prefs['category_interest'][place_pref_category]
            
            # Convertir puntuación a multiplicador
            if score == 5:
                return 1.5  # Boost para categorías favoritas
            elif score == 4:
                return 1.2
            elif score == 3:
                return 1.0  # Neutral
            elif score == 2:
                return 0.5  # Penalización moderada
            elif score == 1:
                return 0.1  # Penalización fuerte
        
        return 1.0  # Sin penalización si no hay mapeo

    def _calculate_enhanced_similarity(self, user_prefs: Dict, place: Dict) -> float:
        """
        Calcula una puntuación mejorada que combina:
        - Similitud semántica de aspectos clave
        - Alineamiento de sentimiento
        """
        # Extraer aspectos del lugar
        place_aspects = self._extract_aspects(place['description'])
        if not place_aspects:
            return 0.0
        
        # Extraer aspectos de las preferencias del usuario con mayor énfasis en preferencias negativas
        category_texts = []
        if 'category_interest' in user_prefs:
            for category, score in user_prefs.get('category_interest', {}).items():
                if score == 5:
                    category_texts.append(f"me encanta {category}")
                    category_texts.append(f"busco {category}")
                elif score == 4:
                    category_texts.append(f"me gusta mucho {category}")
                elif score == 3:
                    category_texts.append(f"me interesa {category}")
                elif score == 2:
                    category_texts.append(f"no me interesa mucho {category}")
                elif score == 1:
                    category_texts.append(f"odio {category}")
                    category_texts.append(f"evito {category}")
        
        user_text = f"{user_prefs.get('user_notes', '')} {' '.join(category_texts)}"
        user_aspects = self._extract_aspects(user_text)
        
        # Calcular similitud semántica entre aspectos
        semantic_sim = self._aspect_similarity(user_aspects, place_aspects)
        
        # Calcular alineamiento de sentimiento
        sentiment_alignment = self._sentiment_alignment(user_prefs, place, place_aspects)
        
        # Calcular penalización por categoría
        category_penalty = self._calculate_category_penalty(user_prefs, place)
        
        # Combinar con pesos y aplicar penalización
        base_similarity = 0.6 * semantic_sim + 0.4 * sentiment_alignment
        final_similarity = base_similarity * category_penalty
        
        return max(0.0, min(1.0, final_similarity))  # Asegurar rango [0, 1]

    def _aspect_similarity(self, user_aspects: List[Tuple[str, float]], 
                          place_aspects: List[Tuple[str, float]]) -> float:
        """Calcula la similitud semántica entre aspectos"""
        if not user_aspects or not place_aspects:
            return 0.0
        
        # Extraer solo los términos de los aspectos
        user_terms = [aspect[0] for aspect in user_aspects]
        place_terms = [aspect[0] for aspect in place_aspects]
        
        # Generar embeddings
        user_embs = self.model.encode(user_terms)
        place_embs = self.model.encode(place_terms)
        
        # Matriz de similitud
        sim_matrix = np.dot(user_embs, place_embs.T)
        
        # Promedio de máximas similitudes (soft alignment)
        max_sim_user = np.max(sim_matrix, axis=1).mean()
        max_sim_place = np.max(sim_matrix, axis=0).mean()
        
        return (max_sim_user + max_sim_place) / 2

    def _sentiment_alignment(self, user_prefs: Dict, place: Dict, 
                            place_aspects: List[Tuple[str, float]]) -> float:
        """Calcula el alineamiento de sentimiento entre usuario y lugar"""
        # Analizar sentimiento del usuario
        user_sentiment = self.analyze_query_sentiment(user_prefs.get('user_notes', ''))
        user_score = user_sentiment['scores']['pos'] - user_sentiment['scores']['neg']
        
        # Analizar sentimiento del lugar (promedio ponderado por aspectos)
        place_sentiments = []
        for aspect, weight in place_aspects:
            aspect_sentiment = self._analyze_aspect_sentiment(aspect, place['description'])
            place_sentiments.append(aspect_sentiment['score'] * weight)
        
        if not place_sentiments:
            return 0.0
            
        place_score = np.mean(place_sentiments)
        
        # Calcular alineamiento (1 - distancia)
        return 1 - abs(user_score - place_score)

    def _generate_place_embeddings(self, places: List[Dict]) -> List[np.ndarray]:
        """Genera embeddings enriquecidos con información de aspectos y sentimiento"""
        enriched_embeddings = []
        for place in places:
            # Extraer aspectos clave
            aspects = self._extract_aspects(place['description'])
            aspects_text = " ".join([aspect[0] for aspect in aspects]) if aspects else ""
            
            # Texto enriquecido para el embedding
            text_representation = f"""
Nombre: {place['name']}
Aspectos clave: {aspects_text}
Descripción: {place.get('description', '')}
Categoría: {place.get('category', '')}
            """.strip()
            
            # Generar embedding semántico
            semantic_embedding = self.model.encode(text_representation)
            
            # Usar el método optimizado para obtener directamente el vector de sentimiento
            sentiment_vector = self._get_sentiment_vector(place.get('description', ''))
            
            # Combinar embedding semántico y de sentimiento
            combined_embedding = np.concatenate([semantic_embedding, sentiment_vector])
            enriched_embeddings.append(combined_embedding)
            
        return enriched_embeddings
        
    def _generate_user_embedding(self, user_preferences: Dict) -> np.ndarray:
        """Generate embedding with sentiment enhancement for user query"""
        preferences_text = []
        
        # Procesar preferencias de categoría con sentimiento explícito y pesos
        if 'category_interest' in user_preferences:
            for category, score in user_preferences['category_interest'].items():
                if score == 5:
                    preferences_text.append(f"Me encanta completamente {category}")
                    # Repetir para dar más peso a las preferencias altas
                    preferences_text.append(f"Busco especialmente {category}")
                    preferences_text.append(f"Prefiero {category}")
                elif score == 4:
                    preferences_text.append(f"Realmente me gusta {category}")
                    preferences_text.append(f"Me interesa mucho {category}")
                elif score == 3:
                    preferences_text.append(f"Tengo interés moderado en {category}")
                elif score == 2:
                    preferences_text.append(f"No me interesa mucho {category}")
                    preferences_text.append(f"Evito un poco {category}")
                elif score == 1:
                    preferences_text.append(f"Odio {category}")
                    preferences_text.append(f"Evito completamente {category}")
                    preferences_text.append(f"No quiero {category}")
        
        # Procesar notas del usuario con análisis de sentimiento
        user_notes_text = ""
        if 'user_notes' in user_preferences and user_preferences['user_notes']:
            notes = user_preferences['user_notes']
            preferences_text.append(notes)
            # Dar más peso a las notas del usuario repitiéndolas
            preferences_text.append(notes)
            user_notes_text = notes
        
        # Si no hay texto de preferencias, crear uno más específico basado en las categorías
        if not preferences_text:
            # Crear texto específico basado en las puntuaciones de categorías
            if 'category_interest' in user_preferences:
                high_cats = [cat for cat, score in user_preferences['category_interest'].items() if score >= 4]
                low_cats = [cat for cat, score in user_preferences['category_interest'].items() if score <= 2]
                
                if high_cats and low_cats:
                    preferences_text.append(f"Busco específicamente {' y '.join(high_cats)}")
                    preferences_text.append(f"Evito completamente {' y '.join(low_cats)}")
                elif high_cats:
                    preferences_text.append(f"Solo me interesan {' y '.join(high_cats)}")
                elif low_cats:
                    preferences_text.append(f"No quiero {' y '.join(low_cats)}")
                else:
                    preferences_text.append("Busco lugares turísticos interesantes")
            else:
                preferences_text.append("Busco lugares turísticos interesantes")
        
        # Generar embedding semántico
        full_text = ". ".join(preferences_text)
        logger.info(f"🔍 User preference text for embedding: {full_text[:200]}...")
        semantic_embedding = self.model.encode(full_text)
        
        # Usar el método optimizado para obtener el vector de sentimiento
        # Si hay notas del usuario, usar esas; si no, usar todo el texto de preferencias
        sentiment_text = user_notes_text if user_notes_text else full_text
        sentiment_vector = self._get_sentiment_vector(sentiment_text)
        
        logger.info(f"📊 User embedding dimensions: semantic={len(semantic_embedding)}, sentiment={len(sentiment_vector)}")
        return np.concatenate([semantic_embedding, sentiment_vector])

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
        return filtered_places

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
            
            return places_data
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

    def _calculate_cosine_similarity(self, user_embedding: np.ndarray, place_embeddings: List[np.ndarray], 
                                   user_preferences: Dict = None, places: List[Dict] = None) -> List[float]:
        """Enhanced cosine similarity with sentiment adjustment and category penalties"""
        similarities = []
        
        logger.info(f"🔍 Calculating similarity for {len(place_embeddings)} places")
        logger.info(f"📊 User embedding shape: {user_embedding.shape}")
        
        # Verificar que el embedding del usuario tenga el tamaño correcto
        if len(user_embedding) < 4:
            logger.error(f"❌ User embedding too small: {len(user_embedding)} dimensions")
            # Fallback: usar solo similitud semántica básica
            for place_emb in place_embeddings:
                if len(place_emb) >= len(user_embedding):
                    semantic_sim = cosine_similarity(
                        user_embedding.reshape(1, -1),
                        place_emb[:len(user_embedding)].reshape(1, -1)
                    )[0][0]
                    similarities.append(max(0.0, semantic_sim))
                else:
                    similarities.append(0.0)
            return similarities
        
        # Separar componentes del embedding del usuario
        user_semantic = user_embedding[:-4]  # Primeras N-4 dimensiones (MiniLM)
        user_sentiment = user_embedding[-4:]  # Últimas 4 dimensiones (sentimiento)
        
        logger.info(f"📊 User semantic dimensions: {len(user_semantic)}, sentiment: {len(user_sentiment)}")
        
        for i, place_emb in enumerate(place_embeddings):
            try:
                # Verificar que el embedding del lugar tenga el tamaño correcto
                if len(place_emb) < 4:
                    logger.warning(f"⚠️ Place {i} embedding too small: {len(place_emb)} dimensions")
                    similarities.append(0.0)
                    continue
                
                # Separar componentes del embedding del lugar
                place_semantic = place_emb[:-4]
                place_sentiment = place_emb[-4:]
                
                # Verificar que las dimensiones semánticas coincidan
                if len(user_semantic) != len(place_semantic):
                    logger.warning(f"⚠️ Dimension mismatch for place {i}: user={len(user_semantic)}, place={len(place_semantic)}")
                    # Usar la dimensión menor
                    min_dim = min(len(user_semantic), len(place_semantic))
                    user_sem_truncated = user_semantic[:min_dim]
                    place_sem_truncated = place_semantic[:min_dim]
                else:
                    user_sem_truncated = user_semantic
                    place_sem_truncated = place_semantic
                
                # 1. Calcular similitud semántica tradicional
                semantic_sim = cosine_similarity(
                    user_sem_truncated.reshape(1, -1),
                    place_sem_truncated.reshape(1, -1)
                )[0][0]
                
                # Asegurar que la similitud esté en rango válido
                semantic_sim = max(-1.0, min(1.0, semantic_sim))
                
                # 2. Calcular alineamiento de sentimiento
                user_pos, user_neg = user_sentiment[0], user_sentiment[1]
                place_pos, place_neg = place_sentiment[0], place_sentiment[1]

                # Premiar la coincidencia de sentimientos (positivo con positivo, negativo con negativo)
                sentiment_similarity = (user_pos * place_pos) + (user_neg * place_neg)
                
                # Penalizar la discrepancia de sentimientos (positivo con negativo)
                sentiment_opposition = (user_pos * place_neg) + (user_neg * place_pos)
                
                # El alineamiento va de -1 (totalmente opuesto) a 1 (totalmente alineado)
                alignment = sentiment_similarity - sentiment_opposition
                
                # Normalizar a rango [0, 1]
                sentiment_alignment = (alignment + 1) / 2
                sentiment_alignment = max(0.0, min(1.0, sentiment_alignment))
                
                # 3. Combinar con peso balanceado (70% semántica, 30% sentimiento)
                # Si el sentimiento es muy bajo, usar más peso semántico
                if sentiment_alignment < 0.3:
                    combined_sim = 0.9 * semantic_sim + 0.1 * sentiment_alignment
                else:
                    combined_sim = 0.7 * semantic_sim + 0.3 * sentiment_alignment
                
                # Aplicar penalización por categoría si está disponible
                if user_preferences and places and i < len(places):
                    category_penalty = self._calculate_category_penalty(user_preferences, places[i])
                    combined_sim = combined_sim * category_penalty
                
                # Asegurar que el resultado esté en rango válido
                combined_sim = max(0.0, min(1.0, combined_sim))
                similarities.append(combined_sim)
                
            except Exception as e:
                logger.error(f"❌ Error calculating similarity for place {i}: {e}")
                similarities.append(0.0)
        
        logger.info(f"✅ Calculated {len(similarities)} similarities, range: {min(similarities):.3f} - {max(similarities):.3f}")
        return similarities


    def _filter_places_by_similarity(self, places: List[Dict], place_embeddings: List[np.ndarray],
                                   user_embedding: np.ndarray, user_preferences: Dict = None, 
                                   similarity_threshold: float = 0.1) -> Tuple[List[Dict], List[np.ndarray], List[float]]:
        """Filter places by enhanced cosine similarity"""
        similarities = self._calculate_cosine_similarity(user_embedding, place_embeddings, user_preferences, places)
        
        # Ordenar por similitud descendente
        sorted_data = sorted(zip(places, place_embeddings, similarities), 
                           key=lambda x: x[2], reverse=True)
        
        # Filtrar por umbral y desempaquetar
        filtered_data = [(p, e, s) for p, e, s in sorted_data if s >= similarity_threshold]
        
        if filtered_data:
            filtered_places, filtered_embeddings, filtered_similarities = zip(*filtered_data)
            return list(filtered_places), list(filtered_embeddings), list(filtered_similarities)
        return [], [], []

    def _call_mistral_llm(self, prompt: str, system: Optional[str] = None) -> str:
        """Call Mistral API for LLM response using the Mistral client."""
        try:
            logger.info("🤖 Calling Mistral API...")
            response = self.mistral_client.generate(prompt, system)
            logger.info(f"🤖 Mistral API response received: {len(response) if response else 0} characters")
            
            if response and response != "[Error de generación]" and response != "[Límite de solicitudes excedido]":
                logger.info("✅ Mistral API call successful")
                return response
            else:
                logger.error(f"❌ Mistral client failed to generate response: {response}")
                raise RuntimeError(f"Mistral client failed to generate response: {response}")
        except Exception as e:
            logger.error(f"❌ Error calling Mistral client: {e}")
            raise RuntimeError(f"Failed to call Mistral LLM: {e}")

    def _get_llm_recommendation_prompt(self, places: List[Dict], user_preferences: Dict) -> str:
        """Generate a prompt for the LLM to estimate time only (categories come from database)."""
        if not places:
            return "No places found matching the criteria."

        prompt = f"""
Based on the following tourist preferences and available places, estimate how much time (in hours) the tourist would be interested in spending at each location.

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
TASK: For each place listed above, provide only the estimated time (in hours) the tourist would be interested in spending there based on their preferences.

Consider:
- The tourist's category interests and ratings
- The type and appeal of each place
- The tourist's additional notes and preferences
- The suggested visit duration for each place

Provide your response in the following format:
Place 1: X.X hours - Brief reasoning
Place 2: X.X hours - Brief reasoning
...

Be realistic with time estimates based on the tourist's interests and the nature of each place.
"""
        return prompt

    def _parse_llm_time_estimates(self, llm_response: str, places: List[Dict]) -> List[float]:
        """Parse LLM response to extract time estimates only (categories come from database)."""
        time_estimates = []
        lines = llm_response.split('\n')

        for i, place in enumerate(places):
            found_time = None

            for line in lines:
                if f"Place {i+1}:" in line or f"{i+1}." in line or place['name'] in line:
                    time_match = re.search(r'(\d+\.?\d*)\s*(?:hours?|h)', line, re.IGNORECASE)
                    if time_match:
                        found_time = float(time_match.group(1))
                    break

            if found_time is None:
                estimated_duration = place.get('estimatedVisitDuration', '2 hours')
                duration_match = re.search(r'(\d+\.?\d*)', estimated_duration)
                found_time = float(duration_match.group(1)) if duration_match else 2.0

            time_estimates.append(found_time)

        return time_estimates

    def process_user_request(self, user_preferences: Dict, user_lat: float, user_lon: float, transport_mode: str) -> Dict[str, Any]:
        """Main RAG processing function."""
        search_query = self._create_search_query_from_preferences(user_preferences)
        logger.info(f"🔍 Search query: {search_query}")
        
        semantic_places = self._semantic_search_chroma(search_query, n_results=100)  # Buscar más lugares
        logger.info(f"📊 Semantic search found {len(semantic_places)} places")

        city = user_preferences.get('city', '')
        logger.info(f"🏙️ Target city: {city}")
        
        if semantic_places and city:
            city_places = self._filter_places_by_city(semantic_places, city)
            logger.info(f"📍 City filtering from semantic search: {len(city_places)} places")
            if not city_places:
                city_places = self._filter_places_by_city(self.places_data, city)
                logger.info(f"📍 City filtering from all data: {len(city_places)} places")
        else:
            city_places = self._filter_places_by_city(self.places_data, city)
            logger.info(f"📍 City filtering from all data: {len(city_places)} places")

        if not city_places:
            logger.error(f"❌ No places found for city: {city}")
            # Let's check what cities are available
            available_cities = set()
            for place in self.places_data:
                available_cities.add(place.get('location', {}).get('city', 'Unknown'))
            logger.info(f"🏙️ Available cities in database: {sorted(list(available_cities))}")
            raise RuntimeError(f"No places found for city: {city}. Available cities: {sorted(list(available_cities))}")

        max_distance_km = user_preferences.get('max_distance', 10)
        logger.info(f"📏 Filtering by distance: {max_distance_km}km from ({user_lat}, {user_lon})")
        
        # Log some places before distance filtering
        logger.info(f"📋 Places found for {city} before distance filtering:")
        for i, place in enumerate(city_places[:5]):  # Show first 5 places
            place_lat, place_lon = self._get_coordinates_from_place(place)
            logger.info(f"  {i+1}. {place['name']} - Coords: {place_lat}, {place_lon}")
        
        distance_filtered_places, _ = self._filter_places_by_distance(
            city_places, max_distance_km, user_lat, user_lon
        )
        logger.info(f"📍 Distance filtering result: {len(distance_filtered_places)} places within {max_distance_km}km")

        if not distance_filtered_places:
            logger.error(f"❌ No places found within {max_distance_km}km")
            # Log detailed distance info for debugging
            logger.info(f"🔍 Distance analysis for {city} places:")
            for i, place in enumerate(city_places[:10]):  # Show first 10 places
                place_lat, place_lon = self._get_coordinates_from_place(place)
                if place_lat is not None and place_lon is not None:
                    distance_km = geodesic((user_lat, user_lon), (place_lat, place_lon)).kilometers
                    logger.info(f"  {i+1}. {place['name']} - {distance_km:.2f}km away")
                else:
                    logger.info(f"  {i+1}. {place['name']} - No coordinates available")
            
            # Try with a larger distance to see if there are any places at all
            logger.info(f"🔍 Trying with 50km radius...")
            extended_places, _ = self._filter_places_by_distance(
                city_places, 50, user_lat, user_lon
            )
            logger.info(f"📍 Places within 50km: {len(extended_places)}")
            
            raise RuntimeError(f"No places found within {max_distance_km}km.")
        
        enhanced_scores = []
        for place in distance_filtered_places:
            score = self._calculate_enhanced_similarity(user_preferences, place)
            enhanced_scores.append(score)
        
        # Ordenar lugares por puntuación mejorada
        sorted_places = sorted(zip(distance_filtered_places, enhanced_scores), 
                             key=lambda x: x[1], reverse=True)

        # Generate embeddings for all distance-filtered places
        logger.info(f"🔄 Generating embeddings for {len(distance_filtered_places)} places...")
        distance_place_embeddings = self._generate_place_embeddings(distance_filtered_places)
        logger.info(f"✅ Generated {len(distance_place_embeddings)} place embeddings")
        
        logger.info(f"🔄 Generating user embedding from preferences...")
        user_embedding = self._generate_user_embedding(user_preferences)
        logger.info(f"✅ Generated user embedding with {len(user_embedding)} dimensions")
        
        # Calculate similarities for all places
        logger.info(f"🔄 Calculating similarities between user and places...")
        all_similarities = self._calculate_cosine_similarity(user_embedding, distance_place_embeddings, 
                                                           user_preferences, distance_filtered_places)
        logger.info(f"✅ Calculated {len(all_similarities)} similarity scores")
        
        # Use all places but order them by similarity to prioritize the most relevant ones
        places_with_similarity = list(zip(distance_filtered_places, distance_place_embeddings, all_similarities))
        places_with_similarity.sort(key=lambda x: x[2], reverse=True)  # Sort by similarity descending
        
        # Imprimir lugares ordenados por similitud en la terminal
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
            
            # Calcular penalización para mostrar información adicional
            category_penalty = self._calculate_category_penalty(user_preferences, place)
            
            # Agregar indicador visual para similitudes altas
            if similarity >= 0.7:
                indicator = "🔥"
            elif similarity >= 0.5:
                indicator = "⭐"
            elif similarity >= 0.3:
                indicator = "✅"
            else:
                indicator = "📍"
            
            # Agregar indicador de penalización
            if category_penalty >= 1.2:
                penalty_indicator = "💚"  # Boost
            elif category_penalty == 1.0:
                penalty_indicator = "⚪"  # Neutral
            elif category_penalty >= 0.5:
                penalty_indicator = "🟡"  # Penalización moderada
            else:
                penalty_indicator = "🔴"  # Penalización fuerte
            
            print(f"{i:2d}. {similarity:.4f} {indicator} | {place_name} ({place_category}) {penalty_indicator} x{category_penalty:.1f}")
        
        print("="*80)
        print(f"📊 Total de lugares ordenados: {len(places_with_similarity)}")
        print(f"📊 Rango de similitud: {min(all_similarities):.4f} - {max(all_similarities):.4f}")
        if len(places_with_similarity) > 0:
            avg_similarity = sum(all_similarities) / len(all_similarities)
            print(f"📊 Similitud promedio: {avg_similarity:.4f}")
        print("="*80 + "\n")
        
        # Separate the sorted data
        sorted_places = [item[0] for item in places_with_similarity]
        place_embeddings = [item[1] for item in places_with_similarity]
        similarity_scores = [item[2] for item in places_with_similarity]
        
        # Use the sorted places instead of the enhanced similarity filtered places
        # This ensures consistency between places, embeddings, and similarity scores
        final_places = sorted_places
        final_scores = similarity_scores
        
        logger.info(f"📊 Using {len(final_places)} places sorted by cosine similarity")
        logger.info(f"📊 Similarity range: {min(similarity_scores):.3f} - {max(similarity_scores):.3f}")

        # Calcular matriz de tiempos para todos los lugares
        time_matrix = self._calculate_time_matrix_ors(final_places, user_lat, user_lon, transport_mode)

        # Procesar con LLM todos los lugares encontrados (solo para estimaciones de tiempo)
        llm_prompt = self._get_llm_recommendation_prompt(final_places, user_preferences)
        llm_response = self._call_mistral_llm(llm_prompt)
        llm_time_estimates = self._parse_llm_time_estimates(llm_response, final_places)

       # Extraer categorías de la base de datos (no del LLM)
        db_categories = [place.get('category', 'general') for place in final_places]

        return {
            'filtered_places': final_places,
            'enhanced_scores': final_scores,
            'time_matrix': time_matrix,
            'place_embeddings': place_embeddings,
            'user_embedding': user_embedding,
            'llm_time_estimates': llm_time_estimates,
            'llm_response': llm_response,
            'similarity_scores': similarity_scores,
            'user_preferences': user_preferences,
            'transport_mode': transport_mode,
            'db_categories': db_categories ,
            'data_source': "ChromaDB semantic search with enhanced cosine similarity ranking"
        }

    def preprocess_text(self, text: str) -> str:
        """Preprocess text using advanced NLTK techniques."""
        if not text:
            return ""
        
        try:
            # Tokenize with NLTK
            tokens = word_tokenize(text, language='spanish')
            
            # POS tagging to identify meaningful words
            pos_tags = pos_tag(tokens)
            
            # Keep only meaningful parts of speech and filter stopwords
            processed_tokens = []
            for token, pos in pos_tags:
                # Keep nouns, adjectives, verbs, and proper nouns
                if (pos.startswith(('NN', 'JJ', 'VB', 'NNP')) and 
                    token.isalpha() and 
                    len(token) > 2 and
                    token.lower() not in self.spanish_stopwords):
                    
                    # Apply stemming for better matching
                    if self.stemmer:
                        stemmed = self.stemmer.stem(token.lower())
                        processed_tokens.append(stemmed)
                    else:
                        processed_tokens.append(token.lower())
            
            # Join tokens back into text
            processed_text = ' '.join(processed_tokens)
            
            # If processing resulted in empty text, return original lowercased
            return processed_text if processed_text.strip() else text.lower()
            
        except Exception as e:
            logger.error(f"Error preprocessing text with NLTK: {e}")
            # Fallback to simple preprocessing
            try:
                tokens = word_tokenize(text, language='spanish')
                filtered_tokens = [
                    token.lower() for token in tokens 
                    if token.isalpha() and 
                    len(token) > 2 and 
                    token.lower() not in self.spanish_stopwords
                ]
                return ' '.join(filtered_tokens) if filtered_tokens else text.lower()
            except:
                return text.lower()

    def extract_query_keywords(self, text: str, max_keywords: int = 5) -> List[str]:
        """Extract relevant keywords using NLTK POS tagging and NER."""
        if not text:
            return []
        
        try:
            # Tokenize and POS tag
            tokens = word_tokenize(text, language='spanish')
            pos_tags = pos_tag(tokens)
            
            # Extract nouns, adjectives, and proper nouns
            keywords = []
            for word, pos in pos_tags:
                if (pos.startswith('NN') or pos.startswith('JJ') or pos.startswith('NNP')) and \
                   len(word) > 3 and word.lower() not in self.spanish_stopwords:
                    
                    # Check if word is tourism-related or significant
                    is_tourism_related = any(
                        word.lower() in category_words 
                        for category_words in self.tourism_keywords.values()
                    )
                    
                    if is_tourism_related or len(word) > 5:
                        keywords.append(word.lower())
            
            # Remove duplicates and limit
            unique_keywords = list(dict.fromkeys(keywords))
            return unique_keywords[:max_keywords]
            
        except Exception as e:
            logger.error(f"Error extracting keywords with NLTK: {e}")
            # Fallback to simple method
            words = text.split()
            return [w.lower() for w in words if len(w) > 4 and w.lower() not in self.spanish_stopwords][:max_keywords]

    def _get_sentiment_vector(self, text: str) -> np.ndarray:
        """
        Método optimizado para obtener solo el vector de sentimiento (4 dimensiones)
        sin toda la información adicional cuando solo necesitamos el vector.
        """
        if not text or not isinstance(text, str) or not text.strip():
            return np.array([0.0, 0.0, 1.0, 0.0])  # neutral por defecto
        
        try:
            analysis = self.sentiment_analyzer.predict(text)
            probas = analysis.probas
            
            return np.array([
                probas.get('POS', 0.0),
                probas.get('NEG', 0.0),
                probas.get('NEU', 1.0),
                probas.get('POS', 0.0) - probas.get('NEG', 0.0)  # compound
            ])
        except Exception as e:
            logger.error(f"Error in sentiment vector calculation: {e}")
            return np.array([0.0, 0.0, 1.0, 0.0])

    def analyze_query_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment using pysentimiento for Spanish.
        """
        if not text or not isinstance(text, str) or not text.strip():
            return {
                'sentiment': 'neutral', 
                'confidence': 1.0, 
                'scores': {'pos': 0.0, 'neg': 0.0, 'neu': 1.0, 'compound': 0.0}
            }

        try:
            # Use pysentimiento analyzer
            analysis = self.sentiment_analyzer.predict(text)
            
            # Map sentiment to a standard format
            sentiment_map = {'POS': 'positive', 'NEG': 'negative', 'NEU': 'neutral'}
            sentiment = sentiment_map.get(analysis.output, 'neutral')
            
            # Get probabilities
            probas = analysis.probas
            confidence = probas.get(analysis.output, 0.0)
            
            # Create a VADER-like score structure for consistency
            scores = {
                'pos': probas.get('POS', 0.0),
                'neg': probas.get('NEG', 0.0),
                'neu': probas.get('NEU', 0.0),
                # Compound score: weighted difference between positive and negative
                'compound': probas.get('POS', 0.0) - probas.get('NEG', 0.0)
            }
            
            return {
                'sentiment': sentiment,
                'confidence': confidence,
                'scores': scores
            }
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment with pysentimiento: {e}")
            return {
                'sentiment': 'neutral', 
                'confidence': 1.0, 
                'scores': {'pos': 0.0, 'neg': 0.0, 'neu': 1.0, 'compound': 0.0}
            }