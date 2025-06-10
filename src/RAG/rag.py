import json
import numpy as np
import requests
import re
from typing import Dict, List, Tuple, Any, Optional
from sentence_transformers import SentenceTransformer
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
from sklearn.metrics.pairwise import cosine_similarity
import logging
import os
from dotenv import load_dotenv
import chromadb

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGPlanner:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", chroma_db_path: str = "db/"):
        """
        Initialize the RAG Planner
        
        Args:
            model_name: Name of the sentence transformer model for embeddings
            chroma_db_path: Path to ChromaDB database
        
        Raises:
            ValueError: If required API keys are not found in environment variables
            RuntimeError: If ChromaDB is not available or has no data
        """
        self.model = SentenceTransformer(model_name)
        self.chroma_db_path = chroma_db_path
        
        # Initialize ChromaDB (required)
        self._init_chromadb()
        self.places_data = self._load_places_from_chroma()
        
        # Initialize geocoder for coordinate lookup
        self.geocoder = Nominatim(user_agent="tourist_guide_app")
        self.coordinate_cache = {}  # Cache for geocoding results
        
        # Get API keys from environment variables
        self.openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
        self.openrouteservice_api_key = os.getenv('OPENROUTESERVICE_API_KEY')
        
        # Validate required API keys
        if not self.openrouter_api_key:
            raise ValueError(
                "OPENROUTER_API_KEY not found in environment variables. "
                "Please set it in your .env file or environment."
            )
        
        if not self.openrouteservice_api_key:
            raise ValueError(
                "OPENROUTESERVICE_API_KEY not found in environment variables. "
                "Please set it in your .env file or environment."
            )
        
        logger.info("RAG Planner initialized with API keys successfully. Using ChromaDB as data source.")
    
    def _init_chromadb(self):
        """Initialize ChromaDB client and collection"""
        try:
            self.chroma_client = chromadb.PersistentClient(path=self.chroma_db_path)
            self.chroma_collection = self.chroma_client.get_or_create_collection(name="tourism_docs")
            logger.info(f"ChromaDB initialized successfully. Collection has {self.chroma_collection.count()} documents.")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise RuntimeError(f"ChromaDB initialization failed: {e}")
    
    def _load_places_from_chroma(self) -> List[Dict]:
        """Load places data from ChromaDB and convert to places format"""
        try:
            # Get all documents from ChromaDB
            all_docs = self.chroma_collection.get(include=["metadatas", "documents"])
            
            if not all_docs["documents"]:
                raise RuntimeError("No documents found in ChromaDB. Please run the crawler first to populate the database.")
            
            # Convert ChromaDB documents to places format
            places_data = []
            processed_urls = set()  # To avoid duplicates from chunks
            
            for i, (doc_text, metadata) in enumerate(zip(all_docs["documents"], all_docs["metadatas"])):
                source_url = metadata.get("source_url", "")
                
                # Skip if we've already processed this URL (avoid duplicates from chunks)
                if source_url in processed_urls:
                    continue
                processed_urls.add(source_url)
                
                # Extract location information from named entities
                named_entities = metadata.get("named_entities", "")
                location_info = self._extract_location_from_entities(named_entities)
                
                # Create place object from ChromaDB document
                place = {
                    "name": self._extract_place_name(doc_text, source_url),
                    "type": "tourist_site",  # Default type
                    "description": doc_text[:500] + "..." if len(doc_text) > 500 else doc_text,
                    "visitorAppeal": self._extract_visitor_appeal(doc_text),
                    "touristClassification": self._classify_tourist_site(doc_text, named_entities),
                    "bestTimeToVisit": "Year-round",  # Default
                    "estimatedVisitDuration": "2 hours",  # Default
                    "location": location_info,
                    "source_url": source_url,
                    "named_entities": named_entities,
                    "chroma_doc_id": all_docs["ids"][i] if i < len(all_docs["ids"]) else None
                }
                
                places_data.append(place)
            
            logger.info(f"Loaded {len(places_data)} places from ChromaDB")
            return places_data
            
        except Exception as e:
            logger.error(f"Failed to load places from ChromaDB: {e}")
            raise RuntimeError(f"ChromaDB is required but failed to load data: {e}")
    
    def _extract_location_from_entities(self, named_entities: str) -> Dict:
        """Extract location information from named entities string"""
        location = {"city": "Unknown", "country": "Spain"}  # Default
        
        if not named_entities:
            return location
        
        # Parse named entities (format: "GPE:Madrid, ORG:Museum, GPE:Spain")
        entities = named_entities.split(", ")
        cities = []
        countries = []
        
        for entity in entities:
            if ":" in entity:
                entity_type, entity_text = entity.split(":", 1)
                if entity_type == "GPE":  # Geopolitical entity
                    if entity_text.lower() in ["spain", "españa", "spanish"]:
                        countries.append(entity_text)
                    else:
                        cities.append(entity_text)
        
        # Use the first city and country found
        if cities:
            location["city"] = cities[0]
        if countries:
            location["country"] = countries[0]
        
        return location
    
    def _extract_place_name(self, doc_text: str, source_url: str) -> str:
        """Extract a meaningful place name from document text or URL"""
        # Try to extract from the first sentence or title-like text
        sentences = doc_text.split('. ')
        if sentences:
            first_sentence = sentences[0].strip()
            # Look for patterns like "Name: Something" or just use first few words
            if len(first_sentence) < 100:
                return first_sentence
        
        # Fallback to extracting from URL
        if source_url:
            url_parts = source_url.split('/')
            for part in reversed(url_parts):
                if part and part not in ['www', 'http:', 'https:', '']:
                    return part.replace('-', ' ').replace('_', ' ').title()
        
        return "Tourist Site"  # Final fallback
    
    def _extract_visitor_appeal(self, doc_text: str) -> str:
        """Extract visitor appeal information from document text"""
        # Look for keywords that indicate visitor appeal
        appeal_keywords = [
            "beautiful", "stunning", "magnificent", "historic", "cultural",
            "popular", "famous", "attraction", "must-see", "recommended",
            "experience", "visit", "explore", "discover"
        ]
        
        doc_lower = doc_text.lower()
        found_appeals = [keyword for keyword in appeal_keywords if keyword in doc_lower]
        
        if found_appeals:
            return f"Features {', '.join(found_appeals[:3])}"
        
        return "Interesting tourist destination"
    
    def _classify_tourist_site(self, doc_text: str, named_entities: str) -> str:
        """Classify the type of tourist site based on content"""
        doc_lower = doc_text.lower()
        entities_lower = named_entities.lower()
        
        # Classification keywords
        classifications = {
            "museum": ["museum", "gallery", "exhibition", "art", "collection"],
            "historical": ["historic", "history", "ancient", "heritage", "monument"],
            "cultural": ["culture", "cultural", "tradition", "festival", "event"],
            "natural": ["park", "nature", "garden", "landscape", "natural"],
            "religious": ["church", "cathedral", "temple", "religious", "sacred"],
            "entertainment": ["entertainment", "show", "theater", "cinema", "fun"]
        }
        
        for classification, keywords in classifications.items():
            if any(keyword in doc_lower or keyword in entities_lower for keyword in keywords):
                return classification
        
        return "general"
    
    def _get_coordinates_from_city(self, city_name: str) -> Tuple[Optional[float], Optional[float]]:
        """
        Get latitude and longitude coordinates for a city using geopy
        
        Args:
            city_name: Name of the city to geocode
            
        Returns:
            Tuple of (latitude, longitude) or (None, None) if not found
        """
        # Check cache first
        cache_key = f"city:{city_name}"
        if cache_key in self.coordinate_cache:
            logger.info(f"Using cached coordinates for {city_name}")
            return self.coordinate_cache[cache_key]
        
        try:
            # Use geocoder to get coordinates
            location = self.geocoder.geocode(city_name, timeout=10)
            
            if location:
                coordinates = (location.latitude, location.longitude)
                self.coordinate_cache[cache_key] = coordinates
                logger.info(f"Found coordinates for {city_name}: {location.latitude}, {location.longitude}")
                return coordinates
            else:
                self.coordinate_cache[cache_key] = (None, None)
                logger.warning(f"Could not find coordinates for city: {city_name}")
                return None, None
                
        except Exception as e:
            logger.error(f"Error geocoding city {city_name}: {e}")
            self.coordinate_cache[cache_key] = (None, None)
            return None, None
    
    def _get_coordinates_from_place(self, place: Dict) -> Tuple[Optional[float], Optional[float]]:
        """
        Get latitude and longitude coordinates for a specific place using geopy
        
        Args:
            place: Place dictionary containing name and location information
            
        Returns:
            Tuple of (latitude, longitude) or (None, None) if not found
        """
        # Create cache key based on place name and location
        place_name = place.get('name', '')
        city = place.get('location', {}).get('city', '')
        country = place.get('location', {}).get('country', '')
        cache_key = f"place:{place_name}:{city}:{country}"
        
        # Check cache first
        if cache_key in self.coordinate_cache:
            logger.info(f"Using cached coordinates for place: {place_name}")
            return self.coordinate_cache[cache_key]
        
        try:
            # Try different search strategies for better accuracy
            
            # Strategy 1: Try full place name with city and country
            if place_name and city and country:
                search_query = f"{place_name}, {city}, {country}"
                location = self.geocoder.geocode(search_query, timeout=10)
                if location:
                    coordinates = (location.latitude, location.longitude)
                    self.coordinate_cache[cache_key] = coordinates
                    logger.info(f"Found coordinates for '{search_query}': {location.latitude}, {location.longitude}")
                    return coordinates
            
            # Strategy 2: Try place name with city
            if place_name and city:
                search_query = f"{place_name}, {city}"
                location = self.geocoder.geocode(search_query, timeout=10)
                if location:
                    coordinates = (location.latitude, location.longitude)
                    self.coordinate_cache[cache_key] = coordinates
                    logger.info(f"Found coordinates for '{search_query}': {location.latitude}, {location.longitude}")
                    return coordinates
            
            # Strategy 3: Try just the place name
            if place_name:
                location = self.geocoder.geocode(place_name, timeout=10)
                if location:
                    coordinates = (location.latitude, location.longitude)
                    self.coordinate_cache[cache_key] = coordinates
                    logger.info(f"Found coordinates for '{place_name}': {location.latitude}, {location.longitude}")
                    return coordinates
            
            # Strategy 4: Fallback to city coordinates
            if city:
                city_coords = self._get_coordinates_from_city(city)
                self.coordinate_cache[cache_key] = city_coords
                return city_coords
            
            logger.warning(f"Could not find coordinates for place: {place_name}")
            self.coordinate_cache[cache_key] = (None, None)
            return None, None
                
        except Exception as e:
            logger.error(f"Error geocoding place {place.get('name', 'Unknown')}: {e}")
            self.coordinate_cache[cache_key] = (None, None)
            return None, None
    
    def _map_transport_mode_to_profile(self, transport_mode: str) -> str:
        """
        Map Spanish transport modes from app.py to OpenRouteService profiles
        
        Args:
            transport_mode: Transport mode in Spanish from user selection
            
        Returns:
            OpenRouteService profile name
        """
        # Map Spanish transport modes to OpenRouteService profiles
        transport_map = {
            "A pie": "foot-walking",
            "Bicicleta": "cycling-regular", 
            "Transporte público": "foot-walking",  # Use walking as approximation
            "Coche/taxi": "driving-car",
            "Otro": "foot-walking"  # Default to walking
        }
        
        return transport_map.get(transport_mode, "foot-walking")
    
    def _calculate_time_matrix_ors(self, places: List[Dict], user_lat: float, user_lon: float, transport_mode: str) -> np.ndarray:
        """
        Calculate time matrix using OpenRouteService API
        
        Args:
            places: List of places with coordinates
            user_lat: User's latitude
            user_lon: User's longitude
            transport_mode: Transportation mode from user selection (Spanish)
            
        Returns:
            Time matrix in minutes
            
        Raises:
            RuntimeError: If OpenRouteService API call fails
        """
        api_key = self.openrouteservice_api_key
        
        # Prepare coordinates for API call
        coordinates = [[user_lon, user_lat]]  # ORS expects [lon, lat]
        for place in places:
            lat, lon = self._get_coordinates_from_place(place)
            if lat is not None and lon is not None:
                coordinates.append([lon, lat])
            else:
                raise RuntimeError(f"Could not get coordinates for place: {place.get('name', 'Unknown')}")
        
        # Map Spanish transport mode to OpenRouteService profile
        profile = self._map_transport_mode_to_profile(transport_mode)
        
        url = f"https://api.openrouteservice.org/v2/matrix/{profile}"
        headers = {
            'Accept': 'application/json, application/geo+json, application/gpx+xml, img/png; charset=utf-8',
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json; charset=utf-8'
        }
        body = {
            "locations": coordinates,
            "metrics": ["duration"],
            "units": "m"
        }
        
        logger.info(f"Calling OpenRouteService API with {len(coordinates)} locations using {profile} profile for transport mode: {transport_mode}")
        
        try:
            response = requests.post(url, json=body, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                durations = np.array(data['durations'])
                
                # Convert from seconds to minutes
                time_matrix = durations / 60
                logger.info(f"Successfully calculated time matrix using OpenRouteService with {profile} profile")
                return time_matrix
            else:
                error_msg = f"OpenRouteService API failed with status {response.status_code}: {response.text}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
                
        except requests.exceptions.RequestException as e:
            error_msg = f"Network error calling OpenRouteService API: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error calling OpenRouteService API: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
        
    def _generate_place_embeddings(self, places: List[Dict]) -> List[np.ndarray]:
        """Generate embeddings for places based on their descriptions and attributes"""
        embeddings = []
        
        for place in places:
            # Create a comprehensive text representation of the place
            text_representation = f"""
            Name: {place.get('name', '')}
            Type: {place.get('type', '')}
            Description: {place.get('description', '')}
            Visitor Appeal: {place.get('visitorAppeal', '')}
            Classification: {place.get('touristClassification', '')}
            Best Time to Visit: {place.get('bestTimeToVisit', '')}
            Location: {place.get('location', {}).get('city', '')} {place.get('location', {}).get('country', '')}
            """.strip()
            
            # Generate embedding
            embedding = self.model.encode(text_representation)
            embeddings.append(embedding)
        
        return embeddings
    
    def _generate_user_embedding(self, user_preferences: Dict) -> np.ndarray:
        """Generate embedding for user preferences"""
        # Create text representation of user preferences
        preferences_text = []
        
        # Add category preferences with specific descriptions
        if 'category_interest' in user_preferences:
            for category, score in user_preferences['category_interest'].items():
                if score == 5:
                    preferences_text.append(f"Absolutely loves and is passionate about {category}, seeks out {category}-related activities whenever possible")
                elif score == 4:
                    preferences_text.append(f"Really enjoys and has strong interest in {category}, actively looks for {category} experiences")
                elif score == 3:
                    preferences_text.append(f"Has moderate interest in {category}, open to {category} activities but not a priority")
                elif score == 2:
                    preferences_text.append(f"Has limited interest in {category}, might participate if convenient but not enthusiastic")
                elif score == 1:
                    preferences_text.append(f"Dislikes or has no interest in {category}, prefers to avoid {category}-related activities")
        
        # Add transport preferences
        if 'transport_modes' in user_preferences:
            transport_text = f"Prefers transportation methods: {', '.join(user_preferences['transport_modes'])}"
            preferences_text.append(transport_text)
        
        # Add additional notes
        if 'user_notes' in user_preferences and user_preferences['user_notes']:
            preferences_text.append(f"Additional preferences and notes: {user_preferences['user_notes']}")
        
        # Add city and duration info
        if 'city' in user_preferences:
            preferences_text.append(f"Currently visiting {user_preferences['city']} for tourism")
        
        if 'available_hours' in user_preferences:
            preferences_text.append(f"Has {user_preferences['available_hours']} hours available for tourism activities")
        
        full_text = ". ".join(preferences_text)
        return self.model.encode(full_text)
    
    def _filter_places_by_distance(self, places: List[Dict], max_distance_km: float, user_lat: float, user_lon: float) -> Tuple[List[Dict], List[int]]:
        """Filter places based on maximum distance from user location"""
        filtered_places = []
        filtered_indices = []
        
        for i, place in enumerate(places):
            place_lat, place_lon = self._get_coordinates_from_place(place)
            
            if place_lat is not None and place_lon is not None:
                # Calculate geodesic distance
                distance_km = geodesic((user_lat, user_lon), (place_lat, place_lon)).kilometers
                
                if distance_km <= max_distance_km:
                    filtered_places.append(place)
                    filtered_indices.append(i + 1)  # +1 because user is at index 0 in distance matrix
                    logger.info(f"Place '{place.get('name', 'Unknown')}' is {distance_km:.2f}km away - included")
                else:
                    logger.info(f"Place '{place.get('name', 'Unknown')}' is {distance_km:.2f}km away - excluded (max: {max_distance_km}km)")
            else:
                logger.warning(f"Could not get coordinates for place '{place.get('name', 'Unknown')}' - excluded")
        
        return filtered_places, filtered_indices
    
    def _filter_places_by_city(self, places: List[Dict], target_city: str) -> List[Dict]:
        """Filter places by city"""
        return [place for place in places if place['location']['city'].lower() == target_city.lower()]
    
    def _semantic_search_chroma(self, query: str, n_results: int = 20) -> List[Dict]:
        """
        Perform semantic search in ChromaDB using user preferences
        
        Args:
            query: Search query based on user preferences
            n_results: Number of results to return
            
        Returns:
            List of places from semantic search results
        """
        try:
            # Perform semantic search in ChromaDB
            search_results = self.chroma_collection.query(
                query_texts=[query],
                n_results=n_results,
                include=["metadatas", "documents", "distances"]
            )
            
            if not search_results["documents"] or not search_results["documents"][0]:
                logger.warning("No semantic search results found in ChromaDB")
                return []
            
            # Convert search results to places format
            places_data = []
            processed_urls = set()
            
            for i, (doc_text, metadata, distance) in enumerate(zip(
                search_results["documents"][0], 
                search_results["metadatas"][0],
                search_results["distances"][0]
            )):
                source_url = metadata.get("source_url", "")
                
                # Skip duplicates from chunks
                if source_url in processed_urls:
                    continue
                processed_urls.add(source_url)
                
                # Extract location information
                named_entities = metadata.get("named_entities", "")
                location_info = self._extract_location_from_entities(named_entities)
                
                # Create place object
                place = {
                    "name": self._extract_place_name(doc_text, source_url),
                    "type": "tourist_site",
                    "description": doc_text[:500] + "..." if len(doc_text) > 500 else doc_text,
                    "visitorAppeal": self._extract_visitor_appeal(doc_text),
                    "touristClassification": self._classify_tourist_site(doc_text, named_entities),
                    "bestTimeToVisit": "Year-round",
                    "estimatedVisitDuration": "2 hours",
                    "location": location_info,
                    "source_url": source_url,
                    "named_entities": named_entities,
                    "semantic_distance": distance,
                    "chroma_doc_id": search_results["ids"][0][i] if i < len(search_results["ids"][0]) else None
                }
                
                places_data.append(place)
            
            logger.info(f"Semantic search returned {len(places_data)} unique places")
            return places_data
            
        except Exception as e:
            logger.error(f"Error performing semantic search in ChromaDB: {e}")
            return []
    
    def _create_search_query_from_preferences(self, user_preferences: Dict) -> str:
        """
        Create a search query string from user preferences for semantic search
        
        Args:
            user_preferences: User preferences dictionary
            
        Returns:
            Search query string
        """
        query_parts = []
        
        # Add city information
        if 'city' in user_preferences:
            query_parts.append(f"tourist attractions in {user_preferences['city']}")
        
        # Add high-interest categories (score >= 4)
        if 'category_interest' in user_preferences:
            high_interest_categories = [
                category for category, score in user_preferences['category_interest'].items() 
                if score >= 4
            ]
            if high_interest_categories:
                query_parts.append(f"interested in {', '.join(high_interest_categories)}")
        
        # Add user notes
        if 'user_notes' in user_preferences and user_preferences['user_notes']:
            query_parts.append(user_preferences['user_notes'])
        
        # Create final query
        search_query = ". ".join(query_parts)
        logger.info(f"Created search query: {search_query}")
        return search_query
    
    def _calculate_cosine_similarity(self, user_embedding: np.ndarray, place_embeddings: List[np.ndarray]) -> List[float]:
        """
        Calculate cosine similarity between user embedding and place embeddings
        
        Args:
            user_embedding: User preference embedding
            place_embeddings: List of place embeddings
            
        Returns:
            List of cosine similarity scores
        """
        similarities = []
        
        # Reshape user embedding to 2D array for sklearn
        user_embedding_2d = user_embedding.reshape(1, -1)
        
        for place_embedding in place_embeddings:
            # Reshape place embedding to 2D array
            place_embedding_2d = place_embedding.reshape(1, -1)
            
            # Calculate cosine similarity
            similarity = cosine_similarity(user_embedding_2d, place_embedding_2d)[0][0]
            similarities.append(similarity)
        
        return similarities
    
    def _filter_places_by_similarity(self, places: List[Dict], place_embeddings: List[np.ndarray], 
                                   user_embedding: np.ndarray, similarity_threshold: float = 0.3) -> Tuple[List[Dict], List[np.ndarray], List[float]]:
        """
        Filter places based on cosine similarity with user preferences
        
        Args:
            places: List of places
            place_embeddings: List of place embeddings
            user_embedding: User preference embedding
            similarity_threshold: Minimum similarity threshold (default: 0.3)
            
        Returns:
            Tuple of (filtered_places, filtered_embeddings, similarity_scores)
        """
        # Calculate cosine similarities
        similarities = self._calculate_cosine_similarity(user_embedding, place_embeddings)
        
        filtered_places = []
        filtered_embeddings = []
        filtered_similarities = []
        
        for i, (place, embedding, similarity) in enumerate(zip(places, place_embeddings, similarities)):
            if similarity >= similarity_threshold:
                filtered_places.append(place)
                filtered_embeddings.append(embedding)
                filtered_similarities.append(similarity)
                logger.info(f"Place '{place.get('name', 'Unknown')}' has similarity {similarity:.3f} - included")
            else:
                logger.info(f"Place '{place.get('name', 'Unknown')}' has similarity {similarity:.3f} - excluded (threshold: {similarity_threshold})")
        
        logger.info(f"Filtered {len(filtered_places)} places out of {len(places)} based on similarity threshold {similarity_threshold}")
        
        return filtered_places, filtered_embeddings, filtered_similarities
    
    def _call_openrouter_llm(self, prompt: str, model: str = "anthropic/claude-3.5-sonnet") -> str:
        """
        Call OpenRouter API to get LLM response
        
        Args:
            prompt: The prompt to send to the LLM
            model: The model to use (default: Claude 3.5 Sonnet)
            
        Returns:
            LLM response text
            
        Raises:
            RuntimeError: If OpenRouter API call fails
        """
        try:
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.openrouter_api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://tourist-guide-app.com",
                "X-Title": "Tourist Guide RAG System"
            }
            
            data = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.7,
                "max_tokens": 2000
            }
            
            logger.info(f"Calling OpenRouter API with model: {model}")
            response = requests.post(url, headers=headers, json=data)
            
            if response.status_code == 200:
                result = response.json()
                llm_response = result['choices'][0]['message']['content']
                logger.info("Successfully received response from OpenRouter API")
                return llm_response
            else:
                logger.error(f"OpenRouter API returned status {response.status_code}: {response.text}")
                raise RuntimeError(f"OpenRouter API failed with status {response.status_code}: {response.text}")
            
        except Exception as e:
            logger.error(f"Error calling OpenRouter API: {e}")
            raise RuntimeError(f"Failed to call OpenRouter LLM: {e}")
    
    def _parse_llm_time_estimates(self, llm_response: str, places: List[Dict]) -> List[float]:
        """
        Parse LLM response to extract time estimates for each place
        
        Args:
            llm_response: Response from LLM
            places: List of places
            
        Returns:
            List of time estimates in hours
        """
        time_estimates = []
        lines = llm_response.split('\n')
        
        # Try to extract time estimates from the response
        for i, place in enumerate(places):
            found_time = None
            
            # Look for patterns like "Place 1: 2.5 hours" or "1. Place Name: 3.0 hours"
            for line in lines:
                if f"Place {i+1}:" in line or f"{i+1}." in line or place['name'] in line:
                    # Extract number followed by "hour" or "h"
                    time_match = re.search(r'(\d+\.?\d*)\s*(?:hours?|h)', line, re.IGNORECASE)
                    if time_match:
                        found_time = float(time_match.group(1))
                        break
            
            # Fallback to estimated duration from place data or default
            if found_time is None:
                estimated_duration = place.get('estimatedVisitDuration', '2 hours')
                # Extract number from estimated duration
                duration_match = re.search(r'(\d+\.?\d*)', estimated_duration)
                if duration_match:
                    found_time = float(duration_match.group(1))
                else:
                    found_time = 2.0  # Default fallback
            
            time_estimates.append(found_time)
        
        return time_estimates
    
    def process_user_request(self, user_preferences: Dict, user_lat: float, user_lon: float, transport_mode: str) -> Dict[str, Any]:
        """
        Main RAG processing function for metaheuristic algorithms
        
        Args:
            user_preferences: Dictionary containing user preferences from app.py
            user_lat: User's latitude
            user_lon: User's longitude
            transport_mode: Transportation mode from user selection (Spanish: "A pie", "Bicicleta", etc.)
            
        Returns:
            Dictionary containing:
            - filtered_places: Places within distance limit and similarity threshold
            - time_matrix: Travel time matrix in minutes
            - place_embeddings: Embeddings for each place
            - user_embedding: User preference embedding
            - llm_time_estimates: LLM estimated time for each place
            - llm_response: Full LLM response
            - similarity_scores: Cosine similarity scores for filtered places
            
        Raises:
            RuntimeError: If API calls fail (no fallbacks)
        """
        # Use semantic search for place discovery
        logger.info("Using ChromaDB semantic search for place discovery")
        
        # Create search query from user preferences
        search_query = self._create_search_query_from_preferences(user_preferences)
        
        # Perform semantic search
        semantic_places = self._semantic_search_chroma(search_query, n_results=50)
        
        if semantic_places:
            # Filter semantic results by city
            city = user_preferences.get('city', '')
            city_places = self._filter_places_by_city(semantic_places, city)
            
            if not city_places:
                logger.warning(f"No semantic search results found for city: {city}. Using all available places.")
                city_places = self._filter_places_by_city(self.places_data, city)
            else:
                logger.info(f"Found {len(city_places)} places from semantic search for city: {city}")
        else:
            logger.warning("Semantic search returned no results. Using all available places.")
            city = user_preferences.get('city', '')
            city_places = self._filter_places_by_city(self.places_data, city)
        
        if not city_places:
            raise RuntimeError(f"No places found for city: {user_preferences.get('city', 'Unknown')}")
        
        # Filter places by maximum distance
        max_distance_km = user_preferences.get('max_distance', 10)
        distance_filtered_places, _ = self._filter_places_by_distance(
            city_places, max_distance_km, user_lat, user_lon
        )
        
        if not distance_filtered_places:
            raise RuntimeError(f"No places found within {max_distance_km}km of user location")
        
        # Generate embeddings for distance-filtered places
        logger.info("Generating place embeddings...")
        distance_place_embeddings = self._generate_place_embeddings(distance_filtered_places)
        
        # Generate user embedding
        logger.info("Generating user embedding...")
        user_embedding = self._generate_user_embedding(user_preferences)
        
        # Filter places by cosine similarity with threshold
        logger.info("Filtering places by cosine similarity...")
        similarity_threshold = user_preferences.get('similarity_threshold', 0.2)
        logger.info(f"Using similarity threshold: {similarity_threshold}")
        
        filtered_places, place_embeddings, similarity_scores = self._filter_places_by_similarity(
            distance_filtered_places, distance_place_embeddings, user_embedding, similarity_threshold
        )
        
        if not filtered_places:
            raise RuntimeError(f"No places found with similarity >= {similarity_threshold}. Try lowering the similarity threshold.")
        
        # Calculate time matrix only for the final filtered places
        logger.info(f"Calculating time matrix with OpenRouteService for {len(filtered_places)} filtered places using transport mode: {transport_mode}")
        time_matrix = self._calculate_time_matrix_ors(filtered_places, user_lat, user_lon, transport_mode)
        
        # Generate LLM prompt and get response
        logger.info("Calling LLM for time estimates...")
        llm_prompt = self._get_llm_recommendation_prompt(filtered_places, user_preferences)
        llm_response = self._call_openrouter_llm(llm_prompt)
        
        # Parse LLM response to get time estimates
        llm_time_estimates = self._parse_llm_time_estimates(llm_response, filtered_places)
        
        logger.info(f"RAG processing complete using ChromaDB semantic search. Found {len(filtered_places)} places after filtering.")
        
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
            'data_source': "ChromaDB semantic search"
        }
    
    def _get_llm_recommendation_prompt(self, places: List[Dict], user_preferences: Dict) -> str:
        """
        Generate a prompt for the LLM to estimate time interest for each place
        
        Args:
            places: List of filtered places
            user_preferences: User preferences dictionary
            
        Returns:
            Formatted prompt for the LLM
        """
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
        
        # Add category interests
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
   - Visitor Appeal: {place.get('visitorAppeal', 'No information')}
   - Classification: {place.get('touristClassification', 'Unknown')}
   - Estimated Visit Duration: {place.get('estimatedVisitDuration', 'Unknown')}
"""
        
        prompt += """
TASK: For each place listed above, estimate how many hours the tourist would be interested in spending there based on their preferences. Consider:
1. The tourist's category interests and ratings
2. The type and appeal of each place
3. The tourist's additional notes and preferences
4. The suggested visit duration for each place

Provide your response in the following format:
Place 1: X.X hours - Brief reasoning
Place 2: X.X hours - Brief reasoning
...

Be realistic with time estimates and consider the tourist's specific interests.
"""
        
        return prompt

