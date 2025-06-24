"""
RAG Planner - Main orchestrator for Retrieval-Augmented Generation
Coordinates all RAG components for tourist place recommendations
"""

import logging
from typing import Dict, List, Any
from dotenv import load_dotenv

from .database_manager import DatabaseManager
from .nlp_processor import NLPProcessor
from .geocoding_service import GeocodingService
from .similarity_calculator import SimilarityCalculator
from .llm_service import LLMService
from .place_filter import PlaceFilter

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGPlanner:
    """Main RAG Planner that orchestrates all components"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", chroma_db_path: str = None):
        """
        Initialize the RAG Planner with all components
        """
        logger.info("Initializing RAG Planner components...")
        
        # Initialize all components
        self.database_manager = DatabaseManager(chroma_db_path)
        self.nlp_processor = NLPProcessor()
        self.geocoding_service = GeocodingService()
        self.similarity_calculator = SimilarityCalculator(model_name, self.nlp_processor)
        self.llm_service = LLMService()
        self.place_filter = PlaceFilter()
        
        # Load places data from database
        self.places_data = self.database_manager.load_places_from_chroma()
        
        # Expose commonly used components for backward compatibility
        self.keybert_model = self.nlp_processor.keybert_model
        self.sentiment_analyzer = self.nlp_processor.sentiment_analyzer
        self.spanish_stopwords = self.nlp_processor.spanish_stopwords
        
        logger.info("RAG Planner initialized successfully with all components.")

    def process_user_request(self, user_preferences: Dict, user_lat: float, 
                           user_lon: float, transport_mode: str) -> Dict[str, Any]:
        """Main RAG processing function."""
        logger.info("🚀 Starting RAG processing pipeline...")
        
        # Step 1: Create search query from preferences
        search_query = self.nlp_processor.create_search_query_from_preferences(user_preferences)
        logger.info(f"🔍 Search query: {search_query}")
        
        # Step 2: Get city for filtering
        city = user_preferences.get('city', '')
        logger.info(f"🏙️ Target city: {city}")
        
        # Step 3: Perform semantic search with city filter
        if city:
            semantic_places = self.database_manager.semantic_search_chroma(search_query, n_results=100, city_filter=city)
            logger.info(f"📊 Semantic search with city filter found {len(semantic_places)} places for {city}")
            
            # If no results with semantic search, fallback to all places from that city
            if not semantic_places:
                logger.info(f"⚠️ No semantic search results for {city}, falling back to all places from city")
                city_places = self.place_filter.filter_places_by_city(self.places_data, city)
                logger.info(f"📍 Fallback city filtering: {len(city_places)} places")
            else:
                city_places = semantic_places
        else:
            # If no city specified, perform semantic search without filter
            semantic_places = self.database_manager.semantic_search_chroma(search_query, n_results=100)
            logger.info(f"📊 Semantic search without city filter found {len(semantic_places)} places")
            city_places = semantic_places

        if not city_places:
            available_cities = self.place_filter.get_available_cities(self.places_data)
            logger.warning(f"⚠️ No places found for city: {city}")
            logger.info(f"🏙️ Available cities in database: {available_cities}")
            
            # Return empty result instead of throwing error to allow dynamic validation
            return {
                'filtered_places': [],
                'enhanced_scores': [],
                'time_matrix': [],
                'place_embeddings': [],
                'user_embedding': [],
                'llm_time_estimates': [],
                'llm_response': f"No places found for {city}. Available cities: {available_cities}",
                'similarity_scores': [],
                'user_preferences': user_preferences,
                'transport_mode': transport_mode,
                'db_categories': [],
                'data_source': f"No data available for {city} - dynamic crawler should be triggered",
                'needs_crawler': True,
                'available_cities': available_cities
            }

        # Step 4: Filter by distance
        max_distance_km = user_preferences.get('max_distance', 10)
        logger.info(f"📏 Filtering by distance: {max_distance_km}km from ({user_lat}, {user_lon})")
        
        # Log some places before distance filtering
        logger.info(f"📋 Places found for {city or 'all cities'} before distance filtering:")
        for i, place in enumerate(city_places[:5]):  # Show first 5 places
            place_lat, place_lon = self.geocoding_service.get_coordinates_from_place(place)
            logger.info(f"  {i+1}. {place['name']} - Coords: {place_lat}, {place_lon}")
        
        distance_filtered_places, _ = self.geocoding_service.filter_places_by_distance(
            city_places, max_distance_km, user_lat, user_lon
        )
        logger.info(f"📍 Distance filtering result: {len(distance_filtered_places)} places within {max_distance_km}km")

        if not distance_filtered_places:
            self._log_distance_analysis(city_places, user_lat, user_lon, max_distance_km)
            logger.warning(f"⚠️ No places found within {max_distance_km}km for {city}")
            
            # Return result with insufficient data to trigger dynamic validation
            return {
                'filtered_places': [],
                'enhanced_scores': [],
                'time_matrix': [],
                'place_embeddings': [],
                'user_embedding': [],
                'llm_time_estimates': [],
                'llm_response': f"No places found within {max_distance_km}km for {city}. Try increasing the distance or check if more data is available.",
                'similarity_scores': [],
                'user_preferences': user_preferences,
                'transport_mode': transport_mode,
                'db_categories': [],
                'data_source': f"Insufficient data for {city} within {max_distance_km}km - may need more data",
                'needs_crawler': True,
                'distance_issue': True
            }
        
        # Step 5: Calculate enhanced similarity scores
        enhanced_scores = []
        for place in distance_filtered_places:
            score = self.similarity_calculator.calculate_enhanced_similarity(user_preferences, place)
            enhanced_scores.append(score)
        
        # Step 6: Generate embeddings
        logger.info(f"🔄 Generating embeddings for {len(distance_filtered_places)} places...")
        distance_place_embeddings = self.similarity_calculator.generate_place_embeddings(distance_filtered_places)
        logger.info(f"✅ Generated {len(distance_place_embeddings)} place embeddings")
        
        logger.info(f"🔄 Generating user embedding from preferences...")
        user_embedding = self.similarity_calculator.generate_user_embedding(user_preferences)
        logger.info(f"✅ Generated user embedding with {len(user_embedding)} dimensions")
        
        # Step 7: Calculate cosine similarities
        logger.info(f"🔄 Calculating similarities between user and places...")
        all_similarities = self.similarity_calculator.calculate_cosine_similarity(
            user_embedding, distance_place_embeddings, user_preferences, distance_filtered_places
        )
        logger.info(f"✅ Calculated {len(all_similarities)} similarity scores")
        
        # Step 8: Sort places by similarity
        places_with_similarity = list(zip(distance_filtered_places, distance_place_embeddings, all_similarities))
        places_with_similarity.sort(key=lambda x: x[2], reverse=True)  # Sort by similarity descending
        
        # Step 9: Print similarity ranking for debugging
        self.place_filter.print_similarity_ranking(places_with_similarity, user_preferences)
        
        # Step 10: Prepare final data and limit to top 50 for time matrix calculation
        sorted_places = [item[0] for item in places_with_similarity]
        place_embeddings = [item[1] for item in places_with_similarity]
        similarity_scores = [item[2] for item in places_with_similarity]
        
        # Limit to top 50 places for time matrix calculation (performance optimization)
        max_places_for_time_matrix = 50
        top_places_for_matrix = sorted_places[:max_places_for_time_matrix]
        top_embeddings_for_matrix = place_embeddings[:max_places_for_time_matrix]
        top_scores_for_matrix = similarity_scores[:max_places_for_time_matrix]
        
        logger.info(f"📊 Total places found: {len(sorted_places)}")
        logger.info(f"📊 Using top {len(top_places_for_matrix)} places for time matrix calculation")
        logger.info(f"📊 Similarity range: {min(similarity_scores):.3f} - {max(similarity_scores):.3f}")

        # Step 11: Calculate time matrix only for top 50 places
        time_matrix = self.geocoding_service.calculate_time_matrix_ors(
            top_places_for_matrix, user_lat, user_lon, transport_mode
        )

        # Step 12: Process with LLM for time estimates (only for top 50)
        llm_response, llm_time_estimates = self.llm_service.process_places_with_llm(
            top_places_for_matrix, user_preferences
        )

        # Step 13: Extract categories from database (only for top 50)
        db_categories = [place.get('category', 'general') for place in top_places_for_matrix]
        
        # Use top 50 as final places for route optimization
        final_places = top_places_for_matrix
        final_embeddings = top_embeddings_for_matrix
        final_scores = top_scores_for_matrix

        # Step 14: Return comprehensive results
        return {
            'filtered_places': final_places,
            'enhanced_scores': final_scores,
            'time_matrix': time_matrix,
            'place_embeddings': final_embeddings,
            'user_embedding': user_embedding,
            'llm_time_estimates': llm_time_estimates,
            'llm_response': llm_response,
            'similarity_scores': final_scores,
            'user_preferences': user_preferences,
            'transport_mode': transport_mode,
            'db_categories': db_categories,
            'data_source': f"ChromaDB semantic search with city filtering - top {len(final_places)} places by similarity"
        }

    def _log_distance_analysis(self, city_places: List[Dict], user_lat: float, 
                              user_lon: float, max_distance_km: float):
        """Log detailed distance analysis for debugging"""
        from geopy.distance import geodesic
        
        logger.error(f"❌ No places found within {max_distance_km}km")
        logger.info(f"🔍 Distance analysis for places:")
        
        for i, place in enumerate(city_places[:10]):  # Show first 10 places
            place_lat, place_lon = self.geocoding_service.get_coordinates_from_place(place)
            if place_lat is not None and place_lon is not None:
                distance_km = geodesic((user_lat, user_lon), (place_lat, place_lon)).kilometers
                logger.info(f"  {i+1}. {place['name']} - {distance_km:.2f}km away")
            else:
                logger.info(f"  {i+1}. {place['name']} - No coordinates available")
        
        # Try with a larger distance to see if there are any places at all
        logger.info(f"🔍 Trying with 50km radius...")
        extended_places, _ = self.geocoding_service.filter_places_by_distance(
            city_places, 50, user_lat, user_lon
        )
        logger.info(f"📍 Places within 50km: {len(extended_places)}")

    # Backward compatibility methods
    def calculate_cosine_similarity(self, user_embedding, place_embeddings, user_preferences=None, places=None):
        """Backward compatibility: delegate to SimilarityCalculator"""
        return self.similarity_calculator.calculate_cosine_similarity(user_embedding, place_embeddings, user_preferences, places)

    def preprocess_text(self, text: str) -> str:
        """Backward compatibility wrapper"""
        return self.nlp_processor.preprocess_text(text)

    def extract_query_keywords(self, text: str, max_keywords: int = 5) -> List[str]:
        """Backward compatibility wrapper"""
        return self.nlp_processor.extract_query_keywords(text, max_keywords)

    def analyze_query_sentiment(self, text: str) -> Dict[str, Any]:
        """Backward compatibility wrapper"""
        return self.nlp_processor.analyze_query_sentiment(text)