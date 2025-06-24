"""
Database Manager for ChromaDB operations
Handles all database connections and data loading operations
"""

import chromadb
import logging
import os
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages ChromaDB operations for tourist places data"""
    
    def __init__(self, chroma_db_path: str = None):
        """Initialize database manager with ChromaDB path"""
        if chroma_db_path is None:
            # Get the absolute path to the project root
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            self.chroma_db_path = os.path.join(project_root, "src", "crawler", "db")
        else:
            self.chroma_db_path = chroma_db_path
        
        self.chroma_client = None
        self.chroma_collection = None
        self._init_chromadb()
    
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
    
    def load_places_from_chroma(self) -> List[Dict]:
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
    
    def semantic_search_chroma(self, query: str, n_results: int = 50, city_filter: str = None) -> List[Dict]:
        """Perform semantic search in ChromaDB with optional city filtering and deduplication."""
        try:
            # Build query parameters
            query_params = {
                "query_texts": [query],
                "n_results": n_results,
                "include": ["metadatas", "documents", "distances"]
            }
            
            # Add city filter if provided
            if city_filter:
                query_params["where"] = {"city": city_filter}
                logger.info(f"🔍 Semantic search with city filter: {city_filter}")
            
            search_results = self.chroma_collection.query(**query_params)

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
    
    def get_collection_count(self) -> int:
        """Get the number of documents in the collection"""
        try:
            return self.chroma_collection.count()
        except Exception as e:
            logger.error(f"Error getting collection count: {e}")
            return 0