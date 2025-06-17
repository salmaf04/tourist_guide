import chromadb
from chromadb.utils import embedding_functions

from sentence_transformers import SentenceTransformer
import logging

from crawler.items import TouristPlaceItem

logger = logging.getLogger(__name__)

class ChromaPipeline:
    def __init__(self):
        self.chroma_client = chromadb.PersistentClient(path="../db/")
        self.collection = self.chroma_client.get_or_create_collection(
            name="tourist_places",
            embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        )
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    def process_item(self, item: TouristPlaceItem, spider):
        text_to_embed = f"""
        Name: {item['name']}
        City: {item['city']}
        Category: {item['category']}
        Description: {item['description']}
        """.strip()
        embedding = self.embedding_model.encode(text_to_embed).tolist()
        metadata = {
            "name": str(item['name']) if item['name'] else "",
            "city": str(item['city']) if item['city'] else "",
            "category": str(item['category']) if item['category'] else "",
            "description": str(item['description']) if item['description'] else "",
            "coordinates": str(item['coordinates']) if item['coordinates'] else "None",
        }
        # Use name + city as unique ID since we no longer have source_url
        unique_id = f"{item['name']}_{item['city']}".replace(" ", "_").lower()
        self.collection.add(
            documents=[text_to_embed],
            metadatas=[metadata],
            ids=[unique_id],
            embeddings=[embedding]
        )
        logger.info(f"Stored place: {item['name']} in Chroma")
        return item