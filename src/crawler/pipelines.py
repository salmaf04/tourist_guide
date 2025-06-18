import chromadb
from chromadb.utils import embedding_functions

from sentence_transformers import SentenceTransformer
import logging

from crawler.items import TouristPlaceItem

logger = logging.getLogger(__name__)

class ChromaPipeline:
    def __init__(self):
        self.chroma_client = chromadb.PersistentClient(path="../db/")
        
        # Intentar obtener la colección existente primero
        try:
            self.collection = self.chroma_client.get_collection(name="tourist_places")
            logger.info("Usando colección existente 'tourist_places' en pipeline")
        except Exception:
            # Si no existe, crear una nueva sin especificar embedding function
            try:
                self.collection = self.chroma_client.create_collection(name="tourist_places")
                logger.info("Creada nueva colección 'tourist_places' en pipeline")
            except Exception:
                # Si ya existe pero hay conflicto, eliminarla y recrearla
                try:
                    self.chroma_client.delete_collection(name="tourist_places")
                    logger.info("Colección existente eliminada en pipeline debido a conflicto")
                except Exception:
                    pass
                self.collection = self.chroma_client.create_collection(name="tourist_places")
                logger.info("Nueva colección 'tourist_places' creada en pipeline después de resolver conflicto")
        
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