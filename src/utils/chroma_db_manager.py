import uuid
from typing import Dict
import chromadb
import numpy as np
import sys
import os
from crawler.model import TouristPlace


class ChromaDBManager:
    """Manejador de operaciones con ChromaDB"""

    def __init__(self, collection: chromadb.Collection):
        self.collection = collection

    def place_exists(self, name: str, city: str) -> bool:
        """Verifica si un lugar ya existe en la colección."""
        docs = self.collection.get(where={"name": name, "city": city})
        return len(docs["ids"]) > 0

    def add_place(self, place: TouristPlace, embedding: np.ndarray) -> str:
        """Añade un lugar turístico a la colección con su embedding."""
        doc_id = str(uuid.uuid4())
        text_representation = f"""
        Name: {place.name}
        City: {place.city}
        Category: {place.category}
        Description: {place.description}
        Coordinates: {place.coordinates if place.coordinates else 'Unknown'}
        """.strip()

        metadata = {
            "name": place.name,
            "city": place.city,
            "category": place.category,
            "description": place.description,
            "coordinates": str(place.coordinates) if place.coordinates else "Unknown",
        }

        self.collection.add(
            documents=[text_representation],
            embeddings=[embedding.tolist()],  # Convertir numpy array a lista
            metadatas=[metadata],
            ids=[doc_id]
        )
        return doc_id

    def get_all_documents(self):
        """Obtiene todos los documentos de la colección."""
        return self.collection.get(include=["metadatas", "documents"])

    def test_collection(self):
        """Prueba la colección con una consulta de ejemplo."""
        if self.collection.count() > 0:
            print(f"\n--- Example ChromaDB Query ---")
            query_results = self.collection.query(
                query_texts=["cultural events in Madrid"],
                n_results=2,
                include=["metadatas", "documents", "distances"]
            )
            if query_results and query_results.get('documents'):
                print("Query Results:")
                for i, doc_text in enumerate(query_results['documents'][0]):
                    metadata = query_results['metadatas'][0][i]
                    distance = query_results['distances'][0][i]
                    print(f"\nResult {i+1}:")
                    print(f"  Name: {metadata.get('name', 'N/A')}")
                    print(f"  City: {metadata.get('city', 'N/A')}")
                    print(f"  Distance: {distance:.4f}")
                    print(f"  Text (snippet): {doc_text[:250]}...")
            else:
                print("No results found or issue with results format.")

            print("\n--- Todos los documentos en la colección ChromaDB ---")
            all_docs = self.collection.get(include=["metadatas", "documents"])
            for i, doc in enumerate(all_docs["documents"]):
                print(f"\nDocumento {i+1}:")
                print(f"  ID: {all_docs['ids'][i]}")
                print(f"  Name: {all_docs['metadatas'][i].get('name', 'N/A')}")
                print(f"  City: {all_docs['metadatas'][i].get('city', 'N/A')}")
                print(f"  Category: {all_docs['metadatas'][i].get('category', 'N/A')}")
                print(f"  Texto (snippet): {doc[:250]}...")
        else:
            print("\nChromaDB collection is empty. No query performed.")
