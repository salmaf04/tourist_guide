import chromadb
from chromadb.config import Settings

# Cambia la ruta si tu base está en otro lugar
chroma_client = chromadb.PersistentClient(path="src/crawler/db")

# Listar todas las colecciones
try:
    collections = chroma_client.list_collections()
except Exception as e:
    print(f"Error listando colecciones: {e}")
    exit(1)
    
print("Colecciones encontradas:")
for col in collections:
    print(f"- {col.name}")

    # Obtener la colección y mostrar los documentos
    collection = chroma_client.get_collection(col.name)
    docs = collection.get()
    if not docs['ids']:
        print(f"  ⚠️ La colección '{col.name}' está vacía.")
    else:
        print(f"  Documentos en '{col.name}':")
        for i, doc_id in enumerate(docs['ids']):
            print(f"    {i+1}. ID: {doc_id}")
            print(f"       Metadatos: {docs['metadatas'][i]}")
            print(f"       Documento: {docs['documents'][i]}")
            # Si quieres ver los embeddings, descomenta la siguiente línea
            # print(f"       Embedding: {docs['embeddings'][i]}")
    print()

print("Fin del listado.")