import chromadb

def ver_base_datos():
    chroma_client = chromadb.PersistentClient(path="db/")
    collection = chroma_client.get_or_create_collection(name="tourism_docs")
    all_docs = collection.get(include=["metadatas", "documents"])
    print("Total documentos:", len(all_docs.get("documents", [])))
    for i, meta in enumerate(all_docs.get("metadatas", [])):
        print(f"Documento {i+1}:")
        print("  Metadatos:", meta)
        if "documents" in all_docs and i < len(all_docs["documents"]):
            print("  Documento:", all_docs["documents"][i])
        print("-" * 40)

if __name__ == "__main__":
    ver_base_datos()