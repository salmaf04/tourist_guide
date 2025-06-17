from agent_generator.client import GeminiClient
import json
import google.generativeai as genai

# Simula un lugar extraído por el crawler
place = {
    "name": "Museo del Prado",
    "city": "Madrid",
    "description": "Uno de los museos de arte más importantes del mundo.",
    "category": "museum"
}

prompt = (
    f"Extrae y devuelve en JSON los siguientes campos del lugar turístico:\n"
    f"nombre, ciudad, descripcion, categoria.\n"
    f"Datos:\n"
    f"Nombre: {place['name']}\n"
    f"Ciudad: {place['city']}\n"
    f"Descripción: {place['description']}\n"
    f"Categoría: {place['category']}\n"
    f"Devuelve solo el JSON."
)

GEMINI_API_KEY = 'AIzaSyCkN0mxdFQpGajEwB8sZm2fUsJzhpTCfvk'  
genai.configure(api_key=GEMINI_API_KEY)
llm = GeminiClient()
respuesta = llm.generate(prompt)
print("Respuesta cruda del LLM:\n", respuesta)

try:
    data = json.loads(respuesta)
    print("\nJSON parseado:")
    print(json.dumps(data, ensure_ascii=False, indent=2))
except Exception as e:
    print("\nError al parsear la respuesta del LLM como JSON:", e)
