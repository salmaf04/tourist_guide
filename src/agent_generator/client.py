from typing import Optional
from datetime import datetime
import time
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

class GeminiClient:
    """
    Cliente para interactuar con la API de Gemini.
    """
    def __init__(self, temperature=0.7):
        # Configure Gemini API
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        self.generation_config = {
            'temperature': temperature,
            'top_p': 1,
            'top_k': 32,
            'max_output_tokens': 2000,
        }
        self.max_retries = 12
        self.retry_delay = 5  # seconds

    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        """
        Genera una respuesta usando Gemini con reintentos por límite de tasa.
        """
        retries = 0
        while retries < self.max_retries:
            try:
                if system:
                    prompt = f"{system}\n{prompt}"
                
                safety_settings = [
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                ]
                
                response = self.model.generate_content(
                    prompt,
                    generation_config=self.generation_config,
                    safety_settings=safety_settings
                )
                
                if response.text:
                    return response.text.strip()
                return "[Respuesta no disponible]"
                
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    retries += 1
                    if retries < self.max_retries:
                        print(f"\nLímite de solicitudes alcanzado. Esperando {self.retry_delay} segundos...")
                        time.sleep(self.retry_delay)
                        continue
                print(f"Error con Gemini después de {retries} intentos: {e}")
                return "[Error de generación]"
            
        return "[Límite de solicitudes excedido]"
