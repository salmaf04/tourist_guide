from typing import Optional
from datetime import datetime
import time
import os
from dotenv import load_dotenv
from mistralai import Mistral
from utils.gemini_api_counter import count_gemini_calls

# Load environment variables
load_dotenv()

class MistralClient:
    """
    Cliente para interactuar con la API de Mistral.
    """
    def __init__(self, temperature=0.7):
        # Configure Mistral API
        api_key = os.getenv('MISTRAL_API_KEY')
        if not api_key:
            raise ValueError("MISTRAL_API_KEY not found in environment variables.")
        
        self.client = Mistral(api_key=api_key)
        self.model = "mistral-large-latest"
        self.temperature = temperature
        self.max_retries = 12
        self.retry_delay = 5  # seconds

    def make_request_with_retry(self, messages):
        """Función para manejar solicitudes con reintentos (por límite de 1 solicitud/segundo)"""
        for attempt in range(self.max_retries):
            try:
                return self.client.chat.complete(
                    model=self.model, 
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=2000
                )
            except Exception as e:
                if "429" in str(e):  # Error de límite de solicitudes
                    if attempt < self.max_retries - 1:
                        print(f"\nLímite de solicitudes alcanzado. Esperando {self.retry_delay} segundos... (intento {attempt + 1}/{self.max_retries})")
                        time.sleep(self.retry_delay)
                        continue
                    else:
                        print(f"Error con Mistral después de {self.max_retries} intentos: {e}")
                        raise e
                else:
                    print(f"Error con Mistral en intento {attempt + 1}: {e}")
                    raise e
        raise Exception("Max retries exceeded")

    @count_gemini_calls  # Reutilizamos el contador existente para mantener compatibilidad
    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        """
        Genera una respuesta usando Mistral con reintentos por límite de tasa.
        """
        try:
            # Preparar mensajes para Mistral
            messages = []
            
            if system:
                messages.append({"role": "system", "content": system})
            
            messages.append({"role": "user", "content": prompt})
            
            # Ejecutar la solicitud con reintentos
            response = self.make_request_with_retry(messages)
            
            if response and response.choices and len(response.choices) > 0:
                return response.choices[0].message.content.strip()
            else:
                return "[Respuesta no disponible]"
                
        except Exception as e:
            print(f"Error con Mistral: {e}")
            return "[Error de generación]"