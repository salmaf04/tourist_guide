"""
Procesador de contenido usando LLM para extraer información turística.
"""

import json
from bs4 import BeautifulSoup
from .filters import ContentFilter


class ContentProcessor:
    """Clase para procesar contenido web y extraer información turística."""
    
    def __init__(self, llm_client, logger=None):
        self.llm_client = llm_client
        self.logger = logger
        self.content_filter = ContentFilter(logger)
    
    def extract_text_from_response(self, response):
        """Extrae texto limpio de la respuesta HTML."""
        try:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remover scripts, estilos y otros elementos no deseados
            for script in soup(["script", "style", "noscript"]):
                script.decompose()
            
            # Extraer texto
            raw_text = ' '.join(soup.stripped_strings)
            
            # Limitar longitud
            return raw_text[:6000]
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error extrayendo texto: {e}")
            return ""
    
    def create_llm_prompt(self, city, url, text):
        """Crea el prompt para el LLM."""
        prompt = (
            f"Analiza el siguiente texto de una página web turística y extrae información sobre lugares turísticos específicos de España. "
            f"La página es sobre {city} (si no es 'Unknown'), pero puede contener información de otras ciudades españolas también. "
            f"IMPORTANTE: Para cada lugar, identifica correctamente la ciudad española donde se encuentra. "
            f"Si el lugar está claramente en {city}, usa '{city}'. "
            f"Si está en otra ciudad española, usa el nombre correcto de esa ciudad. "
            f"Solo incluye lugares turísticos reales de España (museos, monumentos, parques, plazas, etc.). "
            f"NO incluyas lugares de Madrid bajo ninguna circunstancia. "
            f"Responde ÚNICAMENTE con un objeto JSON válido (sin bloques de código markdown). "
            f"Formato requerido: "
            f'{{"lugares": [{{"nombre": "nombre específico del lugar", "ciudad": "nombre correcto de la ciudad española", "categoria": "tipo de atracción", "descripcion": "descripción detallada", "coordenadas": null}}]}}\n\n'
            f"URL de origen: {url}\n"
            f"Texto: {text[:4000]}"
        )
        return prompt
    
    def clean_llm_response(self, response):
        """Limpia la respuesta del LLM para obtener JSON válido."""
        cleaned = response.strip()
        
        # Remover bloques de código markdown
        if cleaned.startswith('```json'):
            cleaned = cleaned[7:]
        elif cleaned.startswith('```'):
            cleaned = cleaned[3:]
        
        if cleaned.endswith('```'):
            cleaned = cleaned[:-3]
        
        cleaned = cleaned.strip()
        
        # Encontrar el JSON válido
        if not cleaned.startswith('{'):
            start = cleaned.find('{')
            end = cleaned.rfind('}')
            if start != -1 and end != -1 and end > start:
                cleaned = cleaned[start:end+1]
        
        return cleaned
    
    def process_llm_response(self, llm_response):
        """Procesa la respuesta del LLM y extrae los lugares."""
        try:
            cleaned_response = self.clean_llm_response(llm_response)
            
            if self.logger:
                self.logger.debug(f"Respuesta limpia del LLM: {cleaned_response[:200]}...")
            
            data = json.loads(cleaned_response)
            
            # Obtener lista de lugares
            lugares = data.get('lugares', [])
            
            # Si no hay lista, intentar como objeto único
            if not lugares and data.get('nombre'):
                lugares = [data]
            
            return lugares
            
        except json.JSONDecodeError as e:
            if self.logger:
                self.logger.error(f"Error decodificando JSON del LLM: {e}")
                self.logger.error(f"Respuesta problemática: {llm_response[:500]}...")
            return []
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error procesando respuesta LLM: {e}")
            return []
    
    def validate_and_filter_places(self, lugares):
        """Valida y filtra los lugares extraídos."""
        valid_places = []
        
        for lugar in lugares:
            if not lugar.get('nombre'):
                continue
            
            # Filtrar contenido de Madrid
            if self.content_filter.is_madrid_content(lugar):
                continue
            
            # Validar datos completos
            if not self.content_filter.validate_place_data(lugar):
                continue
            
            valid_places.append(lugar)
        
        return valid_places
    
    def process_page_content(self, response, city):
        """Procesa el contenido completo de una página."""
        try:
            # Extraer texto
            raw_text = self.extract_text_from_response(response)
            if not raw_text:
                if self.logger:
                    self.logger.warning(f"No se pudo extraer texto de {response.url}")
                return []
            
            # Crear prompt
            prompt = self.create_llm_prompt(city, response.url, raw_text)
            
            if self.logger:
                self.logger.info(f"Enviando texto al LLM para procesar. Longitud: {len(raw_text)} caracteres")
            
            # Procesar con LLM
            llm_response = self.llm_client.generate(prompt)
            
            if self.logger:
                self.logger.info(f"Respuesta del LLM recibida. Longitud: {len(llm_response)} caracteres")
            
            # Procesar respuesta
            lugares = self.process_llm_response(llm_response)
            
            if self.logger:
                self.logger.info(f"Lugares encontrados en {response.url}: {len(lugares)}")
            
            # Validar y filtrar
            valid_places = self.validate_and_filter_places(lugares)
            
            if self.logger:
                self.logger.info(f"Lugares válidos después del filtrado: {len(valid_places)}")
            
            return valid_places
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error procesando contenido de {response.url}: {e}")
            return []
    
    def save_places_to_file(self, lugares, filename="lugares_llm.json"):
        """Guarda los lugares en un archivo JSON."""
        try:
            with open(filename, "a", encoding="utf-8") as f:
                for lugar in lugares:
                    f.write(json.dumps(lugar, ensure_ascii=False) + "\n")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error guardando lugares en archivo: {e}")