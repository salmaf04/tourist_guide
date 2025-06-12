import spacy
from typing import List, Optional, Tuple, Dict
from bs4 import BeautifulSoup
from dataclasses import dataclass
import re
import json
import os
from geopy.geocoders import Nominatim

@dataclass
class TouristPlace:
    name: str
    city: str
    category: str
    description: str
    visitor_appeal: str
    tourist_classification: str
    estimated_visit_duration: str
    coordinates: Optional[Tuple[float, float]]
    source_url: str
    named_entities: str

class ContentExtractor:
    MAIN_CONTENT_SELECTORS = [
        'article', 'main', '.content', '#content',
        '.main-content', '#main-content', 'div[role="main"]',
        '.attraction-info', '.place-info', '.destination-content'
    ]

    def __init__(self):
        # Cargar el modelo de spaCy - priorizar inglés para mejor compatibilidad
        try:
            self.nlp = spacy.load("en_core_web_sm")
            print("✅ Using English spaCy model (en_core_web_sm)")
            self.language = "en"
        except OSError:
            try:
                self.nlp = spacy.load("en_core_web_md")
                print("✅ Using English spaCy model (en_core_web_md)")
                self.language = "en"
            except OSError:
                try:
                    self.nlp = spacy.load("es_core_news_sm")
                    print("⚠️ Using Spanish spaCy model (es_core_news_sm)")
                    self.language = "es"
                except OSError:
                    print("❌ No spaCy model found. Please install: python -m spacy download en_core_web_sm")
                    raise RuntimeError("No spaCy model available. Install with: python -m spacy download en_core_web_sm")
        
        self.geolocator = Nominatim(user_agent="tourism_crawler")
        self.coordinates_cache = self._load_coordinates_cache()

    def _load_coordinates_cache(self) -> Dict[str, Tuple[float, float]]:
        """Carga la caché de coordenadas desde un archivo JSON."""
        cache_file = "coordinates_cache.json"
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading coordinates cache: {e}")
                return {}
        return {}

    def _save_coordinates_cache(self):
        """Guarda la caché de coordenadas en un archivo JSON."""
        cache_file = "coordinates_cache.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump(self.coordinates_cache, f, indent=2)
        except Exception as e:
            print(f"Error saving coordinates cache: {e}")

    def extract_places(self, soup: BeautifulSoup, url: str, nltk_manager) -> List[TouristPlace]:
        """Extrae lugares turísticos estructurados de una página HTML usando spaCy."""
        places = []
        self._remove_unwanted_elements(soup)

        # Extraer contenido principal
        texts = self._extract_main_content(soup)
        full_text = self._clean_text("\n".join(filter(None, texts)))

        if not full_text or len(full_text) < 50:
            print(f"⚠️ Insufficient content extracted from {url}")
            return places

        # Procesar texto con spaCy
        try:
            doc = self.nlp(full_text[:1000000])  # Limitar texto para evitar problemas de memoria
        except Exception as e:
            print(f"Error processing text with spaCy: {e}")
            return places

        # Extraer entidades nombradas (lugares y organizaciones)
        named_entities = []
        cities = []
        
        for ent in doc.ents:
            if ent.label_ in ["LOC", "GPE", "PERSON", "ORG"]:  # Ubicaciones, personas y organizaciones
                entity_text = ent.text.strip()
                if len(entity_text) > 2:  # Filtrar entidades muy cortas
                    named_entities.append({"label": ent.label_, "text": entity_text})
                    
                    # Identificar ciudades conocidas
                    if ent.label_ in ["LOC", "GPE"]:
                        known_cities = nltk_manager._extract_known_cities(full_text)
                        if entity_text in known_cities:
                            cities.append(entity_text)

        # Determinar ciudad principal
        city = cities[0] if cities else self._extract_city_from_url(url)
        entities_str = ", ".join([f"{e['label']}: {e['text']}" for e in named_entities[:10]])  # Limitar entidades

        # Identificar lugares turísticos usando selectores mejorados
        potential_places = soup.select(
            'h1, h2, h3, h4, '
            'li, '
            '.attraction, .place, .destination, .landmark, .poi, '
            '.attraction-name, .place-name, .site-name, '
            '[class*="attraction"], [class*="place"], [class*="destination"]'
        )

        extracted_places = set()  # Para evitar duplicados

        for element in potential_places:
            place_name = element.get_text(strip=True)
            
            # Filtros de calidad para nombres de lugares
            if (len(place_name) < 3 or len(place_name) > 150 or
                place_name.lower() in ['home', 'menu', 'search', 'contact', 'about'] or
                place_name in extracted_places):
                continue

            # Procesar el texto del elemento con spaCy para confirmar si es un lugar
            try:
                place_doc = self.nlp(place_name)
                is_valid_place = False
                
                for ent in place_doc.ents:
                    if ent.label_ in ["LOC", "GPE", "ORG", "PERSON"]:
                        is_valid_place = True
                        place_name = ent.text.strip()
                        break
                
                # Si no se detecta como entidad, verificar si contiene palabras clave turísticas
                if not is_valid_place:
                    tourism_keywords = [
                        'museum', 'gallery', 'park', 'cathedral', 'church', 'palace', 'castle',
                        'square', 'plaza', 'market', 'bridge', 'tower', 'monument', 'temple',
                        'museo', 'galería', 'parque', 'catedral', 'iglesia', 'palacio', 'castillo',
                        'plaza', 'mercado', 'puente', 'torre', 'monumento', 'templo'
                    ]
                    
                    if any(keyword in place_name.lower() for keyword in tourism_keywords):
                        is_valid_place = True

                if not is_valid_place:
                    continue

            except Exception as e:
                print(f"Error processing place name with spaCy: {e}")
                continue

            # Extraer descripción (párrafos cercanos)
            description = self._extract_description(element)
            
            if not description:
                description = full_text[:300]  # Usar texto general como fallback

            # Clasificar categoría
            category = self._classify_place_category(description)

            # Obtener coordenadas
            coordinates = self._get_coordinates(place_name, city)

            place = TouristPlace(
                name=place_name,
                city=city,
                category=category,
                description=description,
                visitor_appeal=self._extract_visitor_appeal(description),
                tourist_classification=self._classify_tourist_site(description, entities_str),
                estimated_visit_duration=self._estimate_visit_duration(category, description),
                coordinates=coordinates,
                source_url=url,
                named_entities=entities_str
            )
            
            places.append(place)
            extracted_places.add(place_name)

        # Fallback: crear un lugar genérico si no se encontraron lugares específicos
        if not places:
            generic_name = self._extract_place_name(full_text, url)
            place = TouristPlace(
                name=generic_name,
                city=city,
                category="general",
                description=full_text[:500] + "..." if len(full_text) > 500 else full_text,
                visitor_appeal=self._extract_visitor_appeal(full_text),
                tourist_classification="general",
                estimated_visit_duration="2 hours",
                coordinates=self._get_coordinates(city, city),
                source_url=url,
                named_entities=entities_str
            )
            places.append(place)

        return places

    def _extract_city_from_url(self, url: str) -> str:
        """Extrae el nombre de la ciudad desde la URL."""
        # Mapeo de ciudades conocidas
        city_mapping = {
            'madrid': 'Madrid',
            'barcelona': 'Barcelona', 
            'valencia': 'Valencia',
            'sevilla': 'Sevilla',
            'seville': 'Sevilla',
            'bilbao': 'Bilbao',
            'granada': 'Granada',
            'toledo': 'Toledo',
            'salamanca': 'Salamanca'
        }
        
        url_lower = url.lower()
        for key, city in city_mapping.items():
            if key in url_lower:
                return city
        
        return "Unknown"

    def _extract_description(self, element) -> str:
        """Extrae descripción de un elemento y sus elementos cercanos."""
        descriptions = []
        
        # Buscar párrafos siguientes
        next_sibling = element.find_next('p')
        if next_sibling:
            desc = next_sibling.get_text(strip=True)
            if len(desc) > 20:
                descriptions.append(desc)
        
        # Buscar divs con descripción
        parent = element.parent
        if parent:
            desc_divs = parent.find_all(['div', 'span'], class_=re.compile(r'desc|info|detail'))
            for div in desc_divs:
                desc = div.get_text(strip=True)
                if len(desc) > 20:
                    descriptions.append(desc)
        
        return " ".join(descriptions[:2])  # Máximo 2 descripciones

    def _classify_place_category(self, text: str) -> str:
        """Clasifica un lugar en una categoría usando palabras clave multiidioma."""
        text_lower = text.lower()
        
        # Palabras clave en inglés y español
        categories = {
            "engineering": [
                "bridge", "structure", "architecture", "engineering", "building", "tower",
                "puente", "estructura", "arquitectura", "ingeniería", "edificio", "torre"
            ],
            "history": [
                "historic", "history", "ancient", "heritage", "monument", "castle", "palace",
                "histórico", "historia", "antiguo", "patrimonio", "monumento", "castillo", "palacio"
            ],
            "food": [
                "restaurant", "cuisine", "gastronomy", "food", "market", "cafe", "bar",
                "restaurante", "cocina", "gastronomía", "comida", "mercado", "café"
            ],
            "culture": [
                "culture", "cultural", "festival", "tradition", "museum", "gallery", "art",
                "cultura", "cultural", "festival", "tradición", "museo", "galería", "arte"
            ],
            "beach": [
                "beach", "coast", "sea", "shore", "waterfront",
                "playa", "costa", "mar", "orilla"
            ],
            "shopping": [
                "shop", "market", "shopping", "store", "mall",
                "tienda", "mercado", "compras", "centro comercial"
            ],
            "nature": [
                "park", "nature", "garden", "landscape", "natural", "forest", "mountain",
                "parque", "naturaleza", "jardín", "paisaje", "natural", "bosque", "montaña"
            ]
        }

        for category, keywords in categories.items():
            if any(keyword in text_lower for keyword in keywords):
                return category
        
        return "general"

    def _extract_place_name(self, text: str, url: str) -> str:
        """Extrae un nombre de lugar del texto o URL."""
        try:
            doc = self.nlp(text[:500])  # Procesar solo los primeros 500 caracteres
            for ent in doc.ents:
                if ent.label_ in ["LOC", "GPE", "ORG"] and len(ent.text.strip()) > 3:
                    return ent.text.strip()
        except Exception:
            pass
        
        # Fallback: extraer de URL
        url_parts = url.split('/')
        for part in reversed(url_parts):
            if part and part not in ['www', 'http:', 'https:', '', 'es', 'en']:
                clean_name = part.replace('-', ' ').replace('_', ' ').title()
                if len(clean_name) > 3:
                    return clean_name
        
        return "Tourist Site"

    def _extract_visitor_appeal(self, text: str) -> str:
        """Extrae información sobre el atractivo para visitantes."""
        text_lower = text.lower()
        
        # Palabras clave de atractivo en inglés y español
        appeal_keywords = [
            "beautiful", "stunning", "magnificent", "historic", "cultural", "popular", "famous",
            "attraction", "must-see", "recommended", "amazing", "spectacular", "unique",
            "hermoso", "impresionante", "magnífico", "histórico", "cultural", "popular", 
            "famoso", "atracción", "imprescindible", "recomendado", "increíble", "espectacular", "único"
        ]
        
        found_appeals = [keyword for keyword in appeal_keywords if keyword in text_lower]
        
        if found_appeals:
            return f"Features: {', '.join(found_appeals[:3])}"
        else:
            return "Interesting tourist destination"

    def _classify_tourist_site(self, text: str, named_entities: str) -> str:
        """Clasifica el tipo de sitio turístico."""
        text_lower = text.lower()
        entities_lower = named_entities.lower()
        
        classifications = {
            "museum": [
                "museum", "gallery", "exhibition", "art", "collection",
                "museo", "galería", "exposición", "arte", "colección"
            ],
            "historical": [
                "historic", "history", "ancient", "heritage", "monument", "castle", "palace",
                "histórico", "historia", "antiguo", "patrimonio", "monumento", "castillo", "palacio"
            ],
            "cultural": [
                "culture", "cultural", "tradition", "festival", "event", "theater",
                "cultura", "cultural", "tradición", "festival", "evento", "teatro"
            ],
            "natural": [
                "park", "nature", "garden", "landscape", "natural", "forest",
                "parque", "naturaleza", "jardín", "paisaje", "natural", "bosque"
            ],
            "religious": [
                "church", "cathedral", "temple", "religious", "sacred", "monastery",
                "iglesia", "catedral", "templo", "religioso", "sagrado", "monasterio"
            ],
            "entertainment": [
                "entertainment", "show", "theater", "cinema", "fun", "amusement",
                "entretenimiento", "espectáculo", "teatro", "cine", "diversión"
            ]
        }
        
        for classification, keywords in classifications.items():
            if (any(keyword in text_lower for keyword in keywords) or 
                any(keyword in entities_lower for keyword in keywords)):
                return classification
        
        return "general"

    def _estimate_visit_duration(self, category: str, description: str) -> str:
        """Estima la duración de visita basada en la categoría y descripción."""
        duration_map = {
            "museum": "2-3 hours",
            "historical": "1-2 hours", 
            "cultural": "1-2 hours",
            "natural": "2-4 hours",
            "religious": "30 minutes - 1 hour",
            "entertainment": "2-3 hours",
            "food": "1-2 hours",
            "shopping": "1-3 hours",
            "beach": "2-6 hours",
            "engineering": "30 minutes - 1 hour"
        }
        
        return duration_map.get(category, "1-2 hours")

    def _get_coordinates(self, place_name: str, city: str) -> Optional[Tuple[float, float]]:
        """Obtiene coordenadas usando geopy con caché persistente."""
        cache_key = f"{place_name}:{city}"
        if cache_key in self.coordinates_cache:
            return tuple(self.coordinates_cache[cache_key])

        try:
            # Intentar con nombre completo primero
            location = self.geolocator.geocode(f"{place_name}, {city}", timeout=10)
            if not location and city != "Unknown":
                # Fallback: solo la ciudad
                location = self.geolocator.geocode(city, timeout=10)
            
            if location:
                coordinates = (location.latitude, location.longitude)
                self.coordinates_cache[cache_key] = list(coordinates)
                self._save_coordinates_cache()
                return coordinates
        except Exception as e:
            print(f"Error geocoding {place_name}, {city}: {e}")
        
        return None

    def _remove_unwanted_elements(self, soup: BeautifulSoup):
        """Elimina elementos no deseados del árbol DOM."""
        unwanted_tags = [
            "script", "style", "header", "footer", "nav", "aside", 
            "form", "button", "input", "select", "textarea",
            "iframe", "embed", "object", "video", "audio"
        ]
        
        for tag in unwanted_tags:
            for element in soup(tag):
                element.decompose()
        
        # Eliminar elementos con clases comunes de navegación/publicidad
        unwanted_classes = [
            "nav", "navigation", "menu", "sidebar", "footer", "header",
            "ad", "advertisement", "banner", "popup", "modal"
        ]
        
        for class_name in unwanted_classes:
            for element in soup.find_all(class_=re.compile(class_name, re.I)):
                element.decompose()

    def _extract_main_content(self, soup: BeautifulSoup) -> List[str]:
        """Intenta extraer el contenido principal usando selectores mejorados."""
        texts = []

        # Intentar selectores específicos primero
        for selector in self.MAIN_CONTENT_SELECTORS:
            main_elements = soup.select(selector)
            if main_elements:
                for el in main_elements:
                    text = el.get_text(separator=' ', strip=True)
                    if len(text) > 100:  # Solo textos sustanciales
                        texts.append(text)
                if texts:
                    return texts

        # Fallback: párrafos con contenido sustancial
        paragraphs = soup.find_all('p')
        substantial_texts = []
        
        for p in paragraphs:
            text = p.get_text(strip=True)
            if len(text) > 50:  # Solo párrafos con contenido
                substantial_texts.append(text)
        
        if substantial_texts:
            return substantial_texts
        
        # Último recurso: todo el body
        if soup.body:
            body_text = soup.body.get_text(separator=' ', strip=True)
            if len(body_text) > 100:
                return [body_text]

        return texts

    def _clean_text(self, text: str) -> str:
        """Limpia y normaliza el texto extraído."""
        if not text:
            return ""
        
        # Eliminar múltiples espacios en blanco y saltos de línea
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n', text)
        
        # Eliminar caracteres especiales problemáticos
        text = re.sub(r'[^\w\s\.,;:!?()-]', '', text)
        
        return text.strip()