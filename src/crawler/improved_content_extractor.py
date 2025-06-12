import spacy
from typing import List, Optional, Tuple, Dict, Set
from bs4 import BeautifulSoup, Tag
from dataclasses import dataclass
import re
import json
import os
from geopy.geocoders import Nominatim
import time
from urllib.parse import urlparse

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

class ImprovedContentExtractor:
    """Extractor de contenido mejorado para sitios turísticos"""
    
    # Selectores específicos para diferentes tipos de sitios
    SITE_SPECIFIC_SELECTORS = {
        'barcelonaturisme.com': {
            'places': ['.attraction-item', '.place-item', '.poi-item', 'article.attraction'],
            'names': ['h2', 'h3', '.attraction-title', '.place-name'],
            'descriptions': ['.description', '.summary', '.excerpt', 'p'],
            'skip_elements': ['.navigation', '.menu', '.breadcrumb', '.footer']
        },
        'lonelyplanet.com': {
            'places': ['.poi-list-item', '.attraction', '.place-card', 'article'],
            'names': ['h2', 'h3', '.poi-name', '.attraction-name'],
            'descriptions': ['.poi-description', '.description', 'p'],
            'skip_elements': ['.navigation', '.sidebar', '.ads']
        },
        'timeout.com': {
            'places': ['.listing-item', '.venue-item', '.attraction-item', 'article'],
            'names': ['h2', 'h3', '.listing-title', '.venue-name'],
            'descriptions': ['.listing-description', '.description', 'p'],
            'skip_elements': ['.navigation', '.sidebar', '.ads']
        },
        'visitbarcelona.com': {
            'places': ['.attraction', '.place', '.poi', 'article'],
            'names': ['h1', 'h2', 'h3', '.title'],
            'descriptions': ['.description', '.content', 'p'],
            'skip_elements': ['.navigation', '.menu', '.sidebar']
        },
        'tripadvisor.com': {
            'places': ['.attraction-element', '.poi-card', '.listing'],
            'names': ['.listing-title', 'h3', 'h2'],
            'descriptions': ['.listing-snippet', '.description'],
            'skip_elements': ['.navigation', '.ads', '.reviews']
        }
    }

    def __init__(self):
        # Cargar modelo de spaCy
        try:
            self.nlp = spacy.load("en_core_web_sm")
            print("✅ Using English spaCy model (en_core_web_sm)")
        except OSError:
            try:
                self.nlp = spacy.load("es_core_news_sm")
                print("⚠️ Using Spanish spaCy model (es_core_news_sm)")
            except OSError:
                print("❌ No spaCy model found. Please install: python -m spacy download en_core_web_sm")
                raise RuntimeError("No spaCy model available")
        
        self.geolocator = Nominatim(user_agent="tourism_crawler_v2")
        self.coordinates_cache = self._load_coordinates_cache()
        self.processed_places = set()  # Para evitar duplicados en la misma sesión

    def _load_coordinates_cache(self) -> Dict[str, Tuple[float, float]]:
        """Carga la caché de coordenadas"""
        cache_file = "coordinates_cache.json"
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading coordinates cache: {e}")
        return {}

    def _save_coordinates_cache(self):
        """Guarda la caché de coordenadas"""
        cache_file = "coordinates_cache.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump(self.coordinates_cache, f, indent=2)
        except Exception as e:
            print(f"Error saving coordinates cache: {e}")

    def extract_places(self, soup: BeautifulSoup, url: str, nltk_manager) -> List[TouristPlace]:
        """Extrae lugares turísticos de forma mejorada"""
        places = []
        
        # Limpiar HTML
        self._remove_unwanted_elements(soup)
        
        # Determinar el dominio para usar selectores específicos
        domain = urlparse(url).netloc.lower()
        site_config = self._get_site_config(domain)
        
        # Extraer ciudad de la URL
        city = self._extract_city_from_url(url)
        
        # Método 1: Usar selectores específicos del sitio
        if site_config:
            places.extend(self._extract_with_site_selectors(soup, url, city, site_config))
        
        # Método 2: Extracción genérica mejorada
        if len(places) < 3:  # Si no encontramos suficientes lugares
            places.extend(self._extract_generic_places(soup, url, city))
        
        # Método 3: Extracción basada en texto estructurado
        if len(places) < 2:
            places.extend(self._extract_from_structured_text(soup, url, city))
        
        # Filtrar duplicados y limpiar
        unique_places = self._filter_and_clean_places(places)
        
        # Enriquecer con información adicional
        for place in unique_places:
            place.named_entities = self._extract_named_entities(place.description)
            if not place.coordinates:
                place.coordinates = self._get_coordinates(place.name, place.city)
        
        return unique_places

    def _get_site_config(self, domain: str) -> Optional[Dict]:
        """Obtiene configuración específica para un dominio"""
        for site_domain, config in self.SITE_SPECIFIC_SELECTORS.items():
            if site_domain in domain:
                return config
        return None

    def _extract_with_site_selectors(self, soup: BeautifulSoup, url: str, city: str, config: Dict) -> List[TouristPlace]:
        """Extrae lugares usando selectores específicos del sitio"""
        places = []
        
        # Remover elementos que debemos saltar
        for skip_selector in config.get('skip_elements', []):
            for element in soup.select(skip_selector):
                element.decompose()
        
        # Buscar contenedores de lugares
        place_containers = []
        for selector in config.get('places', []):
            place_containers.extend(soup.select(selector))
        
        for container in place_containers:
            place = self._extract_place_from_container(container, url, city, config)
            if place and self._is_valid_place(place):
                places.append(place)
        
        return places

    def _extract_place_from_container(self, container: Tag, url: str, city: str, config: Dict) -> Optional[TouristPlace]:
        """Extrae un lugar de un contenedor específico"""
        # Extraer nombre
        name = None
        for name_selector in config.get('names', ['h1', 'h2', 'h3']):
            name_element = container.select_one(name_selector)
            if name_element:
                name = self._clean_place_name(name_element.get_text(strip=True))
                if name and len(name) > 2:
                    break
        
        if not name:
            return None
        
        # Extraer descripción
        description = ""
        for desc_selector in config.get('descriptions', ['p', '.description']):
            desc_elements = container.select(desc_selector)
            descriptions = []
            for elem in desc_elements[:3]:  # Máximo 3 párrafos
                text = elem.get_text(strip=True)
                if len(text) > 20 and text not in descriptions:
                    descriptions.append(text)
            if descriptions:
                description = " ".join(descriptions)
                break
        
        if not description:
            description = container.get_text(strip=True)[:300]
        
        # Clasificar y crear lugar
        category = self._classify_place_category(name + " " + description)
        
        return TouristPlace(
            name=name,
            city=city,
            category=category,
            description=description[:500],  # Limitar descripción
            visitor_appeal=self._extract_visitor_appeal(description),
            tourist_classification=self._classify_tourist_site(description),
            estimated_visit_duration=self._estimate_visit_duration(category, description),
            coordinates=None,  # Se añadirá después
            source_url=url,
            named_entities=""  # Se añadirá después
        )

    def _extract_generic_places(self, soup: BeautifulSoup, url: str, city: str) -> List[TouristPlace]:
        """Extracción genérica mejorada"""
        places = []
        
        # Buscar elementos que probablemente contengan lugares
        potential_elements = soup.select(
            'h1, h2, h3, h4, '
            'li:not(nav li):not(.menu li), '
            '.attraction, .place, .destination, .landmark, .poi, '
            '.card, .item, .listing, '
            'article, section'
        )
        
        for element in potential_elements:
            text = element.get_text(strip=True)
            
            # Filtros de calidad
            if not self._is_potential_place_name(text):
                continue
            
            # Verificar si es realmente un lugar turístico
            if not self._contains_tourism_keywords(text):
                continue
            
            # Extraer descripción del contexto
            description = self._extract_context_description(element)
            
            # Crear lugar
            category = self._classify_place_category(text + " " + description)
            
            place = TouristPlace(
                name=self._clean_place_name(text),
                city=city,
                category=category,
                description=description,
                visitor_appeal=self._extract_visitor_appeal(description),
                tourist_classification=self._classify_tourist_site(description),
                estimated_visit_duration=self._estimate_visit_duration(category, description),
                coordinates=None,
                source_url=url,
                named_entities=""
            )
            
            if self._is_valid_place(place):
                places.append(place)
        
        return places

    def _extract_from_structured_text(self, soup: BeautifulSoup, url: str, city: str) -> List[TouristPlace]:
        """Extrae lugares del texto estructurado usando NLP"""
        places = []
        
        # Obtener texto principal
        main_text = self._get_main_text(soup)
        if len(main_text) < 100:
            return places
        
        # Procesar con spaCy
        try:
            doc = self.nlp(main_text[:5000])  # Limitar para rendimiento
            
            # Buscar entidades que podrían ser lugares
            potential_places = []
            for ent in doc.ents:
                if ent.label_ in ["ORG", "LOC", "GPE", "PERSON"] and len(ent.text.strip()) > 3:
                    potential_places.append(ent.text.strip())
            
            # Crear lugares para entidades válidas
            for place_name in potential_places:
                if self._contains_tourism_keywords(place_name) or self._is_likely_attraction(place_name):
                    # Buscar contexto en el texto
                    context = self._find_text_context(main_text, place_name)
                    
                    category = self._classify_place_category(place_name + " " + context)
                    
                    place = TouristPlace(
                        name=place_name,
                        city=city,
                        category=category,
                        description=context[:300] if context else f"Tourist attraction in {city}",
                        visitor_appeal="Interesting tourist destination",
                        tourist_classification=self._classify_tourist_site(context),
                        estimated_visit_duration=self._estimate_visit_duration(category, context),
                        coordinates=None,
                        source_url=url,
                        named_entities=""
                    )
                    
                    if self._is_valid_place(place):
                        places.append(place)
        
        except Exception as e:
            print(f"Error in NLP processing: {e}")
        
        return places

    def _is_potential_place_name(self, text: str) -> bool:
        """Verifica si un texto podría ser un nombre de lugar"""
        if len(text) < 3 or len(text) > 100:
            return False
        
        # Filtrar elementos de navegación comunes
        navigation_terms = {
            'home', 'menu', 'search', 'contact', 'about', 'login', 'register',
            'inicio', 'menú', 'buscar', 'contacto', 'acerca', 'entrar', 'registrar',
            'click here', 'read more', 'see more', 'view all', 'load more',
            'haz clic', 'leer más', 'ver más', 'ver todo', 'cargar más'
        }
        
        if text.lower().strip() in navigation_terms:
            return False
        
        # Filtrar precios y números
        if re.match(r'^[\d\s€$£,.-]+$', text):
            return False
        
        return True

    def _contains_tourism_keywords(self, text: str) -> bool:
        """Verifica si el texto contiene palabras clave turísticas"""
        text_lower = text.lower()
        
        tourism_keywords = {
            # Inglés
            'museum', 'gallery', 'park', 'cathedral', 'church', 'palace', 'castle',
            'square', 'plaza', 'market', 'bridge', 'tower', 'monument', 'temple',
            'beach', 'garden', 'theater', 'theatre', 'stadium', 'arena', 'center',
            'centre', 'hall', 'house', 'building', 'site', 'attraction', 'landmark',
            # Español
            'museo', 'galería', 'parque', 'catedral', 'iglesia', 'palacio', 'castillo',
            'plaza', 'mercado', 'puente', 'torre', 'monumento', 'templo', 'playa',
            'jardín', 'teatro', 'estadio', 'centro', 'sala', 'casa', 'edificio',
            'sitio', 'atracción', 'lugar'
        }
        
        return any(keyword in text_lower for keyword in tourism_keywords)

    def _is_likely_attraction(self, text: str) -> bool:
        """Verifica si es probable que sea una atracción turística"""
        # Patrones comunes de atracciones
        attraction_patterns = [
            r'\b(museo|museum)\s+\w+',
            r'\b(palacio|palace)\s+\w+',
            r'\b(catedral|cathedral)\s+\w+',
            r'\b(parque|park)\s+\w+',
            r'\b(plaza|square)\s+\w+',
            r'\b(torre|tower)\s+\w+',
            r'\b(castillo|castle)\s+\w+',
            r'\b(iglesia|church)\s+\w+',
            r'\b(mercado|market)\s+\w+',
            r'\b(puente|bridge)\s+\w+',
        ]
        
        text_lower = text.lower()
        return any(re.search(pattern, text_lower) for pattern in attraction_patterns)

    def _extract_context_description(self, element: Tag) -> str:
        """Extrae descripción del contexto del elemento"""
        descriptions = []
        
        # Buscar en hermanos siguientes
        for sibling in element.find_next_siblings(['p', 'div', 'span'])[:3]:
            text = sibling.get_text(strip=True)
            if len(text) > 20 and not self._is_navigation_text(text):
                descriptions.append(text)
        
        # Buscar en el padre
        if element.parent and not descriptions:
            parent_text = element.parent.get_text(strip=True)
            if len(parent_text) > len(element.get_text(strip=True)) + 20:
                descriptions.append(parent_text[:300])
        
        return " ".join(descriptions[:2])

    def _find_text_context(self, text: str, place_name: str) -> str:
        """Encuentra contexto de un lugar en el texto"""
        # Buscar el lugar en el texto y extraer contexto
        place_index = text.lower().find(place_name.lower())
        if place_index == -1:
            return ""
        
        # Extraer contexto (100 caracteres antes y 200 después)
        start = max(0, place_index - 100)
        end = min(len(text), place_index + len(place_name) + 200)
        context = text[start:end].strip()
        
        return context

    def _is_navigation_text(self, text: str) -> bool:
        """Verifica si el texto es de navegación"""
        nav_indicators = [
            'click', 'here', 'more', 'view', 'see', 'read', 'next', 'previous',
            'haz clic', 'aquí', 'más', 'ver', 'leer', 'siguiente', 'anterior'
        ]
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in nav_indicators)

    def _clean_place_name(self, name: str) -> str:
        """Limpia el nombre del lugar"""
        if not name:
            return ""
        
        # Remover precios y símbolos
        name = re.sub(r'from\s*[\d€$£,.-]+', '', name, flags=re.IGNORECASE)
        name = re.sub(r'[\d€$£,.-]+\s*€', '', name)
        name = re.sub(r'^\s*[\d€$£,.-]+\s*', '', name)
        
        # Remover caracteres especiales al inicio/final
        name = re.sub(r'^[^\w\s]+|[^\w\s]+$', '', name)
        
        # Limpiar espacios múltiples
        name = re.sub(r'\s+', ' ', name).strip()
        
        return name

    def _filter_and_clean_places(self, places: List[TouristPlace]) -> List[TouristPlace]:
        """Filtra duplicados y limpia la lista de lugares"""
        seen_names = set()
        unique_places = []
        
        for place in places:
            # Normalizar nombre para comparación
            normalized_name = place.name.lower().strip()
            normalized_name = re.sub(r'\s+', ' ', normalized_name)
            
            # Verificar duplicados
            if normalized_name in seen_names or len(normalized_name) < 3:
                continue
            
            # Verificar que no sea una ciudad
            if self._is_city_name(place.name, place.city):
                continue
            
            seen_names.add(normalized_name)
            unique_places.append(place)
        
        return unique_places

    def _is_city_name(self, place_name: str, city: str) -> bool:
        """Verifica si el nombre del lugar es realmente una ciudad"""
        cities_regions = {
            'madrid', 'barcelona', 'valencia', 'sevilla', 'bilbao', 'granada',
            'toledo', 'salamanca', 'cataluña', 'catalunya', 'andalucía', 'andalucia',
            'país vasco', 'pais vasco', 'euskadi', 'españa', 'spain'
        }
        
        place_lower = place_name.lower().strip()
        return place_lower in cities_regions or place_lower == city.lower()

    def _is_valid_place(self, place: TouristPlace) -> bool:
        """Verifica si un lugar es válido"""
        if not place.name or len(place.name) < 3:
            return False
        
        if not place.description or len(place.description) < 10:
            return False
        
        # Verificar que no sea duplicado en esta sesión
        place_key = f"{place.name}:{place.city}".lower()
        if place_key in self.processed_places:
            return False
        
        self.processed_places.add(place_key)
        return True

    def _extract_city_from_url(self, url: str) -> str:
        """Extrae ciudad de la URL"""
        city_mapping = {
            'madrid': 'Madrid', 'barcelona': 'Barcelona', 'valencia': 'Valencia',
            'sevilla': 'Sevilla', 'seville': 'Sevilla', 'bilbao': 'Bilbao',
            'granada': 'Granada', 'toledo': 'Toledo', 'salamanca': 'Salamanca'
        }
        
        url_lower = url.lower()
        for key, city in city_mapping.items():
            if key in url_lower:
                return city
        
        return "Unknown"

    def _get_main_text(self, soup: BeautifulSoup) -> str:
        """Obtiene el texto principal de la página"""
        # Intentar selectores de contenido principal
        main_selectors = ['main', 'article', '.content', '#content', '.main-content']
        
        for selector in main_selectors:
            elements = soup.select(selector)
            if elements:
                return " ".join(el.get_text(strip=True) for el in elements)
        
        # Fallback: todo el body
        if soup.body:
            return soup.body.get_text(strip=True)
        
        return ""

    def _remove_unwanted_elements(self, soup: BeautifulSoup):
        """Remueve elementos no deseados"""
        unwanted_tags = [
            "script", "style", "header", "footer", "nav", "aside",
            "form", "button", "input", "select", "textarea",
            "iframe", "embed", "object", "video", "audio"
        ]
        
        for tag in unwanted_tags:
            for element in soup(tag):
                element.decompose()
        
        # Remover por clases
        unwanted_classes = [
            "nav", "navigation", "menu", "sidebar", "footer", "header",
            "ad", "advertisement", "banner", "popup", "modal", "cookie"
        ]
        
        for class_name in unwanted_classes:
            for element in soup.find_all(class_=re.compile(class_name, re.I)):
                element.decompose()

    def _classify_place_category(self, text: str) -> str:
        """Clasifica la categoría del lugar"""
        text_lower = text.lower()
        
        categories = {
            "engineering": ["bridge", "structure", "architecture", "tower", "puente", "torre", "estructura"],
            "history": ["historic", "history", "ancient", "heritage", "monument", "castle", "palace", 
                       "histórico", "historia", "antiguo", "patrimonio", "monumento", "castillo", "palacio"],
            "food": ["restaurant", "cuisine", "gastronomy", "food", "market", "cafe", "bar",
                    "restaurante", "cocina", "gastronomía", "comida", "mercado", "café"],
            "culture": ["culture", "cultural", "festival", "tradition", "museum", "gallery", "art",
                       "cultura", "cultural", "festival", "tradición", "museo", "galería", "arte"],
            "beach": ["beach", "coast", "sea", "shore", "playa", "costa", "mar"],
            "shopping": ["shop", "market", "shopping", "store", "mall", "tienda", "compras"],
            "nature": ["park", "nature", "garden", "landscape", "natural", "forest",
                      "parque", "naturaleza", "jardín", "paisaje", "natural", "bosque"]
        }
        
        for category, keywords in categories.items():
            if any(keyword in text_lower for keyword in keywords):
                return category
        
        return "general"

    def _extract_visitor_appeal(self, text: str) -> str:
        """Extrae atractivo para visitantes"""
        text_lower = text.lower()
        
        appeal_keywords = [
            "beautiful", "stunning", "magnificent", "historic", "cultural", "popular",
            "famous", "must-see", "recommended", "amazing", "spectacular", "unique",
            "hermoso", "impresionante", "magnífico", "histórico", "cultural", "popular",
            "famoso", "imprescindible", "recomendado", "increíble", "espectacular", "único"
        ]
        
        found_appeals = [kw for kw in appeal_keywords if kw in text_lower]
        
        if found_appeals:
            return f"Features: {', '.join(found_appeals[:3])}"
        return "Interesting tourist destination"

    def _classify_tourist_site(self, text: str) -> str:
        """Clasifica el tipo de sitio turístico"""
        text_lower = text.lower()
        
        classifications = {
            "museum": ["museum", "gallery", "exhibition", "museo", "galería", "exposición"],
            "historical": ["historic", "history", "ancient", "heritage", "monument", "castle",
                          "histórico", "historia", "antiguo", "patrimonio", "monumento", "castillo"],
            "cultural": ["culture", "cultural", "tradition", "festival", "cultura", "cultural", "tradición"],
            "natural": ["park", "nature", "garden", "landscape", "parque", "naturaleza", "jardín"],
            "religious": ["church", "cathedral", "temple", "religious", "iglesia", "catedral", "templo"],
            "entertainment": ["entertainment", "show", "theater", "cinema", "entretenimiento", "teatro", "cine"]
        }
        
        for classification, keywords in classifications.items():
            if any(keyword in text_lower for keyword in keywords):
                return classification
        
        return "general"

    def _estimate_visit_duration(self, category: str, description: str) -> str:
        """Estima duración de visita"""
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
        """Obtiene coordenadas con caché mejorado"""
        cache_key = f"{place_name}:{city}"
        if cache_key in self.coordinates_cache:
            return tuple(self.coordinates_cache[cache_key])
        
        try:
            # Intentar múltiples variaciones del nombre
            search_queries = [
                f"{place_name}, {city}, Spain",
                f"{place_name}, {city}",
                f"{place_name} {city}",
            ]
            
            # Si el nombre es muy genérico, no geocodificar
            if self._is_too_generic_for_geocoding(place_name):
                print(f"Skipping geocoding for generic name: {place_name}")
                return None
            
            for query in search_queries:
                try:
                    location = self.geolocator.geocode(query, timeout=10)
                    if location:
                        # Verificar que las coordenadas están en la región correcta
                        if self._is_in_expected_region(location.latitude, location.longitude, city):
                            coordinates = (location.latitude, location.longitude)
                            self.coordinates_cache[cache_key] = list(coordinates)
                            self._save_coordinates_cache()
                            print(f"Geocoded {place_name}: {coordinates}")
                            return coordinates
                        else:
                            print(f"Coordinates for {place_name} outside expected region")
                except Exception as e:
                    print(f"Error with query '{query}': {e}")
                    continue
            
            # Si no se encontró nada específico, NO usar fallback de ciudad
            print(f"No specific coordinates found for {place_name}")
            return None
            
        except Exception as e:
            print(f"Error geocoding {place_name}, {city}: {e}")
        
        return None

    def _is_too_generic_for_geocoding(self, place_name: str) -> bool:
        """Verifica si un nombre es demasiado genérico para geocodificar"""
        generic_terms = {
            'beaches', 'museums', 'attractions', 'places', 'sites', 'locations',
            'things to do', 'what to do', 'guide', 'best', 'top', 'ultimate',
            'free access', 'contact', 'nearby', 'open-air', 'exhibition centers',
            'playas', 'museos', 'atracciones', 'lugares', 'sitios', 'qué hacer',
            'guía', 'mejor', 'mejores', 'acceso gratuito', 'contacto', 'cerca'
        }
        
        place_lower = place_name.lower()
        
        # Si el nombre es muy corto
        if len(place_name) < 4:
            return True
        
        # Si contiene términos genéricos
        if any(term in place_lower for term in generic_terms):
            return True
        
        # Si es principalmente números o símbolos
        if len(re.sub(r'[^a-zA-Z]', '', place_name)) < 3:
            return True
        
        return False

    def _is_in_expected_region(self, lat: float, lon: float, city: str) -> bool:
        """Verifica si las coordenadas están en la región esperada"""
        # Definir bounding boxes para ciudades principales
        city_bounds = {
            'Barcelona': {'min_lat': 41.32, 'max_lat': 41.47, 'min_lon': 2.05, 'max_lon': 2.25},
            'Madrid': {'min_lat': 40.35, 'max_lat': 40.50, 'min_lon': -3.80, 'max_lon': -3.60},
            'Valencia': {'min_lat': 39.40, 'max_lat': 39.55, 'min_lon': -0.45, 'max_lon': -0.30},
            'Sevilla': {'min_lat': 37.30, 'max_lat': 37.45, 'min_lon': -6.05, 'max_lon': -5.90},
            'Bilbao': {'min_lat': 43.20, 'max_lat': 43.35, 'min_lon': -2.95, 'max_lon': -2.85},
            'Granada': {'min_lat': 37.15, 'max_lat': 37.20, 'min_lon': -3.65, 'max_lon': -3.55},
        }
        
        bounds = city_bounds.get(city)
        if not bounds:
            return True  # Si no conocemos la ciudad, aceptar las coordenadas
        
        return (bounds['min_lat'] <= lat <= bounds['max_lat'] and 
                bounds['min_lon'] <= lon <= bounds['max_lon'])

    def _extract_named_entities(self, text: str) -> str:
        """Extrae entidades nombradas del texto"""
        try:
            doc = self.nlp(text[:1000])
            entities = []
            
            for ent in doc.ents:
                if ent.label_ in ["LOC", "GPE", "ORG", "PERSON"] and len(ent.text.strip()) > 2:
                    entities.append(f"{ent.label_}: {ent.text.strip()}")
            
            return ", ".join(entities[:10])  # Limitar a 10 entidades
        except Exception as e:
            print(f"Error extracting entities: {e}")
            return ""