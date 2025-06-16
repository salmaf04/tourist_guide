from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from bs4 import BeautifulSoup, Tag
import re
import json
import os
import spacy
import logging
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
from urllib.parse import urlparse
from sentence_transformers import SentenceTransformer
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TouristPlace:
    name: str
    city: str
    category: str
    description: str
    coordinates: Optional[Tuple[float, float]]

class ContentExtractor:
    def __init__(self):
        self.nlp = self._load_spacy_model()
        self.geolocator = Nominatim(user_agent="tourism_crawler_v4")
        self.coordinates_cache = self._load_coordinates_cache()
        self.processed_places = set()
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.last_geocode_time = 0
        self.min_interval = 1.0  # 1 segundo entre solicitudes

    def _load_spacy_model(self):
        try:
            return spacy.load("es_core_news_md")
        except OSError:
            logger.warning("Falling back to English spaCy model")
            return spacy.load("en_core_web_md")

    def extract_places(self, soup: BeautifulSoup, url: str, city: str = "Unknown") -> List[TouristPlace]:
        self._clean_html(soup)
        extracted_city = self._extract_city_name(soup, url) or city
        places = []

        extraction_strategies = [
            self._extract_structured_data,
            self._extract_schema_city,
            self._extract_tourism_blocks,
            self._extract_titles
        ]

        for strategy in extraction_strategies:
            try:
                new_places = strategy(soup, extracted_city, url)
                places.extend(new_places)
                if len(places) >= 5:
                    break
            except Exception as e:
                logger.warning(f"Error in strategy {strategy.__name__}: {e}")

        unique_places = self._filter_duplicates(places)
        return unique_places

    def _clean_html(self, soup: BeautifulSoup):
        for tag in ['script', 'style', 'nav', 'footer', 'header', 'iframe', 'form']:
            for element in soup.find_all(tag):
                element.decompose()
        for class_name in ['nav', 'menu', 'sidebar', 'ad', 'banner', 'cookie']:
            for element in soup.find_all(class_=re.compile(class_name, re.I)):
                element.decompose()

    def _extract_city_name(self, soup: BeautifulSoup, url: str) -> Optional[str]:
        city_mapping = {
            'madrid': 'Madrid', 'barcelona': 'Barcelona', 'valencia': 'Valencia',
            'sevilla': 'Sevilla', 'bilbao': 'Bilbao', 'granada': 'Granada',
            'toledo': 'Toledo', 'salamanca': 'Salamanca', 'malaga': 'Málaga'
        }
        url_lower = url.lower()
        for key, city in city_mapping.items():
            if key in url_lower:
                return city
        main_text = soup.get_text().lower()
        for key, city in city_mapping.items():
            if key in main_text:
                return city
        return None

    def _extract_structured_data(self, soup: BeautifulSoup, city: str, url: str) -> List[TouristPlace]:
        places = []
        for item in soup.select('[itemtype*="schema.org/Place"], [itemtype*="schema.org/TouristAttraction"]'):
            name = self._get_structured_property(item, 'name')
            if not name:
                continue
            description = self._get_structured_property(item, 'description') or f"Atracción en {city}"
            address = self._get_structured_property(item, 'address')
 
            place = TouristPlace(
                name=name,
                city=city,
                category=self._classify_category(name, description),
                description=description,
 
                coordinates=self._get_coordinates(name, city, address),

            )
            places.append(place)
        return places

    def _extract_schema_city(self, soup: BeautifulSoup, city: str, url: str) -> List[TouristPlace]:
        places = []
        for script in soup.select('script[type="application/ld+json"]'):
            try:
                data = json.loads(script.string)
                if isinstance(data, dict) and data.get('@type') in ['Place', 'TouristAttraction']:
                    place = self._parse_schema_item(data, url, city)
                    if place:
                        places.append(place)
                elif isinstance(data.get('@graph'), list):
                    for item in data['@graph']:
                        if item.get('@type') in ['Place', 'TouristAttraction']:
                            place = self._parse_schema_item(item, url, city)
                            if place:
                                places.append(place)
            except Exception as e:
                logger.warning(f"Error processing JSON-LD: {e}")
        return places

    def _parse_schema_item(self, data: Dict, url: str, city: str) -> Optional[TouristPlace]:
        name = data.get('name')
        if not name:
            return None
        description = data.get('description') or f"Atracción en {city}"
        coordinates = None
        if data.get('geo'):
            geo = data['geo']
            if isinstance(geo, dict):
                lat = geo.get('latitude')
                lon = geo.get('longitude')
                if lat and lon:
                    coordinates = (float(lat), float(lon))
        return TouristPlace(
            name=name,
            city=city,
            category=self._classify_category(name, description),
            description=description,

            coordinates=coordinates or self._get_coordinates(name, city, address),
        )

    def _extract_tourism_blocks(self, soup: BeautifulSoup, city: str, url: str) -> List[TouristPlace]:
        places = []
        # Expanded selectors for different website structures
        selectors = [
            '.attraction, .poi, .landmark, .place, .card, .item, [class*="tourism"]',
            '.content-item, .listing-item, .result-item',
            '[class*="attraction"], [class*="place"], [class*="destination"]',
            '.entry, .post, .article-item',
            'section[class*="content"], div[class*="content"]'
        ]
        
        for selector in selectors:
            containers = soup.select(selector)
            for container in containers:
                place = self._extract_from_container(container, url, city)
                if place:
                    places.append(place)
            if places:  # If we found places with this selector, stop trying others
                break
                
        return places

    def _extract_from_container(self, container: Tag, url: str, city: str) -> Optional[TouristPlace]:
        name = None
        for tag in ['h1', 'h2', 'h3']:
            name_element = container.find(tag)
            if name_element and self._is_valid_name(name_element.get_text()):
                name = self._clean_text(name_element.get_text())
                break
        if not name:
            return None
        description = ""
        for p in container.find_all('p', limit=2):
            text = self._clean_text(p.get_text())
            if len(text) > 30:
                description += text + " "
        
        return TouristPlace(
            name=name,
            city=city,
            category=self._classify_category(name, description),
            description=description.strip(),

            coordinates=self._get_coordinates(name, city),

        )

    def _extract_titles(self, soup: BeautifulSoup, city: str, url: str) -> List[TouristPlace]:
        places = []
        processed_names = set()  # Avoid duplicates
        
        for heading in soup.find_all(['h1', 'h2', 'h3', 'h4']):
            text = self._clean_text(heading.get_text())
            if not self._is_valid_name(text) or text.lower() in processed_names:
                continue
                
            processed_names.add(text.lower())
            
            # Try to find description in multiple ways
            description = ""
            
            # Method 1: Next sibling
            next_sib = heading.find_next_sibling()
            if next_sib and next_sib.name in ['p', 'div', 'span']:
                desc_text = self._clean_text(next_sib.get_text())
                if len(desc_text) > 20:
                    description = desc_text[:300]
            
            # Method 2: Parent container description
            if not description:
                parent = heading.parent
                if parent:
                    for p in parent.find_all('p', limit=2):
                        desc_text = self._clean_text(p.get_text())
                        if len(desc_text) > 20 and desc_text != text:
                            description = desc_text[:300]
                            break
            
            # Method 3: Following paragraphs
            if not description:
                current = heading
                for _ in range(3):  # Look at next 3 elements
                    current = current.find_next()
                    if current and current.name == 'p':
                        desc_text = self._clean_text(current.get_text())
                        if len(desc_text) > 20:
                            description = desc_text[:300]
                            break
            
            place = TouristPlace(
                name=text,
                city=city,
                category=self._classify_category(text, description),
                description=description or f"Atracción turística en {city}",
 

            )
            places.append(place)
        return places

    def _is_valid_name(self, text: str) -> bool:
        text = text.strip()
        if len(text) < 3 or len(text) > 100 or len(text.split()) > 12:
            return False
            
        # Skip obvious navigation/UI elements
        nav_terms = ['home', 'menu', 'search', 'contact', 'about', 'login', 'click here', 'read more', 
                    'more info', 'see more', 'view all', 'back to', 'next', 'previous', 'share', 'print']
        text_lower = text.lower()
        if any(term in text_lower for term in nav_terms):
            return False
            
        # Skip generic section headers
        generic_terms = ['overview', 'introduction', 'welcome', 'getting there', 'practical information',
                        'useful information', 'tips', 'advice', 'recommendations']
        if text_lower in generic_terms:
            return False
            
        # Tourism-related terms (expanded list)
        tourism_terms = [
            'museum', 'park', 'cathedral', 'palace', 'square', 'beach', 'museo', 'parque', 'catedral',
            'church', 'temple', 'monastery', 'castle', 'fortress', 'tower', 'bridge', 'garden',
            'gallery', 'theater', 'theatre', 'market', 'plaza', 'avenue', 'street', 'district',
            'neighborhood', 'quarter', 'center', 'centre', 'building', 'monument', 'memorial',
            'fountain', 'statue', 'viewpoint', 'lookout', 'observatory', 'zoo', 'aquarium',
            'restaurant', 'cafe', 'bar', 'shop', 'store', 'mall', 'shopping', 'hotel',
            'iglesia', 'templo', 'monasterio', 'castillo', 'fortaleza', 'torre', 'puente',
            'jardín', 'galería', 'teatro', 'mercado', 'avenida', 'calle', 'barrio', 'centro',
            'edificio', 'monumento', 'fuente', 'estatua', 'mirador', 'restaurante'
        ]
        
        has_tourism_term = any(term in text_lower for term in tourism_terms)
        
        # Check if it looks like a proper name (starts with capital, reasonable length)
        is_proper_name = (text[0].isupper() and 
                         len(text.split()) <= 6 and 
                         not text.isupper() and  # Not all caps
                         not any(char.isdigit() for char in text[:3]))  # No numbers at start
        
        # Check if it contains location indicators
        location_indicators = ['madrid', 'spain', 'spanish', 'de madrid', 'en madrid']
        has_location = any(indicator in text_lower for indicator in location_indicators)
        
        return has_tourism_term or is_proper_name or has_location

    def _clean_text(self, text: str) -> str:
        return re.sub(r'\s+', ' ', re.sub(r'[\r\n\t]+', ' ', text)).strip()

    def _classify_category(self, name: str, description: str) -> str:
        text = f"{name} {description}".lower()
        categories = {
            "history": ["historic", "history", "ancient", "monument", "castle", "histórico", "castillo"],
            "culture": ["museum", "gallery", "art", "cultural", "museo", "galería"],
            "nature": ["park", "garden", "nature", "parque", "jardín"],
            "religious": ["church", "cathedral", "temple", "iglesia", "catedral"],
            "architecture": ["building", "bridge", "tower", "edificio", "puente"],
            "beach": ["beach", "coast", "playa", "costa"]
        }
        for category, keywords in categories.items():
            if any(keyword in text for keyword in keywords):
                return category
        return "general"

    def _extract_appeal(self, description: str) -> str:
        if not description:
            return "Popular tourist destination"
        appeal_terms = {
            "Must-see": ["must-see", "essential", "imprescindible"],
            "Iconic": ["iconic", "famous", "icónico", "famoso"],
            "Historical": ["historical", "historic", "histórico"]
        }
        found = [label for label, terms in appeal_terms.items() if any(term in description.lower() for term in terms)]
        return ", ".join(found) or "Interesting tourist destination"

    def _classify_type(self, name: str, description: str) -> str:
        text = f"{name} {description}".lower()
        types = {
            "museum": ["museum", "gallery", "museo"],
            "historical": ["historic", "monument", "castle", "histórico"],
            "religious": ["church", "cathedral", "iglesia"],
            "park": ["park", "garden", "parque"],
            "building": ["building", "tower", "edificio"],
            "beach": ["beach", "playa"]
        }
        for typ, keywords in types.items():
            if any(keyword in text for keyword in keywords):
                return typ
        return "attraction"

    def _estimate_duration(self, name: str, description: str) -> str:
        category = self._classify_category(name, description)
        durations = {
            "museum": "2-3 hours",
            "history": "1-2 hours",
            "culture": "1-3 hours",
            "nature": "2-4 hours",
            "religious": "30-60 minutes",
            "architecture": "30-90 minutes",
            "beach": "2-6 hours",
            "general": "1-2 hours"
        }
        return durations.get(category, "1-2 hours")

    def _get_coordinates(self, name: str, city: str, address: Optional[str] = None) -> Optional[Tuple[float, float]]:
        cache_key = f"{name}:{city}"
        if cache_key in self.coordinates_cache:
            return tuple(self.coordinates_cache[cache_key])
        if self._is_too_generic(name):
            return None
        current_time = time.time()
        elapsed = current_time - self.last_geocode_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed + 0.1)
        retries = 3
        while retries > 0:
            try:
                query = f"{name}, {city}, Spain"
                if address:
                    query = f"{name}, {address}, {city}, Spain"
                location = self.geolocator.geocode(query, timeout=10)
                self.last_geocode_time = time.time()
                if location:
                    coords = (location.latitude, location.longitude)
                    self.coordinates_cache[cache_key] = coords
                    self._save_coordinates_cache()
                    return coords
                return None
            except GeocoderTimedOut:
                retries -= 1
                time.sleep(2)
                if retries == 0:
                    logger.warning(f"Geocoding failed for {name}, {city}")
                    return None
            except Exception as e:
                logger.warning(f"Geocoding error for {name}, {city}: {e}")
                return None

    def _is_too_generic(self, name: str) -> bool:
        generic_terms = ["museums", "attractions", "places", "guide", "best", "top"]
        return any(term in name.lower() for term in generic_terms)

    def _load_coordinates_cache(self) -> Dict:
        cache_file = "coordinates_cache.json"
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading coordinates cache: {e}")
        return {}

    def _save_coordinates_cache(self):
        cache_file = "coordinates_cache.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump(self.coordinates_cache, f, indent=2)
        except Exception as e:
            logger.warning(f"Error saving coordinates cache: {e}")

    def _filter_duplicates(self, places: List[TouristPlace]) -> List[TouristPlace]:
        seen = set()
        unique = []
        for place in places:
            key = (place.name.lower(), place.city.lower())
            if key not in seen:
                seen.add(key)
                unique.append(place)
        return unique

    

    def _get_structured_property(self, element: Tag, prop: str) -> Optional[str]:
        item = element.find(itemprop=prop)
        if item:
            return self._clean_text(item.get_text())
        meta = element.find('meta', {'itemprop': prop})
        if meta and meta.get('content'):
            return self._clean_text(meta['content'])
        return None