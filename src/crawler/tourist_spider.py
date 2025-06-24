"""
Spider principal reorganizado para el crawler turístico.
"""

import scrapy
import json
import numpy as np
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import chromadb

from crawler.items import TouristPlaceItem
from crawler.model import TouristPlace
from .config import CRAWLER_CONFIG, CITY_URLS, CITY_URLS_EXTRA
from .filters import URLFilter
from .city_utils import CityIdentifier
from .content_processor import ContentProcessor
from .stats_manager import StatsManager
from agent_generator.mistral_client import MistralClient
from utils.chroma_db_manager import ChromaDBManager


class TouristSpider(scrapy.Spider):
    name = "tourist_spider"
    
    def __init__(self, target_city=None, max_pages=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Inicializar componentes
        self.llm_client = MistralClient()
        self.url_filter = URLFilter(self.logger)
        self.city_identifier = CityIdentifier(self.logger)
        self.content_processor = ContentProcessor(self.llm_client, self.logger)
        self.stats_manager = StatsManager(
            max_pages_per_city=CRAWLER_CONFIG['max_pages_per_city'],
            max_places_per_city=CRAWLER_CONFIG['max_places_per_city'],
            logger=self.logger
        )
        
        # Configuración
        self.max_pages = CRAWLER_CONFIG['max_pages']
        self.max_depth = CRAWLER_CONFIG['max_depth']
        self.visited_urls = set()

        # --- NUEVO: Gestión de ciudades pendientes y completadas ---
        self.cities_pending = []
        self.cities_completed = set()
        self.current_city = None
        # ----------------------------------------------------------

        self.target_city = target_city
        
        # Configurar max_pages si se proporciona
        if max_pages:
            try:
                self.max_pages = int(max_pages)
            except (ValueError, TypeError):
                self.max_pages = CRAWLER_CONFIG['max_pages']
        
        # Configurar URLs iniciales
        self._setup_start_urls()
        # Inicializar ChromaDB
        self._setup_chroma_db()
        
        # NUEVO: Cache de lugares existentes para prevenir duplicados
        self.existing_places_cache = set()
        self._load_existing_places_cache()
    
    def _setup_start_urls(self):
        """Configura las URLs iniciales del crawler y la gestión de ciudades."""
        # Combinar ambos diccionarios para la búsqueda
        self.city_urls = CITY_URLS
        self.city_urls_extra = CITY_URLS_EXTRA
        all_city_urls = {**self.city_urls, **self.city_urls_extra}

        # --- NUEVO: Inicializar ciudades pendientes ---
        self.cities_pending = list(all_city_urls.keys())
        self.cities_completed = set()
        self.current_city = self.cities_pending[0] if self.cities_pending else None
        # ------------------------------------------------

        # Si se especifica una ciudad y existe en el diccionario
        if self.target_city and self.target_city in all_city_urls:
            self.start_urls = all_city_urls[self.target_city]
            self.current_city = self.target_city
            self.cities_pending = [self.target_city]
        else:
            # Usar la primera ciudad pendiente
            self.start_urls = all_city_urls[self.current_city] if self.current_city else []

        self.logger.info(f"Configurando crawler para ciudades: {self.cities_pending}")
        self.logger.info(f"Total URLs iniciales: {len(self.start_urls)}")

        # Verificación crítica - asegurar que hay URLs
        if not self.start_urls:
            self.logger.error("¡No hay URLs iniciales válidas después del filtrado!")
            # Opcional: agregar URLs de fallback
            self.start_urls = [
                "https://www.barcelonaturisme.com/wv3/en/",
                "https://www.visitvalencia.com/en"
            ]

    def _setup_chroma_db(self):
        """Inicializa ChromaDB y el modelo de embeddings."""
        try:
            # Inicializar cliente ChromaDB
            import chromadb
            chroma_client = chromadb.PersistentClient(path="src/crawler/db")
            
            # Obtener o crear colección
            try:
                collection = chroma_client.get_collection(name="tourist_places")
                self.logger.info("Usando colección existente 'tourist_places'")
            except Exception:
                collection = chroma_client.create_collection(name="tourist_places")
                self.logger.info("Creada nueva colección 'tourist_places'")
            
            # Inicializar ChromaDB manager con la colección
            self.chroma_manager = ChromaDBManager(collection)
            
            # Inicializar modelo de embeddings
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            self.logger.info("ChromaDB y modelo de embeddings inicializados correctamente")
            
        except Exception as e:
            self.logger.error(f"Error inicializando ChromaDB: {e}")
            # Crear instancias dummy para evitar errores
            self.chroma_manager = None
            self.embedding_model = None
    
    def _load_existing_places_cache(self):
        """Carga un cache de lugares existentes para prevenir duplicados."""
        try:
            if not self.chroma_manager:
                self.logger.warning("ChromaDB no disponible - cache de duplicados deshabilitado")
                return
            
            # Obtener todos los lugares existentes
            all_docs = self.chroma_manager.collection.get(include=["metadatas"])
            
            # Crear cache con clave normalizada (nombre + ciudad)
            for metadata in all_docs["metadatas"]:
                name = metadata.get("name", "").strip().lower()
                city = metadata.get("city", "").strip().lower()
                
                if name and city:
                    # Normalizar texto para comparación
                    normalized_key = self._normalize_place_key(name, city)
                    self.existing_places_cache.add(normalized_key)
            
            self.logger.info(f"Cache de duplicados cargado: {len(self.existing_places_cache)} lugares existentes")
            
        except Exception as e:
            self.logger.error(f"Error cargando cache de duplicados: {e}")
            self.existing_places_cache = set()
    
    def _normalize_place_key(self, name, city):
        """Normaliza nombre y ciudad para crear clave única."""
        import re
        
        def normalize_text(text):
            if not text:
                return ""
            
            # Convertir a minúsculas
            text = text.lower().strip()
            
            # Remover acentos
            replacements = {
                'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u', 'ü': 'u',
                'ñ': 'n', 'ç': 'c'
            }
            for old, new in replacements.items():
                text = text.replace(old, new)
            
            # Remover caracteres especiales y espacios extra
            text = re.sub(r'[^\w\s]', ' ', text)
            text = re.sub(r'\s+', ' ', text)
            
            return text.strip()
        
        normalized_name = normalize_text(name)
        normalized_city = normalize_text(city)
        
        return f"{normalized_name}|{normalized_city}"
    
    def _is_duplicate_place(self, name, city):
        """Verifica si un lugar ya existe en la base de datos."""
        if not name or not city:
            return False
        
        normalized_key = self._normalize_place_key(name, city)
        return normalized_key in self.existing_places_cache

    def _check_and_advance_city(self, city):
        """Marca una ciudad como completada y avanza a la siguiente pendiente si es necesario."""
        if city not in self.cities_completed:
            self.cities_completed.add(city)
            if city in self.cities_pending:
                self.cities_pending.remove(city)
        if self.current_city == city:
            # Avanzar a la siguiente ciudad pendiente
            if self.cities_pending:
                self.current_city = self.cities_pending[0]
                all_city_urls = {**self.city_urls, **self.city_urls_extra}
                self.start_urls = all_city_urls[self.current_city]
                self.logger.info(f"➡️ Cambiando a la siguiente ciudad: {self.current_city}")
            else:
                self.current_city = None
                self.start_urls = []
                self.logger.info("✅ Todas las ciudades procesadas")

    def parse(self, response):
        """Método principal de parsing."""
        self.logger.info(f"Procesando URL: {response.url}")
        
                
        # Control del número máximo de páginas
        if self.stats_manager.crawled_pages >= self.max_pages:
            return
        
        # Identificar ciudad de la URL actual
        current_city = self.city_identifier.get_city_from_url(response.url)

        # Si hay una ciudad objetivo específica, solo procesar esa ciudad
        if self.target_city:
            if current_city.lower() != self.target_city.lower():
                self.logger.debug(f"🚫 Saltando URL de {current_city} - solo procesando {self.target_city}: {response.url}")
                return
            # Forzar current_city a ser la ciudad objetivo para consistencia
            current_city = self.target_city

        # --- NUEVO: Saltar ciudades completadas ---
        if current_city in self.cities_completed:
            self.logger.info(f"🚫 Ciudad {current_city} ya completada. Saltando URL: {response.url}")
            return
        # ------------------------------------------------

        # CONTROL DE UMBRAL POR CIUDAD - Solo aplicar límites si NO hay una ciudad objetivo específica
        if not self.target_city and not self.stats_manager.can_process_city_page(current_city):
            self.logger.info(f"🚫 Límite de páginas alcanzado para {current_city}")
            self._check_and_advance_city(current_city)
            return
        
        # Solo aplicar límites si NO hay una ciudad objetivo específica
        if not self.target_city and not self.stats_manager.can_add_city_place(current_city):
            self.logger.info(f"🚫 Límite de lugares alcanzado para {current_city}")
            self._check_and_advance_city(current_city)
            return
        
        # Actualizar contadores
        self.stats_manager.increment_page_count(current_city)
        self.stats_manager.increment_crawled_pages()
        self.visited_urls.add(response.url)
        
        self.logger.info(f"Processing URL {self.stats_manager.crawled_pages}/{self.max_pages}: {response.url} (Ciudad: {current_city})")
        
        # Procesar contenido de la página
        yield from self._process_page_content(response, current_city)
        
        # Mostrar estadísticas periódicamente
        if self.stats_manager.should_log_stats():
            self.stats_manager.log_diversity_stats()
        
        # Continuar con el crawling
        yield from self._follow_links(response, current_city)
    
    def _process_page_content(self, response, current_city):
        """Procesa el contenido de una página y extrae lugares turísticos."""
        try:
            # Procesar contenido con LLM
            lugares = self.content_processor.process_page_content(response, current_city)
            
            if not lugares:
                self.logger.warning(f"No se encontraron lugares en {response.url}")
                return
            
            # Guardar lugares en archivo
            self.content_processor.save_places_to_file(lugares)
            
            # Procesar cada lugar encontrado
            places_added = 0
            for lugar in lugares:
                try:
                    # Mejorar identificación de ciudad
                    lugar_ciudad = self._improve_city_identification(lugar, response.url, current_city)
                    
                    # Crear item para el pipeline
                    item = self._create_tourist_item(lugar, lugar_ciudad)
                    
                    # Guardar en ChromaDB
                    if self._save_to_chroma_db(item):
                        places_added += 1
                        self.stats_manager.increment_place_count(lugar_ciudad)
                        self.logger.info(f"Lugar agregado: {item['name']} ({lugar_ciudad}) - Total en {lugar_ciudad}: {self.stats_manager.get_city_place_count(lugar_ciudad)}")
                        yield item
                    
                except Exception as e:
                    self.logger.error(f"Error procesando lugar individual {lugar.get('nombre', 'N/A')}: {e}")
            
            if places_added == 0:
                self.logger.warning(f"No se agregaron lugares desde {response.url}")
            else:
                self.logger.info(f"Total de lugares agregados desde {response.url}: {places_added}")
                
        except Exception as e:
            self.logger.error(f"Error general procesando {response.url}: {e}")
    
    def _improve_city_identification(self, lugar, url, current_city):
        """Mejora la identificación de ciudad usando múltiples estrategias."""
        # Si hay una ciudad objetivo específica, siempre usar esa ciudad
        if self.target_city:
            self.logger.debug(f"Usando ciudad objetivo {self.target_city} para {lugar.get('nombre')}")
            return self.target_city
        
        lugar_ciudad = lugar.get('ciudad', current_city)
        
        # Aplicar múltiples estrategias para evitar "Unknown"
        if lugar_ciudad == "Unknown" or not lugar_ciudad or lugar_ciudad.strip() == "":
            # Estrategia 1: Inferir del contexto
            lugar_ciudad = self.city_identifier.infer_city_from_context(
                lugar.get('nombre'), url, current_city
            )
            
            # Estrategia 2: Si sigue siendo Unknown, usar la ciudad de la URL
            if lugar_ciudad == "Unknown" and current_city != "Unknown":
                lugar_ciudad = current_city
                self.logger.debug(f"Usando ciudad de URL: {current_city} para {lugar.get('nombre')}")
            
            # Estrategia 3: Buscar en el nombre del lugar
            if lugar_ciudad == "Unknown":
                lugar_ciudad = self.city_identifier.extract_city_from_place_name(lugar.get('nombre', ''))
            
            # Estrategia 4: Como último recurso, usar la ciudad más común
            if lugar_ciudad == "Unknown":
                lugar_ciudad = self.stats_manager.get_most_common_city() or "Barcelona"
                self.logger.debug(f"Usando ciudad más común como fallback: {lugar_ciudad} para {lugar.get('nombre')}")
        
        return lugar_ciudad
    
    def _create_tourist_item(self, lugar, ciudad):
        """Crea un item TouristPlaceItem desde los datos del lugar."""
        item = TouristPlaceItem()
        item['name'] = lugar.get('nombre')
        item['city'] = ciudad
        item['description'] = lugar.get('descripcion')
        item['category'] = lugar.get('categoria')
        item['coordinates'] = lugar.get('coordenadas')
        return item
    
    def _save_to_chroma_db(self, item):
        """Guarda el item en ChromaDB con verificación de duplicados."""
        try:
            # Verificar si ChromaDB está disponible
            if not self.chroma_manager or not self.embedding_model:
                self.logger.warning("ChromaDB no disponible - saltando guardado")
                return True  # Retornar True para continuar el procesamiento
            
            name = item['name'] or "Sin nombre"
            city_val = item['city'] or "Unknown"
            category = item['category'] or "Sin categoría"
            description = item['description'] or "Sin descripción"
            coordinates = item['coordinates'] if item['coordinates'] else None
            
            # Validar datos
            if not name or not city_val or not category or not description:
                self.logger.warning(f"Lugar con datos incompletos: {name} - saltando")
                return False
            
            # NUEVO: Verificar duplicados antes de guardar
            if self._is_duplicate_place(name, city_val):
                self.logger.info(f"🚫 DUPLICADO DETECTADO: {name} ({city_val}) - saltando")
                return False
            
            # Crear objeto TouristPlace
            place = TouristPlace(
                name=name,
                city=city_val,
                category=category,
                description=description,
                coordinates=coordinates
            )
            
            # Generar embedding
            text_to_embed = f"Name: {name}\nCity: {city_val}\nCategory: {category}\nDescription: {description}"
            embedding = self.embedding_model.encode(text_to_embed)
            
            # Guardar en ChromaDB
            self.chroma_manager.add_place(place, np.array(embedding))
            
            # NUEVO: Añadir al cache para futuras verificaciones
            normalized_key = self._normalize_place_key(name, city_val)
            self.existing_places_cache.add(normalized_key)
            
            self.logger.debug(f"✅ Lugar guardado y añadido al cache: {name} ({city_val})")
            return True
            
        except Exception as e:
            self.logger.error(f"Error guardando en ChromaDB: {e}")
            return False
    
    def _follow_links(self, response, current_city):
        """Sigue enlaces relevantes para continuar el crawling, solo de ciudades pendientes."""
        if self.stats_manager.crawled_pages >= self.max_pages:
            return
        depth = response.meta.get('depth', 0)
        if depth >= self.max_depth:
            return
        try:
            soup = BeautifulSoup(response.text, 'html.parser')
            for link in soup.find_all('a', href=True):
                href = link['href']
                abs_url = urljoin(response.url, href)
                clean_url = abs_url.split('#')[0].split('?')[0]
                if clean_url in self.visited_urls:
                    continue
                # Verificar ciudad del enlace
                link_city = self.city_identifier.get_city_from_url(clean_url)
                
                # Si hay una ciudad objetivo específica, solo seguir enlaces de esa ciudad
                if self.target_city:
                    if link_city.lower() != self.target_city.lower():
                        self.logger.debug(f"🚫 Saltando enlace de {link_city} - solo siguiendo enlaces de {self.target_city}")
                        continue
                    # Forzar link_city a ser la ciudad objetivo
                    link_city = self.target_city
                
                # --- NUEVO: Saltar ciudades completadas ---
                if link_city in self.cities_completed:
                    continue
                # CONTROL DE UMBRAL: No seguir enlaces de ciudades que alcanzaron el límite (solo si no hay ciudad objetivo)
                if not self.target_city and not self.stats_manager.can_process_city_page(link_city):
                    self.logger.debug(f"Saltando enlace de {link_city} - límite de páginas alcanzado")
                    self._check_and_advance_city(link_city)
                    continue
                # Solo aplicar límites si NO hay una ciudad objetivo específica
                if not self.target_city and not self.stats_manager.can_add_city_place(link_city):
                    self.logger.debug(f"Saltando enlace de {link_city} - límite de lugares alcanzado")
                    self._check_and_advance_city(link_city)
                    continue
                if self.url_filter.is_url_relevant(clean_url, link_city):
                    # Priorizar ciudades con menos contenido
                    priority = self.stats_manager.calculate_priority_with_diversity(link, depth + 1, link_city)
                    yield scrapy.Request(
                        clean_url,
                        callback=self.parse,
                        meta={'depth': depth + 1, 'city': link_city},
                        priority=priority
                    )
        except Exception as e:
            self.logger.error(f"Error siguiendo enlaces en {response.url}: {e}")