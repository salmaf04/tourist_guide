from crawler.content_extractor import ContentExtractor
from crawler.items import TouristPlaceItem
import scrapy
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
import re
from agent_generator.client import GeminiClient
import json


class TouristSpider(scrapy.Spider):
    name = "tourist_spider"
    
    # URLs organizadas por ciudades
    city_urls = {
        "Madrid": [
            "https://www.esmadrid.com/en/tourist-information",
            "https://www.esmadrid.com/en/things-to-do-in-madrid",
            "https://www.esmadrid.com/en/parks-gardens-madrid",
            "https://www.esmadrid.com/en/museums",
            "https://www.esmadrid.com/en/amusement-parks-and-zoos",
            "https://www.esmadrid.com/en/visitor-services-madrid",
            "https://www.spain.info/en/destination/madrid/",
            "https://www.timeout.com/madrid/attractions",
            "https://www.lonelyplanet.com/spain/madrid/attractions",
            "https://www.tripadvisor.com/Attractions-g187514-Activities-Madrid.html",
        ],
        "Barcelona": [
            "https://www.barcelonaturisme.com/wv3/en/",
            "https://www.barcelonaturisme.com/wv3/en/what-to-do/",
            "https://www.barcelonaturisme.com/wv3/en/what-to-see/",
            "https://www.barcelonaturisme.com/wv3/en/museums/",
            "https://www.spain.info/en/destination/barcelona/",
            "https://www.barcelona.cat/en/what-to-do-in-bcn",
            "https://www.timeout.com/barcelona/attractions",
            "https://www.lonelyplanet.com/spain/barcelona/attractions",
            "https://www.visitbarcelona.com/en/",
            "https://bcnshop.barcelonaturisme.com/shopv3/en/",
        ],
        "Valencia": [
            "https://www.visitvalencia.com/en",
            "https://www.visitvalencia.com/en/what-to-see",
            "https://www.visitvalencia.com/en/what-to-do",
            "https://www.spain.info/en/destination/valencia/",
            "https://www.valencia.es/en/visitor",
            "https://www.timeout.com/valencia/attractions",
            "https://www.lonelyplanet.com/spain/valencia/attractions",
            "https://turismovalencia.es/en/",
        ],
        "Sevilla": [
            "https://www.visitasevilla.es/en",
            "https://www.visitasevilla.es/en/what-to-see",
            "https://www.visitasevilla.es/en/what-to-do",
            "https://www.spain.info/en/destination/seville/",
            "https://www.sevilla.org/turismo/en",
            "https://www.timeout.com/seville/attractions",
            "https://www.lonelyplanet.com/spain/andalucia/seville/attractions",
            "https://www.andalucia.org/en/seville-tourism",
        ],
        "Bilbao": [
            "https://www.bilbaoturismo.net/BilbaoTurismo/en",
            "https://www.bilbaoturismo.net/BilbaoTurismo/en/what-to-see",
            "https://www.bilbaoturismo.net/BilbaoTurismo/en/what-to-do",
            "https://www.spain.info/en/destination/bilbao/",
            "https://www.bilbao.eus/en/tourism",
            "https://www.timeout.com/bilbao/attractions",
            "https://www.lonelyplanet.com/spain/basque-country/bilbao/attractions",
            "https://turismo.euskadi.eus/en/destinations/bilbao/aa30-12375/en/",
        ],
        "Granada": [
            "https://www.granadatur.com/en/",
            "https://www.granadatur.com/en/what-to-see/",
            "https://www.granadatur.com/en/what-to-do/",
            "https://www.spain.info/en/destination/granada/",
            "https://www.granada.org/inet/wturismo.nsf/home_en",
            "https://www.timeout.com/granada/attractions",
            "https://www.lonelyplanet.com/spain/andalucia/granada/attractions",
            "https://www.andalucia.org/en/granada-tourism",
        ],
        "Toledo": [
            "https://www.toledo-turismo.com/en/",
            "https://www.toledo-turismo.com/en/what-to-see/",
            "https://www.toledo-turismo.com/en/what-to-do/",
            "https://www.spain.info/en/destination/toledo/",
            "https://www.ayto-toledo.org/turismo/en/",
            "https://www.timeout.com/toledo/attractions",
            "https://www.lonelyplanet.com/spain/castilla-la-mancha/toledo/attractions",
            "https://turismo.castillalamancha.es/en/destinations/toledo",
        ],
        "Salamanca": [
            "https://www.salamanca.es/en/tourism",
            "https://www.salamanca.es/en/tourism/what-to-see",
            "https://www.salamanca.es/en/tourism/what-to-do",
            "https://www.spain.info/en/destination/salamanca/",
            "https://www.turismodesalamanca.com/en/",
            "https://www.timeout.com/salamanca/attractions",
            "https://www.lonelyplanet.com/spain/castilla-y-leon/salamanca/attractions",
            "https://turismosalamanaca.com/en/",
        ],
        "Málaga": [
            "https://www.malagaturismo.com/en/",
            "https://www.malagaturismo.com/en/what-to-see/",
            "https://www.malagaturismo.com/en/what-to-do/",
            "https://www.spain.info/en/destination/malaga/",
            "https://www.malaga.eu/en/tourism/",
            "https://www.timeout.com/malaga/attractions",
            "https://www.lonelyplanet.com/spain/andalucia/malaga/attractions",
            "https://www.andalucia.org/en/malaga-tourism",
            "https://www.costadelsol.travel/en/malaga",
        ],
        "San Sebastián": [
            "https://www.sansebastianturismo.com/en/",
            "https://www.sansebastianturismo.com/en/what-to-see/",
            "https://www.sansebastianturismo.com/en/what-to-do/",
            "https://www.spain.info/en/destination/san-sebastian/",
            "https://www.donostia.eus/en/tourism",
            "https://www.timeout.com/san-sebastian/attractions",
            "https://www.lonelyplanet.com/spain/basque-country/san-sebastian/attractions",
            "https://turismo.euskadi.eus/en/destinations/donostia-san-sebastian/aa30-12375/en/",
        ],
        "Córdoba": [
            "https://www.turismodecordoba.org/en",
            "https://www.turismodecordoba.org/en/what-to-see",
            "https://www.turismodecordoba.org/en/what-to-do",
            "https://www.spain.info/en/destination/cordoba/",
            "https://www.cordoba.es/en/tourism",
            "https://www.timeout.com/cordoba/attractions",
            "https://www.lonelyplanet.com/spain/andalucia/cordoba/attractions",
            "https://www.andalucia.org/en/cordoba-tourism",
        ],
        "Zaragoza": [
            "https://www.zaragoza.es/ciudad/turismo/en/",
            "https://www.zaragoza.es/ciudad/turismo/en/que-ver/",
            "https://www.zaragoza.es/ciudad/turismo/en/que-hacer/",
            "https://www.spain.info/en/destination/zaragoza/",
            "https://www.turismodezaragoza.es/en/",
            "https://www.timeout.com/zaragoza/attractions",
            "https://www.lonelyplanet.com/spain/aragon/zaragoza/attractions",
            "https://turismo.aragon.es/en/destinations/zaragoza",
        ],
        "Santander": [
            "https://www.turismodesantander.com/en/",
            "https://www.turismodesantander.com/en/what-to-see/",
            "https://www.turismodesantander.com/en/what-to-do/",
            "https://www.spain.info/en/destination/santander/",
            "https://www.santander.es/en/tourism",
            "https://www.timeout.com/santander/attractions",
            "https://www.lonelyplanet.com/spain/cantabria/santander/attractions",
            "https://turismodecantabria.com/en/destinations/santander",
        ],
        "Cádiz": [
            "https://turismo.cadiz.es/en/",
            "https://turismo.cadiz.es/en/what-to-see/",
            "https://turismo.cadiz.es/en/what-to-do/",
            "https://www.spain.info/en/destination/cadiz/",
            "https://www.cadizturismo.com/en/",
            "https://www.timeout.com/cadiz/attractions",
            "https://www.lonelyplanet.com/spain/andalucia/cadiz/attractions",
            "https://www.andalucia.org/en/cadiz-tourism",
        ],
        "Murcia": [
            "https://www.murciaturistica.es/en/",
            "https://www.murciaturistica.es/en/what-to-see/",
            "https://www.murciaturistica.es/en/what-to-do/",
            "https://www.spain.info/en/destination/murcia/",
            "https://www.murcia.es/en/tourism",
            "https://www.timeout.com/murcia/attractions",
            "https://www.lonelyplanet.com/spain/murcia/attractions",
            "https://www.murciaturistica.es/en/",
        ],
        "Palma de Mallorca": [
            "https://www.visitpalma.com/en/",
            "https://www.visitpalma.com/en/what-to-see/",
            "https://www.visitpalma.com/en/what-to-do/",
            "https://www.spain.info/en/destination/palma-de-mallorca/",
            "https://www.palma.cat/en/tourism",
            "https://www.timeout.com/palma/attractions",
            "https://www.lonelyplanet.com/spain/balearic-islands/palma-de-mallorca/attractions",
            "https://www.illesbalears.travel/en/balearic-islands/mallorca/palma",
        ],
    }

    max_pages = 20
    max_depth = 3
    visited_urls = set()

    def __init__(self, target_city=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.content_extractor = ContentExtractor()
        self.llm_client = GeminiClient()  # Cliente Gemini
        self.crawled_pages = 0
        self.target_city = target_city
        
        # Si se especifica una ciudad, solo usar sus URLs
        if target_city and target_city in self.city_urls:
            self.start_urls = self.city_urls[target_city]
            self.logger.info(f"Crawler configurado para ciudad específica: {target_city}")
            self.logger.info(f"URLs a procesar: {len(self.start_urls)}")
        else:
            # Si no se especifica ciudad, usar todas las URLs (comportamiento anterior)
            self.start_urls = [url for city in self.city_urls.values() for url in city]
            self.logger.info(f"Crawler configurado para todas las ciudades: {len(self.start_urls)} URLs")
        
        # Mapa para asociar URLs a ciudades
        self.url_to_city = {url: city for city, urls in self.city_urls.items() for url in urls}

    def parse(self, response):
        if self.crawled_pages >= self.max_pages:
            return

        self.crawled_pages += 1
        self.visited_urls.add(response.url)
        self.logger.info(f"Processing URL {self.crawled_pages}/{self.max_pages}: {response.url}")

        city = self._get_city_from_url(response.url)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extraer texto crudo principal de la página
        for script in soup(["script", "style", "noscript"]):
            script.decompose()
        raw_text = ' '.join(soup.stripped_strings)
        raw_text = raw_text[:6000]  # Limitar longitud para el LLM si es necesario

        prompt = (
            f"Extrae y devuelve en JSON los siguientes campos del texto de una página turística: "
            f"nombre, ciudad, categoria, descripcion, coordenadas. "
            f"Si no encuentras algún campo, déjalo vacío o null. "
            f"Texto de la página:\n\n{raw_text}\n\n"
            f"Devuelve solo el JSON."
        )
        llm_response = self.llm_client.generate(prompt)
        try:
            data = json.loads(llm_response)
            # Guardar el JSON en un archivo (uno por línea)
            with open("lugares_llm.json", "a", encoding="utf-8") as f:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
            # Crear el item para el pipeline y la base de datos
            item = TouristPlaceItem()
            item['name'] = data.get('nombre')
            item['city'] = data.get('ciudad')
            item['description'] = data.get('descripcion')
            item['category'] = data.get('categoria')
            item['coordinates'] = data.get('coordenadas')
            yield item
        except Exception as e:
            self.logger.warning(f"Error procesando respuesta LLM: {e}")

        if self.crawled_pages < self.max_pages:
            depth = response.meta.get('depth', 0)
            if depth < self.max_depth:
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    abs_url = urljoin(response.url, href)
                    clean_url = urlparse(abs_url)._replace(fragment="", query="").geturl()
                    if clean_url not in self.visited_urls:
                        if self._is_url_relevant(clean_url, city):
                            priority = self._calculate_priority(link, depth + 1)
                            yield scrapy.Request(
                                clean_url,
                                callback=self.parse,
                                meta={'depth': depth + 1, 'city': city},
                                priority=priority
                            )

    def _get_city_from_url(self, url: str) -> str:
        """Obtiene la ciudad asociada a la URL."""
        for city_name, urls in self.city_urls.items():
            for city_url in urls:
                if url.startswith(city_url) or city_name.lower() in url.lower():
                    return city_name
        return "Unknown"

    def _is_url_relevant(self, url: str, city: str) -> bool:
        """Verifica si la URL es relevante para la ciudad."""
        city_lower = city.lower()
        url_lower = url.lower()
        
        # Si tenemos una ciudad objetivo específica, ser más estricto
        if self.target_city:
            target_city_lower = self.target_city.lower()
            # Solo seguir enlaces que contengan el nombre de la ciudad objetivo
            # o que estén en dominios conocidos de esa ciudad
            if target_city_lower not in url_lower:
                # Verificar si la URL pertenece a un dominio conocido de la ciudad objetivo
                for known_url in self.city_urls.get(self.target_city, []):
                    domain = urlparse(known_url).netloc
                    if domain in url_lower:
                        break
                else:
                    return False
        
        # Prioriza URLs que contengan el nombre de la ciudad o parezcan relacionadas
        return city_lower in url_lower or any(term in url_lower for term in ['tourism', 'attraction', 'place', 'visit'])

    def _calculate_priority(self, link, depth):
        priority = 5
        priority -= depth * 2
        link_text = link.get_text().lower()
        if any(term in link_text for term in ['attraction', 'place', 'museum', 'tour', 'visit']):
            priority += 3
        parent_classes = ' '.join(link.parent.get('class', []))
        if any(cls in parent_classes for cls in ['list', 'item', 'card', 'attraction']):
            priority += 2
        return max(1, min(10, priority))