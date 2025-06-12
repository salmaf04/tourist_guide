import requests
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import time
import warnings
import urllib3
from typing import List, Optional
import chromadb
from sentence_transformers import SentenceTransformer
from .improved_content_extractor import ImprovedContentExtractor, TouristPlace
from .url_filter import URLFilter
from .url_queue import URLQueue
from .robots_txt_parser import RobotsTxtParser, RobotsTxtInfo
from .nltk_manager import NLTKManager
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from chroma_db_manager import ChromaDBManager

# Configuración global
warnings.filterwarnings("ignore", message=".*verify=False.*", category=urllib3.exceptions.InsecureRequestWarning)

class TourismCrawler:
    """Crawler principal para sitios de turismo en España"""

    def __init__(
        self,
        start_urls: List[str],
        chroma_manager: ChromaDBManager,
        nltk_manager: NLTKManager,
        max_documents: int = 10,
        user_agent: str = "TourismCrawler/1.0"
    ):
        self.start_urls = start_urls
        self.chroma_manager = chroma_manager
        self.nltk_manager = nltk_manager
        self.max_documents = max_documents
        self.user_agent = user_agent
        self.url_queue = URLQueue()
        self.content_extractor = ImprovedContentExtractor()
        self.model = SentenceTransformer("all-MiniLM-L6-v2")  # Modelo para embeddings
        self.documents_collected = 0

    def crawl(self):
        """Ejecuta el proceso de crawling."""
        for start_url in self.start_urls:
            self._process_start_url(start_url)

        print(f"\nCrawling finished. Collected {self.documents_collected} documents.")

    def _process_start_url(self, start_url: str):
        """Configura el crawling para una URL inicial."""
        robots_info = self._get_robots_info(start_url)
        domain = urlparse(start_url).netloc
        url_filter = URLFilter(robots_info, domain)

        self.url_queue.add(start_url)
        self.documents_collected = 0

        while self.documents_collected < self.max_documents and len(self.url_queue) > 0:
            current_url = self.url_queue.pop()
            if not current_url:
                continue

            self._process_url(current_url, url_filter, robots_info.crawl_delay)

    def _get_robots_info(self, url: str):
        """Obtiene la información de robots.txt para un dominio."""
        robots_url = urljoin(url, '/robots.txt')
        try:
            response = requests.get(
                robots_url,
                timeout=10,
                verify=False,
                headers={'User-Agent': self.user_agent}
            )
            response.raise_for_status()
            return RobotsTxtParser.parse(response.text)
        except requests.exceptions.RequestException as e:
            print(f"Error fetching robots.txt: {e}")
            return RobotsTxtInfo(allowed=[], disallowed=[], crawl_delay=1.0, request_rate=None)

    def _process_url(self, url: str, url_filter: URLFilter, crawl_delay: float):
        """Procesa una URL individual."""
        if not url_filter.is_allowed(url) or self.chroma_manager.url_exists(url):
            return

        try:
            time.sleep(crawl_delay)
            html_content = self._fetch_url_content(url)
            if not html_content:
                return

            soup = BeautifulSoup(html_content, 'html.parser')
            places = self.content_extractor.extract_places(soup, url, self.nltk_manager)

            for place in places:
                if self.documents_collected >= self.max_documents:
                    break

                # Generar embedding para el lugar
                text_representation = f"""
Name: {place.name}
City: {place.city}
Category: {place.category}
Description: {place.description}
Visitor Appeal: {place.visitor_appeal}
Classification: {place.tourist_classification}
Estimated Visit Duration: {place.estimated_visit_duration}
Coordinates: {place.coordinates if place.coordinates else 'Unknown'}
                """.strip()

                embedding = self.model.encode(text_representation)
                doc_id = self.chroma_manager.add_place(place, embedding)

                self.documents_collected += 1
                print(f"Stored place {self.documents_collected}/{self.max_documents}: {place.name} from {url}")

            self._extract_and_queue_links(soup, url, url_filter)

        except Exception as e:
            print(f"Error processing {url}: {e}")

    def _fetch_url_content(self, url: str) -> Optional[str]:
        """Obtiene el contenido de una URL."""
        try:
            response = requests.get(
                url,
                timeout=15,
                verify=False,
                headers={'User-Agent': self.user_agent}
            )
            response.raise_for_status()

            if 'text/html' not in response.headers.get('Content-Type', '').lower():
                print(f"Skipping non-HTML content: {url}")
                return None

            return response.content
        except requests.exceptions.RequestException as e:
            print(f"Request error for {url}: {e}")
            return None

    def _extract_and_queue_links(self, soup: BeautifulSoup, base_url: str, url_filter: URLFilter):
        """Extrae y encola enlaces válidos de la página."""
        for link in soup.select('a[href]'):
            href = link.get('href')
            if href:
                abs_url = urljoin(base_url, href)
                if url_filter.is_allowed(abs_url):
                    self.url_queue.add(abs_url)