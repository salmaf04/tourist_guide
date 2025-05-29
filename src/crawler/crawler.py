"""
Web crawler para turismo en España con arquitectura mejorada siguiendo principios SOLID.
"""

import csv
import requests
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import time
import re
import warnings
import uuid
import traceback
import urllib3
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass
import chromadb
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Configuración global
warnings.filterwarnings("ignore", category=UserWarning, module='nltk.downloader')
warnings.filterwarnings("ignore", message=".*verify=False.*", category=urllib3.exceptions.InsecureRequestWarning)

@dataclass
class RobotsTxtInfo:
    allowed: List[str]
    disallowed: List[str]
    crawl_delay: float
    request_rate: Optional[float]

@dataclass
class NLTKProcessedData:
    lemmatized_tokens: List[str]
    named_entities: List[Dict[str, str]]

class NLTKManager:
    """Manejador centralizado de recursos y procesamiento NLTK"""
    REQUIRED_RESOURCES = {
        'tokenizers/punkt': 'punkt',
        'corpora/stopwords': 'stopwords',
        'corpora/wordnet': 'wordnet',
        'corpora/omw-1.4': 'omw-1.4',
        'taggers/averaged_perceptron_tagger': 'averaged_perceptron_tagger',
        'chunkers/maxent_ne_chunker': 'maxent_ne_chunker',
        'corpora/words': 'words'
    }
    
    def __init__(self, language: str = 'english'):
        self.language = language
        self._download_resources()
        
    def _download_resources(self):
        """Descarga todos los recursos necesarios de NLTK"""
        for resource, package in self.REQUIRED_RESOURCES.items():
            try:
                nltk.download(package)
            except Exception as e:
                print(f"Error downloading NLTK resource {package}: {e}")
    
    def process_text(self, text: str) -> NLTKProcessedData:
        """
        Procesa el texto con NLTK: tokenización, lematización y extracción de entidades nombradas.
        """
        processed_data = NLTKProcessedData(lemmatized_tokens=[], named_entities=[])
        
        try:
            # Tokenización y limpieza
            words = word_tokenize(text.lower(), language=self.language)
            stop_words_set = set(stopwords.words(self.language))
            filtered_words = [w for w in words if w.isalnum() and w not in stop_words_set]
            
            # Lematización
            lemmatizer = WordNetLemmatizer()
            processed_data.lemmatized_tokens = [lemmatizer.lemmatize(w) for w in filtered_words]
            
            # Extracción de entidades
            pos_tags = nltk.pos_tag(processed_data.lemmatized_tokens)
            tree = nltk.ne_chunk(pos_tags)
            
            for i in tree:
                if isinstance(i, nltk.tree.Tree):
                    entity_label = i.label()
                    entity_string = " ".join([token for token, pos in i.leaves()])
                    processed_data.named_entities.append({
                        "label": entity_label, 
                        "text": entity_string
                    })
                    
            # Eliminar duplicados
            unique_entities = {
                (entity['label'], entity['text']): entity 
                for entity in processed_data.named_entities
                if entity['label'] in ['GPE', 'ORG', 'PERSON']
            }
            
            processed_data.named_entities = list(unique_entities.values())
            
        except Exception as e:
            print(f"Error during NLTK processing: {e}")
            traceback.print_exc()
            
        return processed_data

class URLFilter:
    """Filtra URLs basado en robots.txt y otras reglas"""
    
    def __init__(self, robots_info: RobotsTxtInfo, allowed_domain: str):
        self.robots_info = robots_info
        self.allowed_domain = allowed_domain
        
    def is_allowed(self, url: str) -> bool:
        """Determina si una URL está permitida para crawling"""
        parsed_url = urlparse(url)
        
        # Verificar dominio
        if parsed_url.netloc != self.allowed_domain:
            return False
            
        # Verificar robots.txt
        url_path = parsed_url.path or "/"
        return self._is_allowed_by_robots(url_path)
        
    def _is_allowed_by_robots(self, url_path: str) -> bool:
        """Verifica si una ruta está permitida según robots.txt"""
        if self.robots_info.allowed:
            for allowed_path in self.robots_info.allowed:
                if url_path.startswith(allowed_path):
                    return True
                    
        for disallowed_path in self.robots_info.disallowed:
            if disallowed_path == '/':
                return False
            if disallowed_path and url_path.startswith(disallowed_path):
                return False
                
        return True

class ContentExtractor:
    """Extrae contenido relevante de páginas HTML"""
    
    MAIN_CONTENT_SELECTORS = [
        'article', 'main', '.content', '#content', 
        '.main-content', '#main-content', 'div[role="main"]'
    ]
    
    def extract_text(self, soup: BeautifulSoup) -> str:
        """Extrae el texto principal de una página HTML"""
        self._remove_unwanted_elements(soup)
        texts = self._extract_main_content(soup)
        return self._clean_text("\n".join(filter(None, texts)))
        
    def _remove_unwanted_elements(self, soup: BeautifulSoup):
        """Elimina elementos no deseados del árbol DOM"""
        for element in soup(["script", "style", "header", "footer", "nav", 
                           "aside", "form", "button", "input"]):
            element.decompose()
            
    def _extract_main_content(self, soup: BeautifulSoup) -> List[str]:
        """Intenta extraer el contenido principal usando selectores comunes"""
        texts = []
        
        for selector in self.MAIN_CONTENT_SELECTORS:
            main_elements = soup.select(selector)
            if main_elements:
                texts.extend(el.get_text(separator=' ', strip=True) 
                           for el in main_elements)
                return texts
                
        # Fallback a párrafos o body
        paragraphs = soup.find_all('p')
        if paragraphs:
            texts.extend(p.get_text(strip=True) for p in paragraphs)
        elif soup.body:
            texts.append(soup.body.get_text(separator=' ', strip=True))
            
        return texts
        
    def _clean_text(self, text: str) -> str:
        """Limpia y normaliza el texto extraído"""
        text = re.sub(r'\n\s*\n', '\n\n', text)
        return re.sub(r'\s{2,}', ' ', text).strip()

class ChromaDBManager:
    """Manejador de operaciones con ChromaDB"""
    
    def __init__(self, collection: chromadb.Collection):
        self.collection = collection
        
    def url_exists(self, url: str) -> bool:
        """Verifica si una URL ya existe en la colección"""
        docs = self.collection.get(where={"source_url": url})
        return len(docs["ids"]) > 0
        
    def add_document(self, text: str, url: str, metadata: Dict) -> str:
        """Añade un documento a la colección"""
        doc_id = str(uuid.uuid4())
        self.collection.add(
            documents=[text],
            metadatas=[metadata],
            ids=[doc_id]
        )
        return doc_id
        
    def get_all_documents(self):
        """Obtiene todos los documentos de la colección"""
        return self.collection.get(include=["metadatas", "documents"])
    
    def test_collection(self) :
        if self.collection.count() > 0:
            print(f"\n--- Example ChromaDB Query ---")
            query_results = self.collection.query(
                query_texts=["cultural events in Madrid"],
                n_results=2,
                include=["metadatas", "documents", "distances"]
            )
            if query_results and query_results.get('documents'):
                print("Query Results:")
                for i, doc_text in enumerate(query_results['documents'][0]):
                    metadata = query_results['metadatas'][0][i]
                    distance = query_results['distances'][0][i]
                    print(f"\nResult {i+1}:")
                    print(f"  Source: {metadata.get('source_url', 'N/A')}")
                    print(f"  Distance: {distance:.4f}")
                    print(f"  Named Entities: {metadata.get('named_entities', 'N/A')}")
                    print(f"  Text (snippet): {doc_text[:250]}...")
            else:
                print("No results found or issue with results format.")

            print("\n--- Todos los documentos en la colección ChromaDB ---")
            all_docs = self.collection.get(include=["metadatas", "documents"])
            for i, doc in enumerate(all_docs["documents"]):
                print(f"\nDocumento {i+1}:")
                print(f"  ID: {all_docs['ids'][i]}")
                print(f"  Source: {all_docs['metadatas'][i].get('source_url', 'N/A')}")
                print(f"  Named Entities: {all_docs['metadatas'][i].get('named_entities', 'N/A')}")
                print(f"  Tokens: {all_docs['metadatas'][i].get('lemmatized_token_count', 'N/A')}")
                print(f"  Texto (snippet): {doc[:250]}...")
        else:
            print("\nChromaDB collection is empty. No query performed.")

class URLQueue:
    """Cola de URLs para visitar con control de visitados"""
    
    def __init__(self):
        self._queue = []
        self._visited_urls = set()
        
    def add(self, url: str) -> bool:
        """Añade una URL a la cola si no ha sido visitada"""
        if url not in self._visited_urls and url not in self._queue:
            self._queue.append(url)
            return True
        return False
        
    def pop(self) -> Optional[str]:
        """Extrae la siguiente URL de la cola"""
        if not self._queue:
            return None
        url = self._queue.pop(0)
        self._visited_urls.add(url)
        return url
        
    def __len__(self) -> int:
        """Número de URLs pendientes"""
        return len(self._queue)

class RobotsTxtParser:
    """Parser para archivos robots.txt"""
    
    @staticmethod
    def parse(content: str) -> RobotsTxtInfo:
        """
        Parsea el contenido de robots.txt y extrae reglas relevantes para nuestro crawler.
        
        Args:
            content: Texto completo del archivo robots.txt
            
        Returns:
            RobotsTxtInfo: Objeto con las reglas parseadas
        """
        if not content:
            return RobotsTxtInfo(allowed=[], disallowed=[], crawl_delay=1.0, request_rate=None)
        
        lines = content.split('\n')
        user_agents = ['*']  # Por defecto consideramos todas las user-agents
        allowed = []
        disallowed = []
        crawl_delay = 1.0  # Valor por defecto
        request_rate = None
        
        current_user_agent = None
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            # Buscar user-agent
            if line.lower().startswith('user-agent:'):
                current_user_agent = line.split(':', 1)[1].strip()
                if current_user_agent not in user_agents:
                    current_user_agent = None
                continue
                
            # Solo procesamos si es para todas las user-agents o nuestra específica
            if current_user_agent is None and not any(ua in user_agents for ua in user_agents):
                continue
                
            # Procesar allow/disallow
            if line.lower().startswith('allow:'):
                path = line.split(':', 1)[1].strip()
                if path:
                    allowed.append(path)
            elif line.lower().startswith('disallow:'):
                path = line.split(':', 1)[1].strip()
                if path:
                    disallowed.append(path)
            elif line.lower().startswith('crawl-delay:'):
                try:
                    crawl_delay = float(line.split(':', 1)[1].strip())
                except (ValueError, IndexError):
                    crawl_delay = 1.0  # Valor por defecto si hay error
            elif line.lower().startswith('request-rate:'):
                try:
                    request_rate = line.split(':', 1)[1].strip()
                except IndexError:
                    request_rate = None
        
        # Eliminar duplicados y ordenar por longitud (las rutas más específicas primero)
        allowed = sorted(list(set(allowed)), key=len, reverse=True)
        disallowed = sorted(list(set(disallowed)), key=len, reverse=True)
        
        return RobotsTxtInfo(
            allowed=allowed,
            disallowed=disallowed,
            crawl_delay=crawl_delay,
            request_rate=request_rate
        )

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Divide el texto en chunks de tamaño chunk_size (en palabras), con solapamiento opcional.
    """
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        chunks.append(' '.join(chunk))
        if i + chunk_size >= len(words):
            break
        i += chunk_size - overlap
    return chunks

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
        self.content_extractor = ContentExtractor()
        
    def crawl(self):
        """Ejecuta el proceso de crawling"""
        for start_url in self.start_urls:
            self._process_start_url(start_url)
            
        print(f"\nCrawling finished. Collected {self.documents_collected} documents.")
        
    def _process_start_url(self, start_url: str):
        """Configura el crawling para una URL inicial"""
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
    
    def _get_robots_info(self, url: str) -> RobotsTxtInfo:
        """Obtiene la información de robots.txt para un dominio"""
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
    
    def _process_url(
        self, 
        url: str, 
        url_filter: URLFilter,
        crawl_delay: float
    ):
        """Procesa una URL individual"""
        if not url_filter.is_allowed(url) or self.chroma_manager.url_exists(url):
            return
            
        try:
            time.sleep(crawl_delay)
            html_content = self._fetch_url_content(url)
            if not html_content:
                return
                
            soup = BeautifulSoup(html_content, 'html.parser')
            document_text = self.content_extractor.extract_text(soup)
            
            if len(document_text) > 100:
                self._store_document(document_text, url)
                self._extract_and_queue_links(soup, url, url_filter)
                
        except Exception as e:
            print(f"Error processing {url}: {e}")
            traceback.print_exc()
    
    def _fetch_url_content(self, url: str) -> Optional[str]:
        """Obtiene el contenido de una URL"""
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
    
    def _store_document(self, text: str, url: str):
        """Almacena los chunks del documento procesado en ChromaDB"""
        chunks = chunk_text(text, chunk_size=500, overlap=50)
        for idx, chunk in enumerate(chunks):
            processed_data = self.nltk_manager.process_text(chunk)
            entities_str = ", ".join(
                f"{entity['label']}:{entity['text']}" 
                for entity in processed_data.named_entities
            )
            metadata = {
                "source_url": url,
                "named_entities": entities_str,
                "lemmatized_token_count": len(processed_data.lemmatized_tokens),
                "chunk_index": idx
            }
            doc_id = self.chroma_manager.add_document(chunk, url, metadata)
            self.documents_collected += 1
            print(f"Stored chunk {idx+1}/{len(chunks)} (doc {self.documents_collected}/{self.max_documents}) from {url}")
            print(f"  Extracted Entities: {entities_str if entities_str else 'None'}")
            if self.documents_collected >= self.max_documents:
                break
    
    def _extract_and_queue_links(
        self, 
        soup: BeautifulSoup, 
        base_url: str,
        url_filter: URLFilter
    ):
        """Extrae y encola enlaces válidos de la página"""
        for link in soup.select('a[href]'):
            href = link.get('href')
            if href:
                abs_url = urljoin(base_url, href)
                if url_filter.is_allowed(abs_url):
                    self.url_queue.add(abs_url)

def main():
    # Configuración inicial
    start_urls = [
        "https://www.spain.info/",
        "https://www.turismo.europa.eu/"
    ]
    
    # Inicialización de dependencias
    chroma_client = chromadb.Client()
    collection = chroma_client.get_or_create_collection(name="tourism_docs")
    chroma_manager = ChromaDBManager(collection)
    nltk_manager = NLTKManager(language='english')
    
    # Crear y ejecutar crawler
    crawler = TourismCrawler(
        start_urls=start_urls,
        chroma_manager=chroma_manager,
        nltk_manager=nltk_manager,
        max_documents=10
    )
    
    crawler.crawl()

    chroma_manager.test_collection()
    

if __name__ == "__main__":
    main()




