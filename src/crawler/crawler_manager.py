"""
Gestor del crawler para poblar la base de datos automáticamente
"""

import chromadb
import sys
import os
from typing import List, Dict

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from chroma_db_manager import ChromaDBManager
from .crawler import TourismCrawler
from .nltk_manager import NLTKManager

class CrawlerManager:
    """
    Gestor que coordina el crawler para poblar automáticamente la base de datos
    """
    
    def __init__(self, chroma_db_path="db/"):
        """
        Inicializa el gestor del crawler
        
        Args:
            chroma_db_path: Ruta a la base de datos ChromaDB
        """
        self.chroma_db_path = chroma_db_path
        self.chroma_client = chromadb.PersistentClient(path=chroma_db_path)
        self.collection = self.chroma_client.get_or_create_collection(name="tourism_docs")
        self.chroma_manager = ChromaDBManager(self.collection)
        self.nltk_manager = NLTKManager()
    
    def get_tourism_urls_for_city(self, city: str) -> List[str]:
        """
        Obtiene URLs de turismo relevantes para una ciudad específica
        
        Args:
            city: Nombre de la ciudad
            
        Returns:
            Lista de URLs para hacer crawling
        """
        print(f"🔍 Obteniendo URLs de turismo para {city}...")
        
        # URLs mejoradas con fuentes confiables en inglés (mejor para spaCy en inglés)
        city_urls = {
            'Madrid': [
                'https://www.esmadrid.com/en/tourist-information/what-to-see',
                'https://www.spain.info/en/destination/madrid/',
                'https://www.timeout.com/madrid/attractions',
                'https://www.lonelyplanet.com/spain/madrid/attractions',
                'https://www.tripadvisor.com/Attractions-g187514-Activities-Madrid.html',
                'https://www.introducingmadrid.com/attractions',
                'https://en.wikipedia.org/wiki/Tourism_in_Madrid'
            ],
            'Barcelona': [
                'https://www.barcelonaturisme.com/wv3/en/page/97/what-to-do.html',
                'https://www.spain.info/en/destination/barcelona/',
                'https://www.timeout.com/barcelona/attractions',
                'https://www.lonelyplanet.com/spain/barcelona/attractions',
                'https://www.tripadvisor.com/Attractions-g187497-Activities-Barcelona_Catalonia.html',
                'https://www.visitbarcelona.com/en/what-to-do-barcelona',
                'https://en.wikipedia.org/wiki/Tourism_in_Barcelona'
            ],
            'Valencia': [
                'https://www.visitvalencia.com/en/what-to-do-valencia',
                'https://www.spain.info/en/destination/valencia/',
                'https://www.lonelyplanet.com/spain/valencia/attractions',
                'https://www.tripadvisor.com/Attractions-g187529-Activities-Valencia.html',
                'https://www.turisvalencia.es/en/what-to-see',
                'https://en.wikipedia.org/wiki/Tourism_in_Valencia'
            ],
            'Sevilla': [
                'https://www.visitasevilla.es/en/what-to-do',
                'https://www.spain.info/en/destination/seville/',
                'https://www.lonelyplanet.com/spain/seville/attractions',
                'https://www.tripadvisor.com/Attractions-g187443-Activities-Seville_Andalucia.html',
                'https://www.andalucia.org/en/seville-tourism',
                'https://en.wikipedia.org/wiki/Tourism_in_Seville'
            ],
            'Bilbao': [
                'https://www.bilbaoturismo.net/BilbaoTurismo/en/what-to-do',
                'https://www.spain.info/en/destination/bilbao/',
                'https://www.lonelyplanet.com/spain/bilbao/attractions',
                'https://www.tripadvisor.com/Attractions-g187449-Activities-Bilbao_Basque_Country.html',
                'https://www.euskadi.eus/tourism/bilbao/en/',
                'https://en.wikipedia.org/wiki/Tourism_in_Bilbao'
            ],
            'Granada': [
                'https://www.granadatur.com/en/what-to-see/',
                'https://www.spain.info/en/destination/granada/',
                'https://www.lonelyplanet.com/spain/granada/attractions',
                'https://www.tripadvisor.com/Attractions-g187441-Activities-Granada_Andalucia.html',
                'https://en.wikipedia.org/wiki/Tourism_in_Granada'
            ],
            'Toledo': [
                'https://www.toledo-turismo.com/en/what-to-see/',
                'https://www.spain.info/en/destination/toledo/',
                'https://www.lonelyplanet.com/spain/toledo/attractions',
                'https://www.tripadvisor.com/Attractions-g187489-Activities-Toledo_Castile_La_Mancha.html',
                'https://en.wikipedia.org/wiki/Tourism_in_Toledo,_Spain'
            ],
            'Salamanca': [
                'https://www.salamanca.es/en/tourism/',
                'https://www.spain.info/en/destination/salamanca/',
                'https://www.lonelyplanet.com/spain/salamanca/attractions',
                'https://www.tripadvisor.com/Attractions-g187495-Activities-Salamanca_Castile_and_Leon.html',
                'https://en.wikipedia.org/wiki/Tourism_in_Salamanca'
            ]
        }
        
        # URLs específicas de la ciudad o URLs genéricas mejoradas
        if city in city_urls:
            urls = city_urls[city]
            print(f"✅ Encontradas {len(urls)} URLs específicas para {city}")
            return urls
        else:
            # URLs genéricas para ciudades no específicamente configuradas
            generic_urls = [
                f'https://www.spain.info/en/destination/{city.lower()}/',
                f'https://www.lonelyplanet.com/spain/{city.lower()}/attractions',
                f'https://www.tripadvisor.com/Tourism-g{city}-Vacations.html',
                f'https://en.wikipedia.org/wiki/Tourism_in_{city}'
            ]
            print(f"⚠️ Usando {len(generic_urls)} URLs genéricas para {city}")
            return generic_urls
    
    def needs_crawling(self, city: str, min_places: int = 5) -> bool:
        """
        Determina si es necesario hacer crawling para una ciudad
        
        Args:
            city: Nombre de la ciudad
            min_places: Número mínimo de lugares requeridos
            
        Returns:
            True si necesita crawling, False en caso contrario
        """
        try:
            # Contar lugares existentes para la ciudad
            all_docs = self.collection.get(include=["metadatas"])
            city_count = 0
            
            for metadata in all_docs['metadatas']:
                if metadata.get('city', '').lower() == city.lower():
                    city_count += 1
            
            print(f"📊 {city} tiene {city_count} lugares en BD (mínimo requerido: {min_places})")
            return city_count < min_places
            
        except Exception as e:
            print(f"❌ Error verificando necesidad de crawling: {e}")
            return True
    
    def populate_city_data(self, city: str, max_documents: int = 10) -> Dict:
        """
        Pobla la base de datos con información turística de una ciudad
        
        Args:
            city: Nombre de la ciudad
            max_documents: Número máximo de documentos a extraer
            
        Returns:
            Diccionario con estadísticas del crawling
        """
        print(f"🔄 Iniciando crawling para {city}...")
        
        # Obtener URLs relevantes
        urls = self.get_tourism_urls_for_city(city)
        
        if not urls:
            return {
                'success': False,
                'error': f'No se encontraron URLs para {city}',
                'documents_added': 0
            }
        
        # Contar documentos antes del crawling
        initial_count = self.collection.count()
        print(f"📊 Documentos iniciales en BD: {initial_count}")
        
        try:
            # Inicializar crawler con configuración optimizada
            crawler = TourismCrawler(
                start_urls=urls,
                chroma_manager=self.chroma_manager,
                nltk_manager=self.nltk_manager,
                max_documents=max_documents,
                user_agent=f"TourismCrawler/2.0 (Educational; {city})"
            )
            
            print(f"🚀 Ejecutando crawling con {len(urls)} URLs...")
            
            # Ejecutar crawling
            crawler.crawl()
            
            # Contar documentos después del crawling
            final_count = self.collection.count()
            documents_added = final_count - initial_count
            
            print(f"✅ Crawling completado: {documents_added} documentos añadidos")
            
            return {
                'success': True,
                'documents_added': documents_added,
                'total_documents': final_count,
                'urls_processed': len(urls),
                'city': city
            }
            
        except Exception as e:
            print(f"❌ Error durante crawling: {e}")
            return {
                'success': False,
                'error': str(e),
                'documents_added': 0
            }
    
    def ensure_city_data(self, city: str, min_places: int = 5, max_documents: int = 10) -> Dict:
        """
        Asegura que haya suficientes datos para una ciudad, haciendo crawling si es necesario
        
        Args:
            city: Nombre de la ciudad
            min_places: Número mínimo de lugares requeridos
            max_documents: Número máximo de documentos a extraer si es necesario
            
        Returns:
            Diccionario con el resultado de la operación
        """
        print(f"🔍 Verificando datos para {city}...")
        
        if not self.needs_crawling(city, min_places):
            return {
                'success': True,
                'crawling_needed': False,
                'message': f'✅ Ya hay suficientes datos para {city}',
                'documents_added': 0
            }
        
        print(f"🔄 Necesario crawling para {city}")
        
        # Hacer crawling
        result = self.populate_city_data(city, max_documents)
        result['crawling_needed'] = True
        
        if result['success']:
            result['message'] = f'✅ Crawling completado para {city}. Se añadieron {result["documents_added"]} documentos.'
        else:
            result['message'] = f'❌ Error en crawling para {city}: {result.get("error", "Error desconocido")}'
        
        return result
    
    def get_available_cities(self) -> List[str]:
        """
        Obtiene la lista de ciudades disponibles en la base de datos
        
        Returns:
            Lista de nombres de ciudades
        """
        try:
            all_docs = self.collection.get(include=["metadatas"])
            cities = set()
            
            for metadata in all_docs['metadatas']:
                city = metadata.get('city', '')
                if city and city != 'Unknown':
                    cities.add(city)
            
            city_list = sorted(list(cities))
            print(f"📍 Ciudades disponibles: {city_list}")
            return city_list
            
        except Exception as e:
            print(f"❌ Error obteniendo ciudades: {e}")
            return []
    
    def get_city_statistics(self, city: str = None) -> Dict:
        """
        Obtiene estadísticas de la base de datos
        
        Args:
            city: Ciudad específica (opcional)
            
        Returns:
            Diccionario con estadísticas
        """
        try:
            all_docs = self.collection.get(include=["metadatas"])
            
            if city:
                # Estadísticas para una ciudad específica
                city_docs = [m for m in all_docs['metadatas'] if m.get('city', '').lower() == city.lower()]
                
                categories = {}
                sources = set()
                
                for metadata in city_docs:
                    category = metadata.get('category', 'unknown')
                    categories[category] = categories.get(category, 0) + 1
                    
                    source = metadata.get('source_url', '')
                    if source:
                        sources.add(source)
                
                return {
                    'city': city,
                    'total_places': len(city_docs),
                    'categories': categories,
                    'unique_sources': len(sources)
                }
            else:
                # Estadísticas generales
                cities = {}
                categories = {}
                sources = set()
                
                for metadata in all_docs['metadatas']:
                    city_name = metadata.get('city', 'Unknown')
                    cities[city_name] = cities.get(city_name, 0) + 1
                    
                    category = metadata.get('category', 'unknown')
                    categories[category] = categories.get(category, 0) + 1
                    
                    source = metadata.get('source_url', '')
                    if source:
                        sources.add(source)
                
                return {
                    'total_documents': len(all_docs['metadatas']),
                    'cities': cities,
                    'categories': categories,
                    'unique_sources': len(sources)
                }
                
        except Exception as e:
            print(f"❌ Error obteniendo estadísticas: {e}")
            return {}
    
    def test_crawler_sources(self, city: str, max_test_urls: int = 3) -> Dict:
        """
        Prueba las fuentes de crawling para una ciudad sin guardar datos
        
        Args:
            city: Nombre de la ciudad
            max_test_urls: Número máximo de URLs a probar
            
        Returns:
            Diccionario con resultados de la prueba
        """
        print(f"🧪 Probando fuentes de crawling para {city}...")
        
        urls = self.get_tourism_urls_for_city(city)[:max_test_urls]
        results = {
            'city': city,
            'urls_tested': len(urls),
            'successful_urls': [],
            'failed_urls': [],
            'total_content_length': 0
        }
        
        for url in urls:
            try:
                import requests
                from bs4 import BeautifulSoup
                
                print(f"🔍 Probando: {url}")
                
                response = requests.get(
                    url,
                    timeout=15,
                    headers={'User-Agent': 'TourismCrawler/2.0 (Test)'}
                )
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    content_length = len(soup.get_text())
                    
                    results['successful_urls'].append({
                        'url': url,
                        'status_code': response.status_code,
                        'content_length': content_length
                    })
                    results['total_content_length'] += content_length
                    print(f"✅ Éxito: {content_length} caracteres")
                else:
                    results['failed_urls'].append({
                        'url': url,
                        'status_code': response.status_code,
                        'error': f'HTTP {response.status_code}'
                    })
                    print(f"❌ Error HTTP {response.status_code}")
                    
            except Exception as e:
                results['failed_urls'].append({
                    'url': url,
                    'error': str(e)
                })
                print(f"❌ Error: {e}")
        
        success_rate = len(results['successful_urls']) / len(urls) * 100
        print(f"📊 Tasa de éxito: {success_rate:.1f}% ({len(results['successful_urls'])}/{len(urls)})")
        
        return results