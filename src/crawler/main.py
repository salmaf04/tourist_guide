#!/usr/bin/env python3
"""
Script principal para ejecutar el crawler turístico refactorizado.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Agregar el directorio src al path para importaciones
current_dir = Path(__file__).parent
src_dir = current_dir.parent
sys.path.insert(0, str(src_dir))

try:
    import scrapy
    from scrapy.crawler import CrawlerProcess
    from scrapy.utils.project import get_project_settings
except ImportError:
    print("❌ Error: Scrapy no está instalado.")
    print("💡 Instala Scrapy con: pip install scrapy")
    sys.exit(1)

from crawler.tourist_spider import TouristSpider
from crawler.config import CITY_URLS, CITY_URLS_EXTRA


def setup_logging():
    """Configura el logging para el crawler."""
    # Configurar handler para archivo con UTF-8
    file_handler = logging.FileHandler('crawler_execution.log', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # Configurar handler para consola sin emojis
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formato común
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Configurar logger root
    logging.basicConfig(
        level=logging.INFO,
        handlers=[file_handler, console_handler]
    )
    
    return logging.getLogger(__name__)


def get_available_cities():
    """Retorna la lista de ciudades disponibles."""
    all_cities = list(CITY_URLS.keys()) + list(CITY_URLS_EXTRA.keys())
    # Filtrar Madrid
    return [city for city in all_cities if city.lower() != 'madrid']


def create_scrapy_settings():
    """Crea la configuración personalizada para Scrapy."""
    settings = {
        'BOT_NAME': 'tourist_crawler',
        'SPIDER_MODULES': ['crawler'],
        'NEWSPIDER_MODULE': 'crawler',
        
        # Configuración de robots.txt
        'ROBOTSTXT_OBEY': False,
        
        # Configuración de User-Agent
        'USER_AGENT': 'tourist_crawler (+http://www.yourdomain.com)',
        
        # Configuración de delays y concurrencia
        'DOWNLOAD_DELAY': 2,
        'RANDOMIZE_DOWNLOAD_DELAY': 0.5,
        'CONCURRENT_REQUESTS': 8,
        'CONCURRENT_REQUESTS_PER_DOMAIN': 2,
        
        # AutoThrottle
        'AUTOTHROTTLE_ENABLED': True,
        'AUTOTHROTTLE_START_DELAY': 2.0,
        'AUTOTHROTTLE_MAX_DELAY': 10.0,
        'AUTOTHROTTLE_TARGET_CONCURRENCY': 2.0,
        'AUTOTHROTTLE_DEBUG': False,
        
        # Configuración de cookies y headers
        'COOKIES_ENABLED': True,
        'DEFAULT_REQUEST_HEADERS': {
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        },
        
        # Configuración de pipelines
        'ITEM_PIPELINES': {
            'crawler.pipelines.ChromaPipeline': 300,
        },
        
        # Configuración de logging
        'LOG_LEVEL': 'INFO',
        'LOG_FILE': 'crawler_execution.log',
        
        # Configuración de duración
        'CLOSESPIDER_TIMEOUT': 3600,  # 1 hora máximo
        'CLOSESPIDER_PAGECOUNT': 100,  # Máximo 100 páginas por defecto
        
        # Configuración de memoria
        'MEMUSAGE_ENABLED': True,
        'MEMUSAGE_LIMIT_MB': 2048,
        'MEMUSAGE_WARNING_MB': 1024,
        
        # Configuración de duplicados
        'DUPEFILTER_CLASS': 'scrapy.dupefilters.RFPDupeFilter',
        'DUPEFILTER_DEBUG': False,
        
        # Configuración de retry
        'RETRY_ENABLED': True,
        'RETRY_TIMES': 2,
        'RETRY_HTTP_CODES': [500, 502, 503, 504, 408, 429],
        
        # Configuración de redirecciones
        'REDIRECT_ENABLED': True,
        'REDIRECT_MAX_TIMES': 3,
    }
    
    return settings


def run_crawler(target_city=None, max_pages=None):
    """
    Ejecuta el crawler con los parámetros especificados.
    
    Args:
        target_city (str, optional): Ciudad específica a crawlear
        max_pages (int, optional): Número máximo de páginas a procesar
    """
    logger = setup_logging()
    
    # Validar ciudad si se especifica
    if target_city:
        available_cities = get_available_cities()
        if target_city not in available_cities:
            logger.error(f"❌ Ciudad '{target_city}' no disponible.")
            logger.info(f"💡 Ciudades disponibles: {', '.join(available_cities)}")
            return False
    
    # Configurar Scrapy
    settings = create_scrapy_settings()
    
    # Ajustar límite de páginas si se especifica
    if max_pages:
        settings['CLOSESPIDER_PAGECOUNT'] = max_pages
    
    # Crear proceso de crawler
    process = CrawlerProcess(settings)
    
    # Configurar argumentos del spider
    spider_kwargs = {}
    if target_city:
        spider_kwargs['target_city'] = target_city
    if max_pages:
        spider_kwargs['max_pages'] = max_pages
    
    logger.info("Iniciando crawler turístico...")
    if target_city:
        logger.info(f"Ciudad objetivo: {target_city}")
    else:
        logger.info("Procesando todas las ciudades disponibles")
    
    if max_pages:
        logger.info(f"Límite de páginas: {max_pages}")
    
    try:
        # Agregar spider al proceso
        process.crawl(TouristSpider, **spider_kwargs)
        
        # Iniciar el crawler
        process.start()
        
        logger.info("Crawler completado exitosamente")
        return True
        
    except Exception as e:
        logger.error(f"Error ejecutando el crawler: {e}")
        return False


def main():
    """Función principal del script."""
    parser = argparse.ArgumentParser(
        description='Ejecuta el crawler turístico refactorizado',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python main.py                           # Crawlear todas las ciudades
  python main.py -c Barcelona              # Crawlear solo Barcelona
  python main.py -c Valencia -p 20         # Crawlear Valencia con límite de 20 páginas
  python main.py -p 50                     # Crawlear todas las ciudades con límite de 50 páginas
  python main.py --list-cities             # Mostrar ciudades disponibles
        """
    )
    
    parser.add_argument(
        '-c', '--city',
        type=str,
        help='Ciudad específica a crawlear (ej: Barcelona, Valencia, Sevilla)'
    )
    
    parser.add_argument(
        '-p', '--pages',
        type=int,
        help='Número máximo de páginas a procesar (por defecto: configuración del spider)'
    )
    
    parser.add_argument(
        '--list-cities',
        action='store_true',
        help='Mostrar lista de ciudades disponibles y salir'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Mostrar información detallada de ejecución'
    )
    
    args = parser.parse_args()
    
    # Mostrar ciudades disponibles
    if args.list_cities:
        cities = get_available_cities()
        print("Ciudades disponibles para crawlear:")
        print("=" * 50)
        for i, city in enumerate(sorted(cities), 1):
            print(f"{i:2d}. {city}")
        print(f"\nTotal: {len(cities)} ciudades")
        return
    
    # Configurar nivel de logging si es verbose
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Ejecutar crawler
    success = run_crawler(target_city=args.city, max_pages=args.pages)
    
    if success:
        print("\n¡Crawler ejecutado exitosamente!")
        print("Revisa los logs en 'crawler_execution.log' para más detalles")
        print("Los datos se han guardado en ChromaDB")
    else:
        print("\nEl crawler falló. Revisa los logs para más información.")
        sys.exit(1)


if __name__ == "__main__":
    main()