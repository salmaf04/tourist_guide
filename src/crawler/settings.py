BOT_NAME = 'tourism_crawler'

SPIDER_MODULES = ['crawler']
NEWSPIDER_MODULE = 'crawler'

USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'

# Headers adicionales para parecer más como un navegador real
DEFAULT_REQUEST_HEADERS = {
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
}
ROBOTSTXT_OBEY = True
CONCURRENT_REQUESTS = 1  # Reducir concurrencia para ser más conservador
DOWNLOAD_DELAY = 2.0     # Aumentar delay entre requests
AUTOTHROTTLE_ENABLED = True
AUTOTHROTTLE_START_DELAY = 2.0
AUTOTHROTTLE_MAX_DELAY = 10.0
AUTOTHROTTLE_TARGET_CONCURRENCY = 1.0

# Configuración adicional para mejorar el rendimiento
DUPEFILTER_DEBUG = True
DEPTH_LIMIT = 2  # Limitar profundidad globalmente

ITEM_PIPELINES = {
    'crawler.pipelines.ChromaPipeline': 300,
}