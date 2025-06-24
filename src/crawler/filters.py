"""
Filtros y validadores para el crawler turístico.
"""

from urllib.parse import urlparse
from .config import (
    BLOCKED_DOMAINS, EXCLUDED_PATTERNS, 
    EXCLUDED_COUNTRIES, SPANISH_DOMAINS, TOURIST_TERMS,
    CITY_URLS, CITY_URLS_EXTRA
)


class URLFilter:
    """Clase para filtrar URLs según diferentes criterios."""
    
    def __init__(self, logger=None):
        self.logger = logger
        self.all_city_urls = {**CITY_URLS, **CITY_URLS_EXTRA}
    
        
    def is_blocked_domain(self, url):
        """Verifica si la URL pertenece a un dominio bloqueado."""
        url_lower = url.lower()
        
        # Verificar dominios bloqueados
        for blocked_domain in BLOCKED_DOMAINS:
            if blocked_domain in url_lower:
                if self.logger:
                    self.logger.debug(f"🚫 URL BLOQUEADA por dominio: {url} (dominio: {blocked_domain})")
                return True
        
        # Verificar patrones de Wikipedia
        wikipedia_patterns = [
            'wikipedia.org', 'wikimedia.org', 'wikidata.org', '.wiki',
            '/wiki/', 'en.wikipedia', 'es.wikipedia', 'ca.wikipedia',
            'gl.wikipedia', 'eu.wikipedia', 'ast.wikipedia'
        ]
        for pattern in wikipedia_patterns:
            if pattern in url_lower:
                if self.logger:
                    self.logger.debug(f"🚫 URL BLOQUEADA por patrón Wikipedia: {url} (patrón: {pattern})")
                return True
        
        return False
    
    def has_excluded_patterns(self, url):
        """Verifica si la URL contiene patrones excluidos."""
        url_lower = url.lower()
        
        for pattern in EXCLUDED_PATTERNS:
            if isinstance(pattern, str) and pattern in url_lower:
                if self.logger:
                    self.logger.debug(f"URL excluida por patrón: {url} (patrón: {pattern})")
                return True
        
        return False
    
    def has_excluded_countries(self, url):
        """Verifica si la URL contiene países excluidos."""
        url_lower = url.lower()
        
        for country in EXCLUDED_COUNTRIES:
            if isinstance(country, str) and country in url_lower:
                if self.logger:
                    self.logger.debug(f"URL excluida por país no español: {url} (contiene: {country})")
                return True
        
        return False
    
    def is_spanish_domain_allowed(self, url):
        """Verifica si el dominio está permitido."""
        url_lower = url.lower()
        
        for domain in SPANISH_DOMAINS:
            if isinstance(domain, str) and domain in url_lower:
                return True
        
        if self.logger:
            self.logger.debug(f"URL excluida por dominio no permitido: {url}")
        return False
    
    def is_url_relevant(self, url, city):
        """Verifica si la URL es relevante para la ciudad y es de España."""
        try:
            # Filtro 1: Dominios bloqueados
            if self.is_blocked_domain(url):
                return False
            
            # Filtro 2: Patrones excluidos
            if self.has_excluded_patterns(url):
                return False
            
            # Filtro 3: Países excluidos
            if self.has_excluded_countries(url):
                return False
            
            # Filtro 4: Dominios permitidos
            if not self.is_spanish_domain_allowed(url):
                return False
            
            # Filtro 5: Relevancia para la ciudad
            city_lower = city.lower()
            url_lower = url.lower()
            
            city_in_url = city_lower in url_lower
            has_tourist_terms = any(
                isinstance(term, str) and term in url_lower 
                for term in TOURIST_TERMS
            )
            
            result = city_in_url or has_tourist_terms
            
            if result:
                if self.logger:
                    self.logger.debug(f"URL aceptada: {url} (ciudad: {city})")
            else:
                if self.logger:
                    self.logger.debug(f"URL rechazada: {url} (ciudad: {city})")
            
            return result
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error en is_url_relevant para {url}: {e}")
            return False


class ContentFilter:
    """Clase para filtrar contenido extraído."""
    
    def __init__(self, logger=None):
        self.logger = logger
    
        
    def validate_place_data(self, place_data):
        """Valida que los datos del lugar sean completos y válidos."""
        if not place_data:
            return False
        
        # Verificar campos obligatorios
        required_fields = ['nombre', 'ciudad', 'categoria', 'descripcion']
        for field in required_fields:
            if not place_data.get(field) or place_data.get(field).strip() == "":
                if self.logger:
                    self.logger.warning(f"Lugar con campo {field} vacío: {place_data.get('nombre', 'N/A')}")
                return False
        
        return True