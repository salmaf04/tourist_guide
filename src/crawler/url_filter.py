from urllib.parse import urlparse
from typing import List
import re

class URLFilter:
    """Filtro de URLs basado en robots.txt y dominio"""
    
    def __init__(self, robots_info, domain: str):
        self.robots_info = robots_info
        self.domain = domain
        
    def is_allowed(self, url: str) -> bool:
        """Verifica si una URL está permitida para crawling"""
        parsed_url = urlparse(url)
        
        # Solo permitir URLs del mismo dominio
        if parsed_url.netloc != self.domain:
            return False
            
        # Verificar robots.txt
        path = parsed_url.path
        
        # Verificar rutas no permitidas
        for disallowed in self.robots_info.disallowed:
            if path.startswith(disallowed):
                return False
                
        # Verificar rutas permitidas específicamente
        if self.robots_info.allowed:
            for allowed in self.robots_info.allowed:
                if path.startswith(allowed):
                    return True
            return False
            
        return True