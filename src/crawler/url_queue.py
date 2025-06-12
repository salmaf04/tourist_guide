from collections import deque
from typing import Optional, Set

class URLQueue:
    """Cola de URLs para crawling con deduplicación"""
    
    def __init__(self):
        self.queue = deque()
        self.visited = set()
        
    def add(self, url: str):
        """Añade una URL a la cola si no ha sido visitada"""
        if url not in self.visited:
            self.queue.append(url)
            self.visited.add(url)
            
    def pop(self) -> Optional[str]:
        """Obtiene la siguiente URL de la cola"""
        if self.queue:
            return self.queue.popleft()
        return None
        
    def __len__(self) -> int:
        """Retorna el número de URLs en la cola"""
        return len(self.queue)