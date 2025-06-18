"""
Gestor de estadísticas y control de diversidad para el crawler.
"""


class StatsManager:
    """Clase para manejar estadísticas y control de diversidad."""
    
    def __init__(self, max_pages_per_city=3, max_places_per_city=8, logger=None):
        self.max_pages_per_city = max_pages_per_city
        self.max_places_per_city = max_places_per_city
        self.logger = logger
        
        # Contadores
        self.city_page_count = {}
        self.city_place_count = {}
        self.crawled_pages = 0
    
    def increment_page_count(self, city):
        """Incrementa el contador de páginas para una ciudad."""
        if city != "Unknown":
            self.city_page_count[city] = self.city_page_count.get(city, 0) + 1
    
    def increment_place_count(self, city):
        """Incrementa el contador de lugares para una ciudad."""
        if city != "Unknown":
            self.city_place_count[city] = self.city_place_count.get(city, 0) + 1
    
    def increment_crawled_pages(self):
        """Incrementa el contador total de páginas crawleadas."""
        self.crawled_pages += 1
    
    def can_process_city_page(self, city):
        """Verifica si se puede procesar una página más de la ciudad."""
        if city == "Unknown":
            return True
        
        current_pages = self.city_page_count.get(city, 0)
        return current_pages < self.max_pages_per_city
    
    def can_add_city_place(self, city):
        """Verifica si se puede agregar un lugar más de la ciudad."""
        if city == "Unknown":
            return True
        
        current_places = self.city_place_count.get(city, 0)
        return current_places < self.max_places_per_city
    
    def get_city_page_count(self, city):
        """Obtiene el número de páginas procesadas para una ciudad."""
        return self.city_page_count.get(city, 0)
    
    def get_city_place_count(self, city):
        """Obtiene el número de lugares agregados para una ciudad."""
        return self.city_place_count.get(city, 0)
    
    def get_most_common_city(self):
        """Obtiene la ciudad más común en esta sesión de crawling."""
        try:
            if not self.city_place_count:
                return None
            
            # Excluir "Unknown" del conteo
            valid_cities = {city: count for city, count in self.city_place_count.items() if city != "Unknown"}
            
            if not valid_cities:
                return None
            
            most_common = max(valid_cities.items(), key=lambda x: x[1])
            return most_common[0]
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error obteniendo ciudad más común: {e}")
            return None
    
    def calculate_priority_with_diversity(self, link, depth, city):
        """Calcula prioridad favoreciendo diversidad de ciudades."""
        priority = 5
        priority -= depth * 2
        
        # Bonificación por contenido turístico
        link_text = link.get_text().lower()
        if any(term in link_text for term in ['attraction', 'place', 'museum', 'tour', 'visit']):
            priority += 3
        
        parent_classes = ' '.join(link.parent.get('class', []))
        if any(cls in parent_classes for cls in ['list', 'item', 'card', 'attraction']):
            priority += 2
        
        # BONIFICACIÓN POR DIVERSIDAD
        if city != "Unknown":
            city_pages = self.city_page_count.get(city, 0)
            city_places = self.city_place_count.get(city, 0)
            
            # Bonificar ciudades con poco contenido
            if city_pages == 0:
                priority += 5  # Ciudad nueva, máxima prioridad
            elif city_pages < 3:
                priority += 3  # Ciudad con poco contenido
            elif city_places < 5:
                priority += 2  # Ciudad con pocos lugares
            else:
                priority -= 1  # Ciudad con mucho contenido, menor prioridad
        
        return max(1, min(15, priority))
    
    def log_diversity_stats(self):
        """Muestra estadísticas de diversidad de ciudades."""
        if not self.logger:
            return
        
        self.logger.info("📊 ESTADÍSTICAS DE DIVERSIDAD:")
        self.logger.info(f"   Total páginas procesadas: {self.crawled_pages}")
        
        if self.city_page_count:
            self.logger.info("   Páginas por ciudad:")
            for city, count in sorted(self.city_page_count.items(), key=lambda x: x[1], reverse=True):
                self.logger.info(f"     {city}: {count} páginas")
        
        if self.city_place_count:
            self.logger.info("   Lugares por ciudad:")
            for city, count in sorted(self.city_place_count.items(), key=lambda x: x[1], reverse=True):
                self.logger.info(f"     {city}: {count} lugares")
        
        total_cities = len(self.city_place_count)
        unknown_count = self.city_place_count.get("Unknown", 0)
        valid_cities = total_cities - (1 if unknown_count > 0 else 0)
        
        self.logger.info(f"   Total ciudades con contenido: {total_cities}")
        self.logger.info(f"   Ciudades válidas: {valid_cities}")
        self.logger.info(f"   Lugares sin ciudad identificada: {unknown_count}")
    
    def should_log_stats(self):
        """Determina si se deben mostrar las estadísticas."""
        return self.crawled_pages % 10 == 0
    
    def get_summary_stats(self):
        """Obtiene un resumen de las estadísticas."""
        return {
            'total_pages': self.crawled_pages,
            'cities_with_pages': len(self.city_page_count),
            'cities_with_places': len(self.city_place_count),
            'total_places': sum(self.city_place_count.values()),
            'city_page_distribution': dict(self.city_page_count),
            'city_place_distribution': dict(self.city_place_count)
        }