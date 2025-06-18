"""
Utilidades para identificación y manejo de ciudades.
"""

from .config import CITY_URLS, CITY_URLS_EXTRA


class CityIdentifier:
    """Clase para identificar ciudades desde URLs y contenido."""
    
    def __init__(self, logger=None):
        self.logger = logger
        self.all_city_urls = {**CITY_URLS, **CITY_URLS_EXTRA}
        
        # Mapeo de dominios específicos a ciudades
        self.domain_to_city = {
            # Barcelona
            'barcelonaturisme.com': 'Barcelona',
            'barcelona.cat': 'Barcelona',
            'visitbarcelona.com': 'Barcelona',
            'bcnshop.barcelonaturisme.com': 'Barcelona',
            'spain.info/en/destination/barcelona': 'Barcelona',
            'lonelyplanet.com/spain/barcelona': 'Barcelona',
            'timeout.com/barcelona': 'Barcelona',
            # Valencia
            'visitvalencia.com': 'Valencia',
            'valencia.es': 'Valencia',
            'turismovalencia.es': 'Valencia',
            'spain.info/en/destination/valencia': 'Valencia',
            'lonelyplanet.com/spain/valencia': 'Valencia',
            'timeout.com/valencia': 'Valencia',
            # Sevilla
            'visitasevilla.es': 'Sevilla',
            'sevilla.org': 'Sevilla',
            'spain.info/en/destination/seville': 'Sevilla',
            'lonelyplanet.com/spain/andalucia/seville': 'Sevilla',
            'timeout.com/seville': 'Sevilla',
            # Bilbao
            'bilbaoturismo.net': 'Bilbao',
            'bilbao.eus': 'Bilbao',
            'spain.info/en/destination/bilbao': 'Bilbao',
            'lonelyplanet.com/spain/basque-country/bilbao': 'Bilbao',
            'timeout.com/bilbao': 'Bilbao',
            # Granada
            'granadatur.com': 'Granada',
            'granada.org': 'Granada',
            'spain.info/en/destination/granada': 'Granada',
            'lonelyplanet.com/spain/andalucia/granada': 'Granada',
            'timeout.com/granada': 'Granada',
            # Toledo
            'toledo-turismo.com': 'Toledo',
            'ayto-toledo.org': 'Toledo',
            'spain.info/en/destination/toledo': 'Toledo',
            'lonelyplanet.com/spain/castilla-la-mancha/toledo': 'Toledo',
            'timeout.com/toledo': 'Toledo',
            # Salamanca
            'salamanca.es': 'Salamanca',
            'turismodesalamanca.com': 'Salamanca',
            'turismosalamanaca.com': 'Salamanca',
            'spain.info/en/destination/salamanca': 'Salamanca',
            'lonelyplanet.com/spain/castilla-y-leon/salamanca': 'Salamanca',
            'timeout.com/salamanca': 'Salamanca',
            # Málaga
            'malagaturismo.com': 'Málaga',
            'malaga.eu': 'Málaga',
            'spain.info/en/destination/malaga': 'Málaga',
            'lonelyplanet.com/spain/andalucia/malaga': 'Málaga',
            'timeout.com/malaga': 'Málaga',
            # San Sebastián
            'sansebastianturismo.com': 'San Sebastián',
            'donostia.eus': 'San Sebastián',
            'spain.info/en/destination/san-sebastian': 'San Sebastián',
            'lonelyplanet.com/spain/basque-country/san-sebastian': 'San Sebastián',
            'timeout.com/san-sebastian': 'San Sebastián',
            # Córdoba
            'turismodecordoba.org': 'Córdoba',
            'cordoba.es': 'Córdoba',
            'spain.info/en/destination/cordoba': 'Córdoba',
            'lonelyplanet.com/spain/andalucia/cordoba': 'Córdoba',
            'timeout.com/cordoba': 'Córdoba',
            # Zaragoza
            'zaragoza.es': 'Zaragoza',
            'turismodezaragoza.es': 'Zaragoza',
            'spain.info/en/destination/zaragoza': 'Zaragoza',
            'lonelyplanet.com/spain/aragon/zaragoza': 'Zaragoza',
            'timeout.com/zaragoza': 'Zaragoza',
            # Santander
            'turismodesantander.com': 'Santander',
            'santander.es': 'Santander',
            'spain.info/en/destination/santander': 'Santander',
            'lonelyplanet.com/spain/cantabria/santander': 'Santander',
            'timeout.com/santander': 'Santander',
            # Cádiz
            'turismo.cadiz.es': 'Cádiz',
            'cadizturismo.com': 'Cádiz',
            'spain.info/en/destination/cadiz': 'Cádiz',
            'lonelyplanet.com/spain/andalucia/cadiz': 'Cádiz',
            'timeout.com/cadiz': 'Cádiz',
            # Murcia
            'murciaturistica.es': 'Murcia',
            'murcia.es': 'Murcia',
            'spain.info/en/destination/murcia': 'Murcia',
            'lonelyplanet.com/spain/murcia': 'Murcia',
            'timeout.com/murcia': 'Murcia',
            # Palma de Mallorca
            'visitpalma.com': 'Palma de Mallorca',
            'palma.cat': 'Palma de Mallorca',
            'spain.info/en/destination/palma-de-mallorca': 'Palma de Mallorca',
            'lonelyplanet.com/spain/balearic-islands/palma-de-mallorca': 'Palma de Mallorca',
            'timeout.com/palma': 'Palma de Mallorca',
            # Ciudades adicionales
            'visitbenidorm.es': 'Benidorm',
            'santiagoturismo.com': 'Santiago de Compostela',
            'alicanteturismo.com': 'Alicante',
            'marbella.es': 'Marbella',
            'visitsalou.eu': 'Salou',
            'ibiza.travel': 'Ibiza',
            'girona.cat': 'Girona',
            'turismoleon.org': 'León',
            'turismodevigo.org': 'Vigo',
            'coruna.gal': 'A Coruña',
            'oviedo.es': 'Oviedo'
        }
        
        # Lugares famosos y sus ciudades
        self.famous_places = {
            # Barcelona - Expandido
            'sagrada familia': 'Barcelona', 'park guell': 'Barcelona', 'casa batllo': 'Barcelona',
            'casa mila': 'Barcelona', 'la pedrera': 'Barcelona', 'picasso museum': 'Barcelona',
            'camp nou': 'Barcelona', 'las ramblas': 'Barcelona', 'barrio gotico': 'Barcelona',
            'eixample': 'Barcelona', 'gracia': 'Barcelona', 'barceloneta': 'Barcelona',
            'montjuic': 'Barcelona', 'tibidabo': 'Barcelona', 'macba': 'Barcelona',
            'museu nacional': 'Barcelona', 'palau sant jordi': 'Barcelona',
            'fundacio miro': 'Barcelona', 'palau musica catalana': 'Barcelona',
            
            # Sevilla - Expandido
            'alcazar sevilla': 'Sevilla', 'giralda': 'Sevilla', 'catedral sevilla': 'Sevilla',
            'barrio santa cruz': 'Sevilla', 'triana': 'Sevilla', 'maestranza': 'Sevilla',
            'archivo indias': 'Sevilla', 'museo bellas artes sevilla': 'Sevilla',
            'plaza españa sevilla': 'Sevilla', 'parque maria luisa': 'Sevilla',
            
            # Valencia - Expandido
            'ciudad artes ciencias': 'Valencia', 'oceanografic': 'Valencia',
            'palau arts': 'Valencia', 'mercado central valencia': 'Valencia',
            'lonja valencia': 'Valencia', 'torres serranos': 'Valencia',
            'museo fallero': 'Valencia', 'ivam': 'Valencia',
            'catedral valencia': 'Valencia', 'miguelete': 'Valencia',
            
            # Bilbao - Expandido
            'guggenheim': 'Bilbao', 'guggenheim bilbao': 'Bilbao',
            'casco viejo bilbao': 'Bilbao', 'puente colgante': 'Bilbao',
            'museo bellas artes bilbao': 'Bilbao', 'teatro arriaga': 'Bilbao',
            'san mames': 'Bilbao', 'mercado ribera': 'Bilbao',
            
            # Granada - Expandido
            'alhambra': 'Granada', 'generalife': 'Granada', 'albaicin': 'Granada',
            'sacromonte': 'Granada', 'capilla real granada': 'Granada',
            'catedral granada': 'Granada', 'monasterio cartuja': 'Granada',
            'museo arqueologico granada': 'Granada', 'casa zafra': 'Granada',
            
            # Málaga - Expandido
            'picasso malaga': 'Málaga', 'alcazaba malaga': 'Málaga',
            'teatro romano malaga': 'Málaga', 'pompidou malaga': 'Málaga',
            'catedral malaga': 'Málaga', 'gibralfaro': 'Málaga',
            'museo thyssen malaga': 'Málaga', 'cac malaga': 'Málaga',
            
            # Toledo - Expandido
            'alcazar toledo': 'Toledo', 'catedral toledo': 'Toledo',
            'sinagoga transito': 'Toledo', 'casa greco': 'Toledo',
            'monasterio san juan reyes': 'Toledo', 'puerta bisagra': 'Toledo',
            'museo santa cruz': 'Toledo', 'sinagoga maria blanca': 'Toledo',
            
            # Córdoba - Expandido
            'mezquita': 'Córdoba', 'mezquita cordoba': 'Córdoba',
            'alcazar cordoba': 'Córdoba', 'juderia cordoba': 'Córdoba',
            'palacio viana': 'Córdoba', 'museo arqueologico cordoba': 'Córdoba',
            'calleja flores': 'Córdoba', 'puente romano cordoba': 'Córdoba',
            
            # Santiago de Compostela - Expandido
            'catedral santiago': 'Santiago de Compostela', 'obradoiro': 'Santiago de Compostela',
            'camino santiago': 'Santiago de Compostela', 'portico gloria': 'Santiago de Compostela',
            'hostal reyes catolicos': 'Santiago de Compostela', 'pazo raxoi': 'Santiago de Compostela',
            
            # San Sebastián - Expandido
            'kursaal': 'San Sebastián', 'monte igueldo': 'San Sebastián',
            'playa concha': 'San Sebastián', 'parte vieja': 'San Sebastián',
            'monte urgull': 'San Sebastián', 'aquarium': 'San Sebastián',
            'palacio miramar': 'San Sebastián', 'zurriola': 'San Sebastián',
            
            # Ciudades adicionales
            'benidorm': 'Benidorm', 'terra mitica': 'Benidorm',
            'alicante': 'Alicante', 'castillo santa barbara': 'Alicante',
            'marbella': 'Marbella', 'puerto banus': 'Marbella',
            'salou': 'Salou', 'portaventura': 'Salou',
            'ibiza': 'Ibiza', 'dalt vila': 'Ibiza',
            'girona': 'Girona', 'barrio judio girona': 'Girona',
            'leon': 'León', 'catedral leon': 'León',
            'pamplona': 'Pamplona', 'sanfermines': 'Pamplona',
            'vigo': 'Vigo', 'castro vigo': 'Vigo',
            'coruña': 'A Coruña', 'torre hercules': 'A Coruña',
            'oviedo': 'Oviedo', 'catedral oviedo': 'Oviedo'
        }
    
    def get_city_from_url(self, url):
        """Obtiene la ciudad asociada a la URL con mejor precisión."""
        try:
            # PASO 1: Coincidencia exacta con las URLs base
            for city_name, urls in self.all_city_urls.items():
                for city_url in urls:
                    if url.startswith(city_url):
                        if self.logger:
                            self.logger.debug(f"Ciudad identificada por URL exacta: {city_name} para {url}")
                        return city_name
            
            # PASO 2: Buscar por dominio específico
            url_lower = url.lower()
            
            for domain, city in self.domain_to_city.items():
                if domain in url_lower:
                    if self.logger:
                        self.logger.debug(f"Ciudad identificada por dominio: {city} para {url}")
                    return city
            
            # PASO 3: Buscar por nombre de ciudad en la URL
            for city_name, urls in self.all_city_urls.items():
                city_variations = [
                    city_name.lower(),
                    city_name.lower().replace(' ', '-'),
                    city_name.lower().replace(' ', '_'),
                    city_name.lower().replace('ñ', 'n'),
                    city_name.lower().replace('á', 'a').replace('é', 'e').replace('í', 'i').replace('ó', 'o').replace('ú', 'u')
                ]
                
                # Casos especiales
                if city_name == "San Sebastián":
                    city_variations.extend(['donostia', 'san-sebastian', 'sansebastian'])
                elif city_name == "A Coruña":
                    city_variations.extend(['coruna', 'a-coruna', 'corunna'])
                elif city_name == "Palma de Mallorca":
                    city_variations.extend(['palma', 'mallorca'])
                elif city_name == "Santiago de Compostela":
                    city_variations.extend(['santiago', 'compostela'])
                
                for variation in city_variations:
                    if variation and len(variation) > 2:
                        if (f"/{variation}/" in url_lower or 
                            f"/{variation}" in url_lower or 
                            f"{variation}/" in url_lower or
                            variation in url_lower.split('/')[-1] or
                            variation in url_lower.split('.')[-2]):
                            
                            if self.logger:
                                self.logger.debug(f"Ciudad identificada por nombre: {city_name} para {url} (variación: {variation})")
                            return city_name
            
            # PASO 4: Para sitios generales
            if any(site in url_lower for site in ['timeout.com', 'lonelyplanet.com']):
                for city_name in self.all_city_urls.keys():
                    city_lower = city_name.lower().replace(' ', '-').replace('ñ', 'n')
                    if city_lower in url_lower:
                        if self.logger:
                            self.logger.debug(f"Ciudad identificada en sitio general: {city_name} para {url}")
                        return city_name
            
            if self.logger:
                self.logger.debug(f"No se pudo identificar ciudad para URL: {url}")
            return "Unknown"
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error en get_city_from_url para {url}: {e}")
            return "Unknown"
    
    def infer_city_from_context(self, place_name, url, current_city):
        """Infiere la ciudad de un lugar basándose en el contexto."""
        try:
            # PASO 1: Si ya tenemos una ciudad válida
            if current_city and current_city != "Unknown":
                place_lower = place_name.lower()
                current_lower = current_city.lower()
                
                if current_lower in place_lower or any(
                    var in place_lower for var in [
                        current_lower.replace(' ', ''),
                        current_lower.replace(' ', '-'),
                        current_lower.replace('ñ', 'n')
                    ]
                ):
                    return current_city
            
            # PASO 2: Buscar en lugares famosos
            place_lower = place_name.lower()
            for famous_place, city_name in self.famous_places.items():
                if famous_place in place_lower:
                    if self.logger:
                        self.logger.debug(f"Ciudad inferida por lugar famoso: {city_name} para {place_name}")
                    return city_name
            
            # PASO 3: Extraer del nombre del lugar
            if '(' in place_name and ')' in place_name:
                city_match = place_name[place_name.rfind('(')+1:place_name.rfind(')')]
                if city_match and len(city_match) > 2:
                    all_cities = list(self.all_city_urls.keys())
                    for known_city in all_cities:
                        if city_match.lower() == known_city.lower():
                            if self.logger:
                                self.logger.debug(f"Ciudad inferida del nombre: {known_city} para {place_name}")
                            return known_city
            
            # PASO 4: Buscar nombres de ciudades en el lugar
            all_cities = list(self.all_city_urls.keys())
            for city_name in all_cities:
                city_variations = [
                    city_name.lower(),
                    city_name.lower().replace(' ', ''),
                    city_name.lower().replace('ñ', 'n'),
                    city_name.lower().replace('á', 'a').replace('é', 'e').replace('í', 'i').replace('ó', 'o').replace('ú', 'u')
                ]
                
                for variation in city_variations:
                    if variation in place_lower and len(variation) > 3:
                        if self.logger:
                            self.logger.debug(f"Ciudad inferida por nombre en lugar: {city_name} para {place_name}")
                        return city_name
            
            # PASO 5: Usar ciudad actual si es válida
            if current_city and current_city != "Unknown":
                return current_city
            
            # PASO 6: Inferir de la URL
            url_city = self.get_city_from_url(url)
            if url_city and url_city != "Unknown":
                if self.logger:
                    self.logger.debug(f"Ciudad inferida de URL: {url_city} para {place_name}")
                return url_city
            
            if self.logger:
                self.logger.debug(f"No se pudo inferir ciudad para: {place_name}")
            return "Unknown"
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error infiriendo ciudad para {place_name}: {e}")
            return current_city if current_city != "Unknown" else "Unknown"
    
    def extract_city_from_place_name(self, place_name):
        """Extrae la ciudad del nombre del lugar usando patrones comunes."""
        try:
            if not place_name:
                return "Unknown"
            
            place_lower = place_name.lower()
            all_cities = list(self.all_city_urls.keys())
            
            for city_name in all_cities:
                city_variations = [
                    city_name.lower(),
                    city_name.lower().replace(' ', ''),
                    city_name.lower().replace(' ', '-'),
                    city_name.lower().replace('ñ', 'n'),
                    city_name.lower().replace('á', 'a').replace('é', 'e').replace('í', 'i').replace('ó', 'o').replace('ú', 'u')
                ]
                
                # Casos especiales
                if city_name == "San Sebastián":
                    city_variations.extend(['donostia', 'san sebastian'])
                elif city_name == "A Coruña":
                    city_variations.extend(['coruna', 'corunna'])
                elif city_name == "Palma de Mallorca":
                    city_variations.extend(['palma', 'mallorca'])
                elif city_name == "Santiago de Compostela":
                    city_variations.extend(['santiago', 'compostela'])
                
                for variation in city_variations:
                    if variation and len(variation) > 2 and variation in place_lower:
                        if self.logger:
                            self.logger.debug(f"Ciudad extraída del nombre: {city_name} de {place_name}")
                        return city_name
            
            return "Unknown"
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error extrayendo ciudad de {place_name}: {e}")
            return "Unknown"