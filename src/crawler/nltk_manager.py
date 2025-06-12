import nltk
from typing import List, Set
import re

class NLTKManager:
    """Manejador de NLTK para procesamiento de texto"""
    
    def __init__(self):
        self._download_required_data()
        self.spanish_cities = self._load_spanish_cities()
        
    def _download_required_data(self):
        """Descarga los datos necesarios de NLTK"""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
            
    def _load_spanish_cities(self) -> Set[str]:
        """Carga una lista de ciudades españolas conocidas"""
        cities = {
            'Madrid', 'Barcelona', 'Valencia', 'Sevilla', 'Zaragoza',
            'Málaga', 'Murcia', 'Palma', 'Las Palmas', 'Bilbao',
            'Alicante', 'Córdoba', 'Valladolid', 'Vigo', 'Gijón',
            'Hospitalet', 'Vitoria', 'Granada', 'Elche', 'Oviedo',
            'Badalona', 'Cartagena', 'Terrassa', 'Jerez', 'Sabadell',
            'Móstoles', 'Santa Cruz', 'Pamplona', 'Almería', 'Fuenlabrada',
            'Toledo', 'Burgos', 'Santander', 'Castellón', 'Getafe',
            'Alcorcón', 'Logroño', 'San Sebastián', 'Badajoz', 'Salamanca',
            'Huelva', 'Marbella', 'Lleida', 'Tarragona', 'León',
            'Cádiz', 'Dos Hermanas', 'Parla', 'Mataró', 'Santa Coloma',
            'Alcalá de Henares', 'Torrejón', 'Reus', 'Ourense', 'Manresa'
        }
        return cities
        
    def _extract_known_cities(self, text: str) -> List[str]:
        """Extrae ciudades conocidas del texto"""
        found_cities = []
        text_upper = text.upper()
        
        for city in self.spanish_cities:
            if city.upper() in text_upper:
                found_cities.append(city)
                
        return found_cities
        
    def tokenize_sentences(self, text: str) -> List[str]:
        """Tokeniza el texto en oraciones"""
        return nltk.sent_tokenize(text, language='spanish')
        
    def tokenize_words(self, text: str) -> List[str]:
        """Tokeniza el texto en palabras"""
        return nltk.word_tokenize(text, language='spanish')