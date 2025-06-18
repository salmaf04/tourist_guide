"""
Módulo crawler reorganizado para el sistema de guía turística.
"""

from .tourist_spider import TouristSpider
from .config import CRAWLER_CONFIG, CITY_URLS, CITY_URLS_EXTRA
from .filters import URLFilter, ContentFilter
from .city_utils import CityIdentifier
from .content_processor import ContentProcessor
from .stats_manager import StatsManager

__all__ = [
    'TouristSpider',
    'CRAWLER_CONFIG',
    'CITY_URLS', 
    'CITY_URLS_EXTRA',
    'URLFilter',
    'ContentFilter',
    'CityIdentifier',
    'ContentProcessor',
    'StatsManager'
]