import scrapy

class TouristPlaceItem(scrapy.Item):
    name = scrapy.Field()  # nombre del lugar
    city = scrapy.Field()  # ciudades
    description = scrapy.Field()  # descripción
    category = scrapy.Field()  # categoría
    coordinates = scrapy.Field()  # geolocalización@@