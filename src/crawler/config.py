"""
Configuración y constantes para el crawler turístico.
"""

# Configuración general del crawler
CRAWLER_CONFIG = {
    'max_pages': 50,
    'max_depth': 2,
    'max_pages_per_city': 100,
    'max_places_per_city': 100,
}

# URLs organizadas por ciudades principales
CITY_URLS = {
    "Madrid": [
        "https://www.spain.info/en/destination/madrid/",
        "https://www.timeout.com/madrid/things-to-do",
        "https://www.lonelyplanet.com/spain/madrid/attractions"
    ],
    "Seville": [
        "https://www.spain.info/en/destination/seville/",
        "https://www.onthegotours.com/es/spain/guides/cities-to-visit-in-spain/seville/",
        "https://thetourguy.com/travel-blog/spain/seville/top-things-to-do-in-seville/"
    ],
    "Granada": [
        "https://www.spain.info/en/destination/granada/",
        "https://www.travelandleisure.com/travel-guide/granada",
        "https://www.europeanbestdestinations.com/destinations/granada/",
        "https://www.spain.info/en/destination/granada/",
        "https://www.travelandleisure.com/travel-guide/granada",
        "https://www.europeanbestdestinations.com/destinations/granada/",
        "https://www.andalucia.org/en/granada",
        "https://www.lonelyplanet.com/spain/andalucia/granada",
        "https://www.visitspain.com/en/cities/granada/"
    ],
    "Córdoba": [
        "https://www.spain.info/en/destination/cordoba/",
        "https://travel.usnews.com/Cordoba_Spain/Things_To_Do/",
        "https://www.kiwi.com/stories/five-of-the-best-cities-and-places-to-visit-in-spain/"
    ],
    "Toledo": [
        "https://www.spain.info/en/destination/toledo/",
        "https://www.planetware.com/tourist-attractions-/toledo-e-cas-to.htm",
        "https://www.cuddlynest.com/blog/tourist-attractions-in-spain/"
    ],
    "Santiago de Compostela": [
        "https://www.spain.info/en/destination/santiago-de-compostela/",
        "https://www.lonelyplanet.com/spain/galicia/santiago-de-compostela/attractions",
        "https://travel.usnews.com/Santiago_de_Compostela_Spain/Things_To_Do/"
    ],
    "San Sebastián": [
        "https://www.spain.info/en/destination/san-sebastian/",
        "https://www.europeanbestdestinations.com/destinations/san-sebastian/",
        "https://www.onthegotours.com/es/spain/guides/best-places-to-visit-in-spain/"
    ],
    "Málaga": [
        "https://www.spain.info/en/destination/malaga/",
        "https://www.hdfcergo.com/travel-insurance/travel-guide/tourist-places-to-visit-in-spain",
        "https://www.europeanbestdestinations.com/destinations/malaga/"
    ],
    "Cádiz": [
        "https://www.spain.info/en/destination/cadiz/",
        "https://travel.usnews.com/Cadiz_Spain/Things_To_Do/",
        "https://www.europeanbestdestinations.com/destinations/cadiz/"
    ],
    "Ronda": [
        "https://www.spain.info/en/destination/ronda/",
        "https://www.europeanbestdestinations.com/destinations/ronda/",
        "https://makespain.com/ronda/"
    ],
    "León": [
        "https://www.spain.info/en/destination/leon/",
        "https://www.lonelyplanet.com/spain/castilla-y-leon/leon/attractions",
        "https://www.cuddlynest.com/blog/tourist-attractions-in-spain/"
    ],
    "Cuenca": [
        "https://www.spain.info/en/destination/cuenca/",
        "https://www.cuddlynest.com/blog/tourist-attractions-in-spain/",
        "https://www.timeout.com/spain/things-to-do/best-cities-and-places-to-visit-in-spain"
    ],
    "Ávila": [
        "https://www.spain.info/en/destination/avila/",
        "https://thetourguy.com/travel-blog/spain/top-things-to-see-in-spain/",
        "https://www.tripadvisor.com/Attractions-g187491-Activities-Avila_Castile_and_Leon.html"
    ],
    "Palma de Mallorca": [
        "https://www.spain.info/en/destination/palma-de-mallorca/",
        "https://www.travelandleisure.com/travel-guide/palma-de-mallorca",
        "https://www.europeanbestdestinations.com/destinations/palma-de-mallorca/"
    ],
    "Tenerife": [
        "https://www.spain.info/en/destination/tenerife/",
        "https://www.holidify.com/places/tenerife/sightseeing-and-things-to-do.html",
        "https://travel.usnews.com/Tenerife_Spain/Things_To_Do/"
    ],
    "Murcia": [
        "https://www.spain.info/en/destination/murcia/",
        "https://www.murciaturistica.es/en/",
        "https://www.murcia.es/web/portal/turismo",
        "https://www.lonelyplanet.com/spain/murcia",
        "https://www.timeout.com/murcia/attractions",
        "https://www.europeanbestdestinations.com/destinations/murcia/"
    ],
    "Barcelona": [
        "https://www.spain.info/en/destination/barcelona/",
        "https://www.barcelonaturisme.com/wv3/en/",
        "https://www.timeout.com/barcelona/attractions",
        "https://www.lonelyplanet.com/spain/catalonia/barcelona",
        "https://www.europeanbestdestinations.com/destinations/barcelona/"
    ],
    "Valencia": [
        "https://www.spain.info/en/destination/valencia/",
        "https://www.visitvalencia.com/en/",
        "https://www.timeout.com/valencia/attractions",
        "https://www.lonelyplanet.com/spain/valencia/valencia",
        "https://www.comunitatvalenciana.com/en/ciudad/valencia"
    ],
    "Bilbao": [
        "https://www.spain.info/en/destination/bilbao/",
        "https://www.bilbaoturismo.net/BilbaoTurismo/en/",
        "https://www.timeout.com/bilbao/attractions",
        "https://www.lonelyplanet.com/spain/basque-country/bilbao",
        "https://www.europeanbestdestinations.com/destinations/bilbao/"
    ],
    "Salamanca": [
        "https://www.spain.info/en/destination/salamanca/",
        "https://www.salamanca.es/en/",
        "https://www.timeout.com/salamanca/attractions",
        "https://www.lonelyplanet.com/spain/castilla-y-leon/salamanca",
        "https://www.europeanbestdestinations.com/destinations/salamanca/"
    ],
    "Zaragoza": [
        "https://www.spain.info/en/destination/zaragoza/",
        "https://www.zaragoza.es/ciudad/turismo/en/",
        "https://www.timeout.com/zaragoza/attractions",
        "https://www.lonelyplanet.com/spain/aragon/zaragoza",
        "https://www.europeanbestdestinations.com/destinations/zaragoza/"
    ],
    "Santander": [
        "https://www.spain.info/en/destination/santander/",
        "https://www.turismodesantander.com/en/",
        "https://www.timeout.com/santander/attractions",
        "https://www.lonelyplanet.com/spain/cantabria/santander",
        "https://www.europeanbestdestinations.com/destinations/santander/"
    ]
}

# URLs de ciudades adicionales
CITY_URLS_EXTRA = {
    "Benidorm": [
        "https://www.visitbenidorm.es/ver/1265/benidorm-tourist-info-offices.html",
        "https://www.comunitatvalenciana.com/alacant-alicante/benidorm/oficines-de-turisme/tourist-info-benidorm-centro",
        "https://www.spain.info/en/destination/benidorm/",
        "https://www.timeout.com/benidorm/attractions",
        "https://www.lonelyplanet.com/spain/valencia/benidorm",
    ],
    "Santiago de Compostela": [
        "https://www.santiagoturismo.com/",
        "https://www.santiagoturismo.com/que-ver",
        "https://www.santiagoturismo.com/que-hacer",
        "https://www.spain.info/en/destination/santiago-de-compostela/",
        "https://www.timeout.com/santiago-de-compostela/attractions",
        "https://www.lonelyplanet.com/spain/galicia/santiago-de-compostela",
        "https://en.wikipedia.org/wiki/Santiago_de_Compostela",
    ],
    "Alicante": [
        "https://www.alicanteturismo.com/",
        "https://www.alicanteturismo.com/que-ver/",
        "https://www.alicanteturismo.com/que-hacer/",
        "https://www.spain.info/en/destination/alicante/",
        "https://www.comunitatvalenciana.com/en/ciudad/alicante",
        "https://www.timeout.com/alicante/attractions",
        "https://www.lonelyplanet.com/spain/valencia/alicante",
        "https://en.wikipedia.org/wiki/Alicante",
    ],
    "Marbella": [
        "https://www.marbella.es/ayuntamiento/turismo.html",
        "https://www.spain.info/en/destination/marbella/",
        "https://www.andalucia.org/en/marbella-tourism",
        "https://www.timeout.com/marbella/attractions",
        "https://www.lonelyplanet.com/spain/andalucia/marbella",
        "https://en.wikipedia.org/wiki/Marbella",
    ],
    "Salou": [
        "https://www.visitsalou.eu/",
        "https://www.visitsalou.eu/en/what-to-see",
        "https://www.visitsalou.eu/en/what-to-do",
        "https://www.spain.info/en/destination/salou/",
        "https://www.timeout.com/salou/attractions",
        "https://www.lonelyplanet.com/spain/catalonia/salou",
        "https://en.wikipedia.org/wiki/Salou",
    ],
    "Ibiza": [
        "https://www.ibiza.travel/",
        "https://www.ibiza.travel/en/what-to-see/",
        "https://www.ibiza.travel/en/what-to-do/",
        "https://www.spain.info/en/destination/ibiza/",
        "https://www.illesbalears.travel/en/balearic-islands/ibiza",
        "https://www.timeout.com/ibiza/attractions",
        "https://www.lonelyplanet.com/spain/balearic-islands/ibiza",
        "https://en.wikipedia.org/wiki/Ibiza",
    ],
    "Girona": [
        "https://www.girona.cat/turisme/eng/",
        "https://www.girona.cat/turisme/eng/descobreix_girona",
        "https://www.spain.info/en/destination/girona/",
        "https://www.timeout.com/girona/attractions",
        "https://www.lonelyplanet.com/spain/catalonia/girona",
        "https://en.wikipedia.org/wiki/Girona",
    ],
    "León": [
        "https://www.turismoleon.org/",
        "https://www.turismoleon.org/que-ver/",
        "https://www.turismoleon.org/que-hacer/",
        "https://www.spain.info/en/destination/leon/",
        "https://www.timeout.com/leon/attractions",
        "https://www.lonelyplanet.com/spain/castilla-y-leon/leon",
        "https://en.wikipedia.org/wiki/León,_Spain",
    ],
    "Pamplona": [
        "https://www.turismo.navarra.es/esp/organice-viaje/recurso/Localidades/3092/Pamplona.htm",
        "https://www.spain.info/en/destination/pamplona/",
        "https://www.timeout.com/pamplona/attractions",
        "https://www.lonelyplanet.com/spain/navarra/pamplona",
        "https://en.wikipedia.org/wiki/Pamplona",
    ],
    "Vigo": [
        "https://www.turismodevigo.org/",
        "https://www.turismodevigo.org/que-ver/",
        "https://www.turismodevigo.org/que-hacer/",
        "https://www.spain.info/en/destination/vigo/",
        "https://www.timeout.com/vigo/attractions",
        "https://www.lonelyplanet.com/spain/galicia/vigo",
        "https://en.wikipedia.org/wiki/Vigo",
    ],
    "A Coruña": [
        "https://www.coruna.gal/turismo/en",
        "https://www.coruna.gal/turismo/en/what-to-see",
        "https://www.coruna.gal/turismo/en/what-to-do",
        "https://www.spain.info/en/destination/a-coruna/",
        "https://www.timeout.com/a-coruna/attractions",
        "https://www.lonelyplanet.com/spain/galicia/a-coruna",
        "https://en.wikipedia.org/wiki/A_Coruña",
    ],
    "Oviedo": [
        "https://www.oviedo.es/en/tourism",
        "https://www.oviedo.es/en/tourism/what-to-see",
        "https://www.oviedo.es/en/tourism/what-to-do",
        "https://www.spain.info/en/destination/oviedo/",
        "https://www.timeout.com/oviedo/attractions",
        "https://www.lonelyplanet.com/spain/asturias/oviedo",
        "https://en.wikipedia.org/wiki/Oviedo",
    ],
}

# Dominios bloqueados completamente
BLOCKED_DOMAINS = {
    # Wikipedia y sitios relacionados - COMPLETAMENTE BLOQUEADOS
    'wikipedia.org', 'wikimedia.org', 'wikidata.org', 'wikisource.org',
    'wiktionary.org', 'wikinews.org', 'wikiquote.org', 'wikibooks.org',
    'wikiversity.org', 'wikivoyage.org', 'commons.wikimedia.org',
    # Redes sociales
    'facebook.com', 'twitter.com', 'instagram.com', 'youtube.com',
    'linkedin.com', 'pinterest.com', 'tiktok.com', 'snapchat.com',
    # Sitios de reseñas y booking
    'tripadvisor.com', 'booking.com', 'expedia.com', 'hotels.com',
    'airbnb.com', 'vrbo.com', 'kayak.com', 'skyscanner.com',
    # Otros sitios no relevantes
    'amazon.com', 'ebay.com', 'aliexpress.com', 'alibaba.com',
    'google.com', 'bing.com', 'yahoo.com', 'duckduckgo.com'
}

# Patrones de Madrid para bloquear
MADRID_PATTERNS = [
    'esmadrid.com', 'madrid.es', 'madrid.city-tour.com',
    '/madrid/', 'madrid-', '-madrid', 'madrid_',
    'spain.info/en/destination/madrid', 'lonelyplanet.com/spain/madrid',
    'timeout.com/madrid', 'visitmadrid', 'turismomadrid'
]

# Patrones excluidos
EXCLUDED_PATTERNS = [
    'robots.txt', 'sitemap', 'privacy', 'policy', 'legal', 'terms',
    'cookies', 'aviso-legal', 'politica', 'condiciones',
    'special:', 'talk:', 'category_talk:', 'user:', 'file:',
    'template:', 'help:', 'mailto:', 'javascript:',
    'tel:', 'fax:', 'skype:', 'whatsapp:', 'facebook.com', 'twitter.com',
    'instagram.com', 'youtube.com', 'linkedin.com', 'pinterest.com'
]

# Países excluidos (no España)
EXCLUDED_COUNTRIES = [
    # África
    'morocco', 'egypt', 'tunisia', 'algeria', 'libya', 'sudan', 'ethiopia',
    'kenya', 'tanzania', 'uganda', 'ghana', 'nigeria', 'senegal', 'mali',
    'burkina', 'niger', 'chad', 'cameroon', 'gabon', 'congo', 'angola',
    'zambia', 'zimbabwe', 'botswana', 'namibia', 'south-africa', 'madagascar',
    # Asia
    'china', 'japan', 'korea', 'thailand', 'vietnam', 'cambodia', 'laos',
    'myanmar', 'malaysia', 'singapore', 'indonesia', 'philippines', 'india',
    'pakistan', 'bangladesh', 'sri-lanka', 'nepal', 'bhutan', 'afghanistan',
    'iran', 'iraq', 'syria', 'lebanon', 'jordan', 'israel', 'palestine',
    'saudi-arabia', 'yemen', 'oman', 'uae', 'qatar', 'bahrain', 'kuwait',
    'turkey', 'georgia', 'armenia', 'azerbaijan', 'kazakhstan', 'uzbekistan',
    'kyrgyzstan', 'tajikistan', 'turkmenistan', 'mongolia',
    # América
    'usa', 'canada', 'mexico', 'guatemala', 'belize', 'honduras', 'salvador',
    'nicaragua', 'costa-rica', 'panama', 'colombia', 'venezuela', 'guyana',
    'suriname', 'brazil', 'ecuador', 'peru', 'bolivia', 'chile', 'argentina',
    'uruguay', 'paraguay', 'cuba', 'jamaica', 'haiti', 'dominican-republic',
    # Europa (no España)
    'france', 'germany', 'italy', 'portugal', 'united-kingdom', 'ireland',
    'netherlands', 'belgium', 'luxembourg', 'switzerland', 'austria',
    'poland', 'czech-republic', 'slovakia', 'hungary', 'romania', 'bulgaria',
    'greece', 'albania', 'macedonia', 'montenegro', 'serbia', 'bosnia',
    'croatia', 'slovenia', 'denmark', 'sweden', 'norway', 'finland',
    'estonia', 'latvia', 'lithuania', 'belarus', 'ukraine', 'moldova',
    'russia', 'iceland', 'malta', 'cyprus',
    # Oceanía
    'australia', 'new-zealand', 'fiji', 'papua-new-guinea', 'samoa', 'tonga'
]

# Dominios españoles permitidos
SPANISH_DOMAINS = [
    # Dominios oficiales españoles
    'spain.info', 'turespaña.es', 'tourspain.es',
    # Barcelona
    'barcelonaturisme.com', 'barcelona.cat', 'visitbarcelona.com',
    'bcnshop.barcelonaturisme.com',
    # Valencia
    'visitvalencia.com', 'valencia.es', 'turismovalencia.es',
    'comunitatvalenciana.com',
    # Andalucía
    'visitasevilla.es', 'sevilla.org', 'andalucia.org',
    'granadatur.com', 'granada.org', 'turismodecordoba.org',
    'cordoba.es', 'malagaturismo.com', 'malaga.eu', 'costadelsol.travel',
    'turismo.cadiz.es', 'cadizturismo.com', 'marbella.es',
    # País Vasco
    'bilbaoturismo.net', 'bilbao.eus', 'turismo.euskadi.eus',
    'sansebastianturismo.com', 'donostia.eus',
    # Castilla y León
    'toledo-turismo.com', 'ayto-toledo.org', 'turismo.castillalamancha.es',
    'salamanca.es', 'turismodesalamanca.com', 'turismosalamanaca.com',
    'turismoleon.org',
    # Aragón
    'zaragoza.es', 'turismodezaragoza.es', 'turismo.aragon.es',
    # Cantabria
    'turismodesantander.com', 'santander.es', 'turismodecantabria.com',
    # Murcia
    'murciaturistica.es', 'murcia.es',
    # Baleares
    'visitpalma.com', 'palma.cat', 'illesbalears.travel', 'ibiza.travel',
    # Canarias
    'hellocanaryislands.com', 'grancanaria.com', 'webtenerife.com',
    # Galicia
    'santiagoturismo.com', 'turismodevigo.org', 'coruna.gal',
    'turismo.gal', 'turgalicia.es',
    # Asturias
    'oviedo.es', 'turismoasturias.es',
    # Navarra
    'turismo.navarra.es',
    # Cataluña
    'girona.cat', 'visitsalou.eu', 'catalunya.com',
    # Comunidad Valenciana
    'visitbenidorm.es', 'alicanteturismo.com',
    # Generales internacionales (solo para España)
    'timeout.com', 'lonelyplanet.com', 'en.wikipedia.org'
]

# Términos turísticos relevantes
TOURIST_TERMS = [
    'tourism', 'attraction', 'place', 'visit', 'museum', 'monument', 
    'palace', 'park', 'what-to-see', 'what-to-do', 'destinations',
    'turismo', 'atracciones', 'lugares', 'visitar', 'museo', 'monumentos'
]