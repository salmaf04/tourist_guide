import agent_generator.mistral_client as CL
import google.generativeai as genai
from dotenv import load_dotenv
import os
import sys
import random
import json
from BestRoutes.meta_routes import RouteOptimizer
import re
import app as app
import RAG.rag as rag

"""
AVAILEABLE_CITIES = [
    'Madrid',
    'Barcelona',
    'Valencia',
    'Sevilla',
    'Bilbao',
    'Granada',
    'Toledo',
    'Salamanca',
    'Málaga',
    'San Sebastián',
    'Córdoba',
    'Zaragoza',
    'Santander',
    'Cádiz',
    'Murcia',
    'Palma de Mallorca'
]

CATEGORIES = [
    ("engineering", "Ingeniería"),
    ("history", "Historia"),
    ("food", "Comida"),
    ("culture", "Cultura"),
    ("beach", "Playas"),
    ("shopping", "Compras"),
    ("nature", "Naturaleza"),
]

SCORE_LABELS = {
    1: "No me gusta nada",
    2: "No me atrae",
    3: "Me es indiferente",
    4: "Me gusta",
    5: "Me encanta"
}

TRANSPORT_MODES = [
    "Caminar",
    "Bicicleta", 
    "Bus",
    "Coche/taxi",
    "Otro"
]
"""

class TestCreator:
    def __init__(self):
        os.environ['GEMINI_API_KEY'] = 'AIzaSyAlTAfeGz0amVAat8fyt3ZEtLBIQ9OFO5o'
        self.mistral_model_FC=CL.MistralClient(0.7)
        self.mistral_model_FE=CL.MistralClient(0.1)


    def get_user_preferences_from_llm(self):
        """
        Obtiene preferencias de usuario simuladas de Gemini y las parsea en un diccionario

        Args:
            gemini_model: Modelo Gemini inicializado (ej: gemini-pro)

        Returns:
            dict: Diccionario con las preferencias del usuario
        """
        # Definición de constantes
        AVAILABLE_CITIES = [
            'Madrid', 'Barcelona', 'Valencia', 'Sevilla', 'Bilbao', 'Granada', 'Toledo',
            'Salamanca', 'Málaga', 'San Sebastián', 'Córdoba', 'Zaragoza', 'Santander',
            'Cádiz', 'Murcia', 'Palma de Mallorca'
        ]

        TRANSPORT_MODES = ["Caminar", "Bicicleta", "Bus", "Coche/taxi", "Otro"]

        # Construir el prompt
        prompt = f"""
        Actúa como un turista real planeando un viaje. Responde ÚNICAMENTE con un objeto JSON válido.
        Instrucciones:
        1. Selecciona una ciudad de esta lista: {', '.join(AVAILABLE_CITIES)}
        2. Horas disponibles (1-16): Un número entero aleatorio entre 4 y 12
        3. Distancia máxima (1-50 km): Un número entero aleatorio entre 5 y 30
        4. Transporte: Selecciona 1-3 modos de {TRANSPORT_MODES} (incluye siempre 'Caminar')
        5. Para cada categoría, asigna un número del 1 al 5 según:
            - 1: No me gusta nada
            - 2: No me atrae
            - 3: Me es indiferente
            - 4: Me gusta
            - 5: Me encanta
        6. Notas: 1-2 frases realistas como turista

        Estructura JSON requerida:
        {{
            "city": "Ciudad seleccionada",
            "available_hours": número,
            "max_distance": número,
            "transport_modes": ["lista", "de", "transportes"],
            "category_interest": {{
                "engineering": número,
                "history": número,
                "food": número,
                "culture": número,
                "beach": número,
                "shopping": número,
                "nature": número
            }},
            "user_notes": "Tus notas aquí"
        }}

        Ejemplo de respuesta válida:
        {{
            "city": "Barcelona",
            "available_hours": 8,
            "max_distance": 15,
            "transport_modes": ["Caminar", "Bus"],
            "category_interest": {{
                "engineering": 4,
                "history": 5,
                "food": 5,
                "culture": 4,
                "beach": 3,
                "shopping": 2,
                "nature": 3
            }},
            "user_notes": "Me interesa la arquitectura modernista y probar tapas auténticas"
        }}
        """

        try:
            # Enviar prompt a Gemini (ajustar según SDK)
            response_text = self.mistral_model_FC.generate(prompt)

            # Extraer JSON de la respuesta (manejo de posibles artefactos)
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if not json_match:
                raise ValueError("No se encontró JSON en la respuesta")

            data = json.loads(json_match.group())

            # Validar y normalizar datos
            if data['city'] not in AVAILABLE_CITIES:
                data['city'] = 'Madrid'  # Valor por defecto

            data['available_hours'] = max(1, min(16, int(data['available_hours'])))
            data['max_distance'] = max(1, min(50, int(data['max_distance'])))

            # Normalizar transportes
            valid_transport = [t for t in data['transport_modes'] if t in TRANSPORT_MODES]
            if not valid_transport:
                valid_transport = ['Caminar']
            data['transport_modes'] = valid_transport

            # Normalizar categorías
            for category in ['engineering', 'history', 'food', 'culture', 'beach', 'shopping', 'nature']:
                data['category_interest'][category] = max(1, min(5, int(data['category_interest'].get(category, 3))))

            # Construir diccionario final
            user_preferences = {
                'city': data['city'],
                'available_hours': data['available_hours'],
                'category_interest': data['category_interest'],
                'transport_modes': data['transport_modes'],
                'max_distance': data['max_distance'],
                'user_notes': data['user_notes'].strip()
            }

            return user_preferences

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # Manejo de errores con valores por defecto
            print(f"Error parsing response: {str(e)}. Using default preferences")
            return {
                'city': 'Error',
                'available_hours': 8,
                'category_interest': {c: 3 for c in ['engineering', 'history', 'food', 'culture', 'beach', 'shopping', 'nature']},
                'transport_modes': ['Caminar'],
                'max_distance': 10,
                'user_notes': ""
            }
            
    def generate_preferences_from_route(self, route, user_preferences:dict, meta_data):
        """
        Genera preferencias de usuario (category_interest y user_notes) basadas en una ruta turística

        Args:
            gemini_model: Modelo Gemini inicializado
            route: Lista de diccionarios con los POIs de la ruta en orden

        Returns:
            dict: Diccionario con 'category_interest' y 'user_notes'
        """
        # Construir descripción de la ruta
        route_description = "\n\n".join(
            f"POI {i+1}: {meta_data['node_params'][poi]['place_data']['name']}\n"
            f"Descripción: {meta_data['node_params'][poi]['place_data']['description']}"
            for i, poi in enumerate(route[1:-1]))

        # Construir el prompt
        prompt = f"""
        Eres un turista que acaba de completar esta ruta turística:
        {route_description}

        Tarea:
        1. Analiza las categorías predominantes en la ruta (usa estas categorías: engineering, history, food, culture, beach, shopping, nature)
        2. Genera un JSON con:
            a) 'category_interest': Puntuaciones 1-5 para CADA categoría (5=me encanta) basadas en la ruta
            b) 'user_notes': Descripción breve (2-3 oraciones) de los intereses o gustos reflejados en la ruta

        Reglas:
        - Las categorías con más apariciones deben tener puntuaciones más altas (4-5)
        - Las categorías ausentes deben tener puntuaciones bajas (1-2)
        - Mantén las notas realistas y coherentes con la ruta
        - ¡SOLO RESPONDE CON EL JSON REQUERIDO!

        Estructura JSON requerida:
        {{
            "category_interest": {{
                "engineering": <puntaje>,
                "history": <puntaje>,
                "food": <puntaje>,
                "culture": <puntaje>,
                "beach": <puntaje>,
                "shopping": <puntaje>,
                "nature": <puntaje>
            }},
            "user_notes": "<texto descriptivo>"
        }}

        Ejemplo de respuesta válida:
        {{
            "category_interest": {{
                "history": 5,
                "culture": 4,
                "engineering": 3,
                "food": 4,
                "beach": 1,
                "shopping": 2,
                "nature": 3
            }},
            "user_notes": "Me fascina la historia y la cultura local, con especial interés en la gastronomía tradicional. Disfruto explorando sitios patrimoniales y probando platos auténticos."
        }}
        """
        print(prompt)

        try:
            # Enviar a Gemini
            response_text = self.mistral_model_FE.generate(prompt)

            # Extraer JSON
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if not json_match:
                raise ValueError("Respuesta no contiene JSON válido")

            data = json.loads(json_match.group())

            # Validar y normalizar
            valid_categories = ['engineering', 'history', 'food', 'culture', 'beach', 'shopping', 'nature']
            for category in valid_categories:
                score = data['category_interest'].get(category, 3)
                data['category_interest'][category] = max(1, min(5, int(score)))

            # Limpieza de notas
            user_notes = data['user_notes'].strip().replace('"', '')

            new_user_preferences = user_preferences.copy()
            new_user_preferences['user_notes'] = user_notes
            new_user_preferences['category_interest'] = data['category_interest']

            return new_user_preferences

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Error generando preferencias: {str(e)}. Usando valores por defecto")
            return {
                'category_interest': {c: 3 for c in ['engineering', 'history', 'food', 'culture', 'beach', 'shopping', 'nature']},
                'user_notes': "No tengo preferencias especiales."
            }
    
    def evaluate(self, routes, user_preferences, meta_data):
        ra=rag.RAGPlanner()
        emb1=ra._generate_user_embedding(user_preferences)
        score=-1
        for route in routes:
            emb2=ra._generate_user_embedding(self.generate_preferences_from_route(route,user_preferences,meta_data))
            score=max(ra._calculate_cosine_similarity(emb1,[emb2])[0],score)
        
        return score
    
    def generate_test(self):
        user_preferences = self.get_user_preferences_from_llm()
        city_coords = {
                'Madrid': (40.4168, -3.7038),
                'Barcelona': (41.3851, 2.1734),
                'Valencia': (39.4699, -0.3763),
                'Sevilla': (37.3891, -5.9845),
                'Bilbao': (43.2627, -2.9253),
                'Granada': (37.1773, -3.5986),
                'Toledo': (39.8628, -4.0273),
                'Salamanca': (40.9701, -5.6635),
                'Málaga': (36.7213, -4.4214),
                'San Sebastián': (43.3183, -1.9812),
                'Córdoba': (37.8882, -4.7794),
                'Zaragoza': (41.6488, -0.8891),
                'Santander': (43.4623, -3.8099),
                'Cádiz': (36.5297, -6.2920),
                'Murcia': (37.9922, -1.1307),
                'Palma de Mallorca': (39.5696, 2.6502)
            }
        lat, lon = city_coords.get(user_preferences['city'], (40.4168, -3.7038))
        transport_mode = user_preferences['transport_modes'][0] if user_preferences['transport_modes'] else "Caminar"
        
        crawler_success = app.fetch_tourism_data(user_preferences['city'])

        rag_planner = app.RAGPlanner()  # Usa la ruta por defecto que ya está configurada correctamente

        rag_data = rag_planner.process_user_request(user_preferences, lat, lon, transport_mode)

        meta_data = app.prepare_metaheuristic_data(rag_data, user_preferences)
        
        route_optimizer = RouteOptimizer(
                adjacency_matrix=meta_data['adjacency_matrix'],
                node_params=meta_data['node_params'],
                max_time=meta_data['max_time_minutes'],
                tourist_param=meta_data['tourist_param']
            )
            
        # Obtener rutas optimizadas de la metaheurística
        all_routes = route_optimizer.get_routes()
            
        # Evaluar todas las rutas con la función objetivo tradicional
        route_evaluations = []
        for route in all_routes:
            goal_value = route_optimizer.RouteFinder.goal_func(route, meta_data['max_time_minutes'], meta_data['tourist_param'])
            route_evaluations.append({
                'route': route,
                'goal_value': goal_value
            })
        # Ordenar por función objetivo (mayor es mejor) y tomar las 10 mejores
        route_evaluations.sort(key=lambda x: x['goal_value'], reverse=True)
        top_10_routes = [eval_data['route'] for eval_data in route_evaluations[:2]]
        
        # Simular solo las 10 mejores rutas para obtener las mejores por satisfacción
        from agent_generator.route_simulator import simulate_and_rank_routes
        optimized_routes_with_details = simulate_and_rank_routes(
            routes=top_10_routes,
            node_params=meta_data['node_params'],
            top_n=2,
            simulation_steps=4,
            tourist_name="Turista_Simulado"
        )
        optimized_routes = [result['route'] for result in optimized_routes_with_details]

        print(f"Prefenecias del turista: {user_preferences}\n")
        print(f"Metaheuristica score: {self.evaluate(top_10_routes[:3], user_preferences,meta_data)}\n")
        print(f"Rutas de la Metaheuristica: {top_10_routes[:1]}\n\n")
        print(f"Metaheuristica + Simulación score: {self.evaluate(optimized_routes , user_preferences,meta_data)}")
        print(f"Rutas de la Metaheuristica + Simulación: {optimized_routes}\n\n")