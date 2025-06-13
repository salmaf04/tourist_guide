"""
Módulo para simular rutas turísticas usando el sistema de agentes
y evaluar su calidad basada en la satisfacción del turista.
"""

import random
from typing import List, Dict, Tuple
import sys
import os

# Add the current directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

try:
    from agent_generator.generator import ModeloTurismo
    from agent_generator.models import Nodo
except ImportError as e:
    print(f"Error importing agent_generator modules: {e}")
    # Try alternative import
    try:
        import agent_generator.generator as gen
        import agent_generator.models as models
        ModeloTurismo = gen.ModeloTurismo
        Nodo = models.Nodo
    except ImportError as e2:
        print(f"Alternative import also failed: {e2}")
        raise ImportError("Could not import required agent_generator modules")


class RouteSimulator:
    """
    Clase para simular rutas turísticas y evaluar su calidad
    basada en la satisfacción del turista.
    """
    
    def __init__(self, places_data: List[Dict], tourist_name: str = "Turista"):
        """
        Inicializa el simulador de rutas.
        
        :param places_data: Lista de datos de lugares/nodos
        :param tourist_name: Nombre del turista para la simulación
        """
        self.places_data = places_data
        self.tourist_name = tourist_name
        
    def convert_route_to_nodes(self, route: List[int], node_params: List[Dict]) -> List[Dict]:
        """
        Convierte una ruta (lista de índices) a una lista de nodos para la simulación.
        
        :param route: Lista de índices de nodos que representan la ruta
        :param node_params: Parámetros de los nodos del optimizador
        :return: Lista de diccionarios con información de nodos para la simulación
        """
        nodos_simulacion = []
        
        for i, node_idx in enumerate(route[1:], 1):  # Saltar el nodo de inicio (índice 0)
            if node_idx < len(node_params) and node_idx > 0:  # Asegurar que no es el nodo de inicio
                node_param = node_params[node_idx]
                
                # Crear nodo para la simulación
                nodo = {
                    "id": f"nodo_{node_idx}",
                    "nombre": node_param.get('place_name', f"Lugar {node_idx}"),
                    "tipo": self._get_place_type(node_param),
                    "descripcion": self._get_place_description(node_param),
                    "agentes": self._get_place_agents(node_param)
                }
                nodos_simulacion.append(nodo)
        
        return nodos_simulacion
    
    def _get_place_type(self, node_param: Dict) -> str:
        """
        Determina el tipo de lugar basado en los parámetros del nodo.
        """
        place_data = node_param.get('place_data', {})
        classification = place_data.get('touristClassification', '').lower()
        
        # Mapear clasificaciones a tipos de lugares
        type_mapping = {
            'museum': 'museo',
            'restaurant': 'restaurante',
            'park': 'parque',
            'monument': 'monumento',
            'church': 'iglesia',
            'market': 'mercado',
            'shopping': 'tienda',
            'beach': 'playa',
            'viewpoint': 'mirador'
        }
        
        for key, value in type_mapping.items():
            if key in classification:
                return value
        
        return 'atraccion'  # Tipo por defecto
    
    def _get_place_description(self, node_param: Dict) -> str:
        """
        Obtiene la descripción del lugar.
        """
        place_data = node_param.get('place_data', {})
        description = place_data.get('description', '')
        
        if not description:
            place_name = node_param.get('place_name', 'Este lugar')
            description = f"{place_name} es un destino turístico interesante que vale la pena visitar."
        
        return description
    
    def _get_place_agents(self, node_param: Dict) -> List[str]:
        """
        Determina qué tipos de agentes debe tener un lugar.
        """
        place_type = self._get_place_type(node_param)
        
        # Mapear tipos de lugares a agentes apropiados
        agent_mapping = {
            'museo': ['guía', 'curador'],
            'restaurante': ['mesero', 'chef'],
            'parque': ['guía', 'jardinero'],
            'monumento': ['guía', 'historiador'],
            'iglesia': ['guía', 'sacerdote'],
            'mercado': ['vendedor', 'guía'],
            'tienda': ['vendedor', 'asistente'],
            'playa': ['salvavidas', 'guía'],
            'mirador': ['guía', 'fotógrafo'],
            'atraccion': ['guía', 'asistente']
        }
        
        return agent_mapping.get(place_type, ['guía', 'asistente'])
    
    def simulate_route(self, route: List[int], node_params: List[Dict], 
                      simulation_steps: int = 5) -> Dict:
        """
        Simula una ruta específica y devuelve métricas de satisfacción.
        
        :param route: Lista de índices de nodos que representan la ruta
        :param node_params: Parámetros de los nodos del optimizador
        :param simulation_steps: Número de pasos de simulación por lugar
        :return: Diccionario con métricas de la simulación
        """
        if len(route) <= 2:  # Solo inicio y fin, sin lugares intermedios
            return {
                'satisfaction_score': 0.0,
                'num_places_visited': 0,
                'significant_memories': [],
                'total_interactions': 0,
                'route': route
            }
        
        # Convertir ruta a nodos para simulación
        nodos_simulacion = self.convert_route_to_nodes(route, node_params)
        
        if not nodos_simulacion:
            return {
                'satisfaction_score': 0.0,
                'num_places_visited': 0,
                'significant_memories': [],
                'total_interactions': 0,
                'route': route
            }
        
        try:
            # Crear modelo de simulación
            modelo = ModeloTurismo(nodos_simulacion, nombre_turista=self.tourist_name)
            
            # Ejecutar simulación
            for step in range(simulation_steps):
                modelo.step()
            
            # Obtener métricas finales
            satisfaction_score = modelo.turista.satisfaccion
            significant_memories = modelo.turista.recuerdos_significativos()
            
            # Contar interacciones totales
            total_interactions = 0
            for agente in modelo.schedule.agents:
                if hasattr(agente, 'interacciones'):
                    total_interactions += len(agente.interacciones)
            
            return {
                'satisfaction_score': satisfaction_score,
                'num_places_visited': len(nodos_simulacion),
                'significant_memories': significant_memories,
                'total_interactions': total_interactions,
                'route': route,
                'simulation_success': True
            }
            
        except Exception as e:
            print(f"Error en simulación de ruta {route}: {str(e)}")
            return {
                'satisfaction_score': 0.0,
                'num_places_visited': len(nodos_simulacion),
                'significant_memories': [],
                'total_interactions': 0,
                'route': route,
                'simulation_success': False,
                'error': str(e)
            }
    
    def simulate_multiple_routes(self, routes: List[List[int]], node_params: List[Dict],
                               simulation_steps: int = 5) -> List[Dict]:
        """
        Simula múltiples rutas y devuelve sus métricas ordenadas por satisfacción.
        
        :param routes: Lista de rutas a simular
        :param node_params: Parámetros de los nodos del optimizador
        :param simulation_steps: Número de pasos de simulación por lugar
        :return: Lista de métricas ordenadas por satisfacción (mayor a menor)
        """
        simulation_results = []
        
        print(f"Iniciando simulación de {len(routes)} rutas...")
        
        for i, route in enumerate(routes):
            print(f"Simulando ruta {i+1}/{len(routes)}: {route}")
            
            # Simular la ruta
            result = self.simulate_route(route, node_params, simulation_steps)
            result['route_index'] = i
            simulation_results.append(result)
            
            print(f"Ruta {i+1} completada - Satisfacción: {result['satisfaction_score']:.2f}")
        
        # Ordenar por satisfacción (mayor a menor)
        simulation_results.sort(key=lambda x: x['satisfaction_score'], reverse=True)
        
        print("Simulación completada. Rutas ordenadas por satisfacción:")
        for i, result in enumerate(simulation_results):
            print(f"  {i+1}. Ruta {result['route_index']+1}: {result['satisfaction_score']:.2f}")
        
        return simulation_results
    
    def get_best_simulated_routes(self, routes: List[List[int]], node_params: List[Dict],
                                 top_n: int = 3, simulation_steps: int = 5) -> List[Dict]:
        """
        Obtiene las mejores N rutas basadas en la simulación de satisfacción.
        
        :param routes: Lista de rutas generadas por la metaheurística
        :param node_params: Parámetros de los nodos del optimizador
        :param top_n: Número de mejores rutas a devolver
        :param simulation_steps: Número de pasos de simulación por lugar
        :return: Lista de las mejores rutas con sus métricas de simulación
        """
        # Simular todas las rutas
        simulation_results = self.simulate_multiple_routes(routes, node_params, simulation_steps)
        
        # Devolver las mejores N rutas
        best_routes = simulation_results[:top_n]
        
        print(f"\nMejores {len(best_routes)} rutas seleccionadas:")
        for i, result in enumerate(best_routes):
            print(f"  {i+1}. Satisfacción: {result['satisfaction_score']:.2f}, "
                  f"Lugares: {result['num_places_visited']}, "
                  f"Interacciones: {result['total_interactions']}")
        
        return best_routes


def simulate_and_rank_routes(routes: List[List[int]], node_params: List[Dict],
                           places_data: List[Dict] = None, top_n: int = 3,
                           simulation_steps: int = 5, tourist_name: str = "Turista") -> List[Dict]:
    """
    Función de conveniencia para simular y clasificar rutas.
    
    :param routes: Lista de rutas generadas por la metaheurística
    :param node_params: Parámetros de los nodos del optimizador
    :param places_data: Datos de lugares (opcional)
    :param top_n: Número de mejores rutas a devolver
    :param simulation_steps: Número de pasos de simulación por lugar
    :param tourist_name: Nombre del turista para la simulación
    :return: Lista de las mejores rutas con sus métricas de simulación
    """
    if places_data is None:
        places_data = []
    
    simulator = RouteSimulator(places_data, tourist_name)
    return simulator.get_best_simulated_routes(routes, node_params, top_n, simulation_steps)