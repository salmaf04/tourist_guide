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
        
    def convert_route_to_nodes(self, route: List[int], node_params: List[Dict]) -> List:
        """
        Convierte una ruta (lista de índices) a una lista de nodos para la simulación.
        
        :param route: Lista de índices de nodos que representan la ruta
        :param node_params: Parámetros de los nodos del optimizador
        :return: Lista de objetos Nodo para la simulación
        """
        nodos_simulacion = []
        
        for i, node_idx in enumerate(route[1:], 1):  # Saltar el nodo de inicio (índice 0)
            if node_idx < len(node_params) and node_idx > 0:  # Asegurar que no es el nodo de inicio
                node_param = node_params[node_idx]
                
                # Crear nodo para la simulación usando la clase Nodo
                nodo = Nodo(
                    id=f"nodo_{node_idx}",
                    nombre=node_param.get('place_name', f"Lugar {node_idx}"),
                    tipo=self._get_place_type(node_param),
                    descripcion=self._get_place_description(node_param),
                    agentes=self._get_place_agents(node_param)
                )
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
        Simula una ruta específica y devuelve métricas de satisfacción mejoradas.
        
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
                'route': route,
                'route_quality_factors': {}
            }
        
        # Convertir ruta a nodos para simulación
        nodos_simulacion = self.convert_route_to_nodes(route, node_params)
        
        if not nodos_simulacion:
            return {
                'satisfaction_score': 0.0,
                'num_places_visited': 0,
                'significant_memories': [],
                'total_interactions': 0,
                'route': route,
                'route_quality_factors': {}
            }
        
        try:
            # Analizar calidad de la ruta antes de la simulación
            route_quality_factors = self._analyze_route_quality(route, node_params, nodos_simulacion)
            
            # Crear modelo de simulación
            modelo = ModeloTurismo(nodos_simulacion, nombre_turista=self.tourist_name)
            
            # Aplicar factores de ruta al turista inicial
            self._apply_route_factors_to_tourist(modelo.turista, route_quality_factors)
            
            # Ejecutar simulación mejorada
            for step in range(simulation_steps):
                # Simular fatiga progresiva
                self._simulate_progressive_fatigue(modelo.turista, step, simulation_steps)
                
                # Ejecutar paso del modelo
                modelo.step()
                
                # Aplicar efectos específicos de la secuencia de lugares
                self._apply_sequence_effects(modelo.turista, nodos_simulacion, step)
            
            # Obtener métricas finales
            satisfaction_score = modelo.turista.satisfaccion
            significant_memories = modelo.turista.recuerdos_significativos()
            
            # Contar interacciones totales
            total_interactions = 0
            for agente in modelo.schedule.agents:
                if hasattr(agente, 'interacciones'):
                    total_interactions += len(agente.interacciones)
            
            # Aplicar bonificaciones/penalizaciones finales basadas en la calidad de la ruta
            final_satisfaction = self._apply_final_route_adjustments(
                satisfaction_score, route_quality_factors, len(nodos_simulacion)
            )
            
            return {
                'satisfaction_score': final_satisfaction,
                'num_places_visited': len(nodos_simulacion),
                'significant_memories': significant_memories,
                'total_interactions': total_interactions,
                'route': route,
                'simulation_success': True,
                'route_quality_factors': route_quality_factors
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
                'error': str(e),
                'route_quality_factors': {}
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

    def _analyze_route_quality(self, route: List[int], node_params: List[Dict], nodos_simulacion: List) -> Dict:
        """
        Analiza la calidad de una ruta considerando múltiples factores.
        """
        factors = {
            'diversity_score': 0.0,
            'flow_score': 0.0,
            'balance_score': 0.0,
            'efficiency_score': 0.0,
            'experience_richness': 0.0
        }
        
        if len(nodos_simulacion) <= 1:
            return factors
        
        # 1. Diversidad de tipos de lugares
        place_types = [nodo.tipo for nodo in nodos_simulacion]
        unique_types = len(set(place_types))
        factors['diversity_score'] = min(unique_types / 4.0, 1.0)  # Normalizado a 1.0
        
        # 2. Flujo de la experiencia (secuencia lógica)
        flow_score = 0.0
        type_transitions = {
            ('museo', 'restaurante'): 0.8,
            ('parque', 'mirador'): 0.9,
            ('monumento', 'museo'): 0.7,
            ('restaurante', 'parque'): 0.8,
            ('tienda', 'restaurante'): 0.6,
            ('iglesia', 'monumento'): 0.8,
            ('mercado', 'restaurante'): 0.9,
            ('playa', 'restaurante'): 0.7,
            ('mirador', 'restaurante'): 0.8
        }
        
        for i in range(len(place_types) - 1):
            transition = (place_types[i], place_types[i + 1])
            flow_score += type_transitions.get(transition, 0.5)
        
        factors['flow_score'] = flow_score / max(len(place_types) - 1, 1)
        
        # 3. Balance de actividades (no demasiado intenso o relajado)
        intensity_map = {
            'museo': 0.7, 'monumento': 0.6, 'parque': 0.3, 'restaurante': 0.4,
            'iglesia': 0.5, 'mercado': 0.8, 'tienda': 0.6, 'playa': 0.2,
            'mirador': 0.4, 'atraccion': 0.6
        }
        
        intensities = [intensity_map.get(tipo, 0.5) for tipo in place_types]
        avg_intensity = sum(intensities) / len(intensities)
        # Penalizar extremos (muy intenso o muy relajado)
        factors['balance_score'] = 1.0 - abs(avg_intensity - 0.5) * 2
        
        # 4. Eficiencia (relación lugares visitados vs tiempo total estimado)
        total_time = sum(node_params[route[i]].get('time', 120) for i in range(1, len(route)) if route[i] < len(node_params))
        if total_time > 0:
            factors['efficiency_score'] = min(len(nodos_simulacion) * 60 / total_time, 1.0)
        
        # 5. Riqueza de experiencia (basada en descripciones y tipos)
        experience_points = 0
        for nodo in nodos_simulacion:
            if len(nodo.descripcion) > 50:
                experience_points += 0.3
            if nodo.tipo in ['museo', 'monumento', 'mirador']:
                experience_points += 0.4
            if nodo.tipo in ['mercado', 'playa']:
                experience_points += 0.3
        
        factors['experience_richness'] = min(experience_points / len(nodos_simulacion), 1.0)
        
        return factors

    def _apply_route_factors_to_tourist(self, turista, route_factors: Dict):
        """
        Aplica factores de calidad de ruta al estado inicial del turista.
        """
        # Ajustar satisfacción inicial basada en la calidad percibida de la ruta
        quality_bonus = (
            route_factors['diversity_score'] * 0.5 +
            route_factors['flow_score'] * 0.3 +
            route_factors['experience_richness'] * 0.4
        )
        
        # Aplicar bonificación/penalización inicial
        initial_adjustment = (quality_bonus - 0.6) * 2  # Centrado en 0.6, rango ±0.8
        turista.satisfaccion = max(0, min(10, turista.satisfaccion + initial_adjustment))
        
        print(f"DEBUG - Ajuste inicial por calidad de ruta: {initial_adjustment:.2f}, "
              f"satisfacción inicial: {turista.satisfaccion:.2f}")

    def _simulate_progressive_fatigue(self, turista, step: int, total_steps: int):
        """
        Simula fatiga progresiva durante la ruta.
        """
        # La fatiga aumenta progresivamente, pero puede ser mitigada por experiencias positivas
        fatigue_factor = step / total_steps
        base_fatigue = fatigue_factor * 0.3  # Máximo 0.3 puntos de fatiga
        
        # Reducir fatiga si la satisfacción es alta
        if turista.satisfaccion > 7:
            base_fatigue *= 0.5
        elif turista.satisfaccion < 4:
            base_fatigue *= 1.5
        
        # Aplicar fatiga como pequeña reducción de satisfacción
        turista.satisfaccion = max(0, turista.satisfaccion - base_fatigue)

    def _apply_sequence_effects(self, turista, nodos_simulacion: List, step: int):
        """
        Aplica efectos específicos basados en la secuencia de lugares visitados.
        """
        if step >= len(nodos_simulacion):
            return
        
        current_place = nodos_simulacion[step % len(nodos_simulacion)]
        
        # Efectos de secuencia
        sequence_effects = {
            'restaurante': {
                'after_museum': 0.8,  # Comer después de museo es bueno
                'after_park': 0.6,    # Después de parque es normal
                'consecutive': -0.4   # Dos restaurantes seguidos es malo
            },
            'museo': {
                'after_restaurant': 0.5,  # Después de comer, menos energía para museo
                'consecutive': -0.6       # Dos museos seguidos es cansado
            },
            'parque': {
                'after_museum': 0.7,     # Relajarse después de museo
                'after_restaurant': 0.4  # Después de comer, caminar es bueno
            }
        }
        
        if step > 0:
            prev_place = nodos_simulacion[(step - 1) % len(nodos_simulacion)]
            current_type = current_place.tipo
            prev_type = prev_place.tipo
            
            effects = sequence_effects.get(current_type, {})
            
            if prev_type == current_type and 'consecutive' in effects:
                adjustment = effects['consecutive']
                turista.satisfaccion = max(0, min(10, turista.satisfaccion + adjustment))
                print(f"DEBUG - Penalización por lugares consecutivos del mismo tipo: {adjustment:.2f}")
            
            elif f'after_{prev_type}' in effects:
                adjustment = effects[f'after_{prev_type}']
                turista.satisfaccion = max(0, min(10, turista.satisfaccion + adjustment))
                print(f"DEBUG - Efecto de secuencia {prev_type}->{current_type}: {adjustment:.2f}")

    def _apply_final_route_adjustments(self, satisfaction: float, route_factors: Dict, num_places: int) -> float:
        """
        Aplica ajustes finales basados en la calidad general de la ruta.
        """
        # Calcular puntuación de calidad general
        overall_quality = (
            route_factors.get('diversity_score', 0) * 0.25 +
            route_factors.get('flow_score', 0) * 0.20 +
            route_factors.get('balance_score', 0) * 0.20 +
            route_factors.get('efficiency_score', 0) * 0.15 +
            route_factors.get('experience_richness', 0) * 0.20
        )
        
        # Bonificación por calidad de ruta
        quality_bonus = (overall_quality - 0.5) * 2  # Rango ±1.0
        
        # Bonificación por número óptimo de lugares (3-5 lugares es ideal)
        places_bonus = 0
        if 3 <= num_places <= 5:
            places_bonus = 0.5
        elif num_places == 2 or num_places == 6:
            places_bonus = 0.2
        elif num_places == 1 or num_places >= 7:
            places_bonus = -0.3
        
        # Aplicar ajustes
        final_satisfaction = satisfaction + quality_bonus + places_bonus
        final_satisfaction = max(0, min(10, final_satisfaction))
        
        print(f"DEBUG - Ajustes finales: calidad={quality_bonus:.2f}, lugares={places_bonus:.2f}, "
              f"satisfacción final={final_satisfaction:.2f}")
        
        return final_satisfaction


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