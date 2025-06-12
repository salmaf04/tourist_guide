import numpy as np
import random
import math
from MetaH.simulated_annealing import RouteFinder


class RouteOptimizer:
    def __init__(self, adjacency_matrix, node_params, max_time, tourist_param):
        """
        Constructor de la clase RouteOptimizer.

        :param adjacency_matrix: Matriz de adyacencia que representa la gráfica.
        :type adjacency_matrix: list[list[float]]
        :param node_params: Vector de parámetros de los nodos.
        :type node_params: list[dict]
        :param max_time: Tiempo máximo para la ruta.
        :type max_time: float
        :param tourist_param: Parámetro del turista.
        :type tourist_param: numpy.ndarray
        """
        self.adjacency_matrix = adjacency_matrix
        self.node_params = node_params
        self.max_time = max_time
        self.tourist_param = tourist_param
        self.num_nodes = len(adjacency_matrix)
        self.RouteFinder = RouteFinder(self.adjacency_matrix, self.node_params)

    def get_routes(self):
        routes = []
        # Generate 3 routes with original parameters
        for _ in range(0, 3):
            route = self.RouteFinder.find_route(self.tourist_param, self.max_time, 0)
            routes.append(route)
        
        # Generate 17 routes with perturbed parameters
        for i in range(0, 17):
            paramcop = self.tourist_param.copy()
            c = np.random.normal(loc=1, scale=0.1, size=len(paramcop))  # Reduced scale for stability
            paramcop *= c
            route = self.RouteFinder.find_route(paramcop, self.max_time, 0)  # Always start from node 0
            routes.append(route)
        return routes