import numpy as np
import random
import math
import MetaH.simulated_annealing


class RouteOptimizer:
    def __init__(self, adjacency_matrix, node_params, max_time, tourist_param:numpy.ndarray):
        """
        Constructor de la clase RouteOptimizer.

        :param adjacency_matrix: Matriz de adyacencia que representa la gráfica.
        :type adjacency_matrix: list[list[float]]
        :param node_params: Vector de parámetros de los nodos.
        :type node_params: list[dict]
        :param max_time: Tiempo máximo para la ruta.
        :type max_time: float
        :param tourist_param: Parámetro del turista.
        :type tourist_param: float
        """
        self.adjacency_matrix = adjacency_matrix
        self.node_params = node_params
        self.max_time = max_time
        self.tourist_param = tourist_param
        self.num_nodes = len(adjacency_matrix)
        self.RouteFinder = MetaH.simulated_annealing.RouteFinder(self.adjacency_matrix, self.node_params)

    def get_routes(self):
        routes = []
        for _ in range(0,8):
            routes.append(self.RouteFinder.find_route(self.tourist_param, self.max_time, False))
        for _ in range(0,12):
            paramcop=self.tourist_param.copy()
            c = np.random.normal(loc=1, scale=1, size=len(paramcop))
            paramcop*=c
            route = self.RouteFinder.find_route(paramcop, self.max_time, False)
            routes.append(route)
        return routes