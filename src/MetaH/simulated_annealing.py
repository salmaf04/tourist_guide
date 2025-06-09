import random
import math
import numpy

class RouteFinder:
    def __init__(self, distance_matrix, node_params):
        """
        Constructor de la clase RouteFinder.

        :param distance_matrix: Matriz de distancias entre nodos.
        :type distance_matrix: list[list[float]]
        :param node_params: Array de parámetros de los nodos.
        :type node_params: list[dict]
        :param tourist_param: Valor del parámetro del turista.
        :type tourist_param: float
        """
        self.distance_matrix = distance_matrix
        self.node_params = node_params
        self.num_nodes = len(distance_matrix)
        self.num_cities = len(node_params)
    

    def find_route(self, tourist_param, time, starting_node):
        """
        Encuentra una ruta que maximice la función goal_func(route) y cumpla con la restricción de tiempo.

        :param time: Tiempo máximo en horas.
        :type time: float
        :param starting_node: Nodo de partida.
        :type starting_node: int
        :return: La ruta óptima encontrada.
        :rtype: list[int]
        """

        route = random.sample(range(self.num_cities), self.num_cities)
        #Haz que el starting_node sea el primero
        route.remove(starting_node)
        route.insert(0, starting_node)

        # Inicializa la temperatura
        temperature = 1000.0

        # Inicializa la mejor ruta encontrada
        best_route = route

        # Inicializa el mejor valor de la función objetivo
        best_value = self.goal_func(route)


        # Ciclo de enfriamiento simulado
        while temperature > 1.0:
            # Genera una nueva solución vecina
            new_route = self.perturb_route(route)

            # Calcula el valor de la función objetivo para la nueva ruta
            new_value = self.goal_func(new_route, time)

            # Verifica si la nueva ruta es mejor que la mejor ruta encontrada hasta ahora
            if new_value > best_value:
                best_route = new_route
                best_value = new_value

            # Aplica la función de aceptación de Metropolis
            delta = new_value - self.goal_func(route, time)
            if delta > 0 or random.random() < math.exp(delta / temperature):
                route = new_route

            # Enfría la temperatura
            temperature *= 0.99

        return best_route


    def perturb_route(route):
        # Crea una copia de la lista
        new_route = route[:]  # o new_route = route.copy()

        # Intercambia dos nodos en la ruta
        i = random.randint(1, len(new_route) - 2)
        j = random.randint(max(1,i-3), min(len(new_route) - 2),i+3)
        new_route[i], new_route[j] = new_route[j], new_route[i]
        return new_route


    def goal_func(self, route, time):
        """
        Función objetivo que se maximiza.

        :param route: Ruta.
        :type route: list[int]
        :return: Valor de la función objetivo.
        :rtype: float
        """
        length = 1 
        for i in range(1, len(route)+1):
            if self.get_time(route, i) <= time:
                length = i
            else:
                break

        sum=0
        for i in range(1,length):
            #Dot product entre los dos vectores
            sum+=numpy.dot(self.node_params[i]['vector'], self.tourist_param)

        return sum

    def get_time(self, route, length):
        """
        Calcula el tiempo total de la ruta.

        :param route: Ruta.
        :type route: list[int]
        :return: Tiempo total.
        :rtype: float
        """
        t=0
        for i in range(1, length):
            t+=self.distance_matrix[route[i-1]][route[i]]
            t+=self.node_params[route[i]]['time']

        t+=self.distance_matrix[route[length-1]][route[0]]
 
        return t