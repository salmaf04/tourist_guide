import random
import math
import numpy
import RAG.rag

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
        self.ra=RAG.rag.RAGPlanner()
        self.distance_matrix = distance_matrix
        self.node_params = node_params
        self.num_nodes = len(distance_matrix)


        self.alpha = 0.001 
        self.beta = 10000 
        self.ganma = 80
    

    def find_route(self, tourist_param, time, randomize=False):
        """
        Encuentra una ruta que maximice la función goal_func(route) y cumpla con la restricción de tiempo.

        :param time: Tiempo máximo en horas.
        :type time: float
        :param tourist_param: Embedding del Turista
        :type tourist_param: ndarray
        :return: La ruta óptima encontrada.
        :rtype: list[int]
        """

        
        C=[0]
        route = [0]

        for i in range (1,self.num_nodes):
            route.append(i)
            C.append(self.node_goal_func(i,tourist_param))

        # Inicializa la temperatura
        temperature = self.beta

        # Inicializa la mejor ruta encontrada
        best_route = route

        # Inicializa el mejor valor de la función objetivo
        best_value = self.goal_func(route, time, tourist_param)


        # Cantidad de iteraciones
        it=0
        # Ciclo de enfriamiento simulado
        while temperature > self.ganma:
            # Genera una nueva solución vecina
            new_route = self.perturb_route(route)

            # Calcula el valor de la función objetivo para la nueva ruta
            new_value = self.goal_func(new_route, time, tourist_param)

            # Verifica si la nueva ruta es mejor que la mejor ruta encontrada hasta ahora
            if new_value > best_value:
                best_route = new_route
                best_value = new_value

            # Funcion de Aceptación de Simulated Annealing
            delta = new_value - self.goal_func(route, time, tourist_param)
            if delta > 0 or random.random() < (temperature-self.ganma)/(self.beta-self.ganma)*math.exp(delta / temperature):
                route = new_route

            
            # Enfría la temperatura
            it=it+1
            temperature = self.cooling_function(temperature, it)

        answer=[0] 
        for i in range(1, len(best_route)):
            if self.get_time(route, i+1) <= time:
                answer.append(best_route[i])
            else:
                break
        answer.append(0)

        return answer


    def perturb_route(self, route):
        # Crea una copia de la lista
        new_route = route[:]  # o new_route = route.copy()

        # Intercambia dos nodos en la ruta
        if len(new_route) > 2:
            i = random.randint(1, len(new_route) - 1)
            j = random.randint(1, len(new_route) - 1)
            while i == j:
                j = random.randint(1, len(new_route) - 1)
            new_route[i], new_route[j] = new_route[j], new_route[i]
        return new_route

    def cooling_function(self,T, it):
        return T* math.exp(-self.alpha*it)

    #Similitud coseno entre ambos embeddings
    def node_goal_func(self,node_id,tourist_param):
        return self.ra.calculate_cosine_similarity(tourist_param,[self.node_params[node_id]['vector']])[0] 
       

    def goal_func(self, route, time, tourist_param):
        """
        Función objetivo que se maximiza.

        :param route: Ruta.
        :type route: list[int]
        :param time: Tiempo máximo disponible
        :type time: float
        :param tourist_param: Parámetros del turista
        :type tourist_param: numpy.ndarray
        :return: Valor de la función objetivo.
        :rtype: float
        """
        length = 1 
        for i in range(1, len(route)):
            if self.get_time(route, i+1) <= time:
                length = i + 1
            else:
                break

        sum=0
        for i in range(1,length):
            sum+=self.node_goal_func(route[i],tourist_param)

        return sum

    def get_time(self, route, length):
        """
        Calcula el tiempo total de la ruta.

        :param route: Ruta.
        :type route: list[int]
        :param length: Longitud de la ruta a considerar
        :type length: int
        :return: Tiempo total.
        :rtype: float
        """
        if length <= 1:
            return 0
            
        t = 0
        for i in range(1, length):
            # Add travel time from previous node to current node
            t += self.distance_matrix[route[i-1]][route[i]]
            
            # Add visit time at current node
            t += self.node_params[route[i]]['time']

        # Add return time to starting point
        t += self.distance_matrix[route[length-1]][route[0]]
 
        return t