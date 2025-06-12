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


        self.alpha = 0.001 
        self.beta = 10000 
        self.ganma = 80
    

    def find_route(self, tourist_param, time, sort):
        """
        Encuentra una ruta que maximice la función goal_func(route) y cumpla con la restricción de tiempo.

        :param time: Tiempo máximo en horas.
        :type time: float
        :param starting_node: Nodo de partida.
        :type starting_node: int
        :return: La ruta óptima encontrada.
        :rtype: list[int]
        """
        
        self.tourist_param=tourist_param

        
        C=[0]
        route = []

        for i in range (1,self.num_nodes):
            route.append(i)
            C.append(self.node_goal_function(i))
        #Haz que el starting_node sea el primero
        route.insert(0, 0)

        # Inicializa la temperatura
        temperature = self.beta

        # Inicializa la mejor ruta encontrada
        best_route = route

        # Inicializa el mejor valor de la función objetivo
        best_value = self.goal_func(route)


        # Cantidad de iteraciones
        it=0
        # Ciclo de enfriamiento simulado
        while temperature > self.ganma:
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
            if delta > 0 or random.random() < (temperature-self.ganma)/(self.beta-self.ganma)*math.exp(delta / temperature):
                route = new_route

            
            # Enfría la temperatura
            it=it+1
            temperature = self.CoolingFunction(temperature,it)

        answer=[0]
        length = 1 
        for i in range(1, len(best_route)+1):
            if self.get_time(best_route, i) <= time:
                length = i
                answer.append(best_route[i])
            else:
                break
        answer.append(0)

        return answer


    def perturb_route(route):
        # Crea una copia de la lista
        new_route = route[:]  # o new_route = route.copy()

        # Intercambia dos nodos en la ruta
        i = random.randint(1, len(new_route) - 2)
        j = random.randint(1, len(new_route) - 2)
        new_route[i], new_route[j] = new_route[j], new_route[i]
        return new_route

    def cooling_function(self,T, it):
        return T* math.exp(-self.alpha*it)

    #Similitud coseno entre ambos embeddings
    def node_goal_func(self,node_id):
        return numpy.dot(self.node_params[node_id]['vector'], self.tourist_param)/(numpy.linalg.norm(self.node_params[node_id]['vector'])* numpy.linalg.norm(self.tourist_param))

    def goal_func(self, route, time, C):
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
            sum+=C[i]

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