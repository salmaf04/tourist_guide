from typing import List, Dict, Optional
import random
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from mesa import Model, Agent
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from mesa.time import RandomActivation
import os
import requests
from openai import OpenAI

os.environ['GRPC_DNS_RESOLVER'] = 'native'

OPENROUTER_API_KEY = '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'
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json"
}

class OpenRouterClient:
    """
    Cliente para interactuar con la API de OpenRouter.
    """
    def __init__(self, api_key: str, base_url: str = "https://openrouter.ai/api/v1"):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def generate(self, prompt: str, system: Optional[str] = None, model: str = "deepseek/deepseek-r1:free") -> str:
        """
        Genera una respuesta usando el modelo especificado de OpenRouter.
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        data = {
            "model": model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 256
        }
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=data,
                timeout=30
            )
            if response.status_code != 200:
                print("Respuesta de error OpenRouter:", response.status_code, response.text)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"Error con OpenRouter/DeepSeek: {e}")
            return "[Respuesta no disponible]"

openrouter_client = OpenRouterClient(OPENROUTER_API_KEY)

@dataclass
class Nodo:
    """
    Representa un nodo/lugar del recorrido turístico.
    """
    id: str
    nombre: str
    tipo: str
    descripcion: str
    agentes: List[str]

@dataclass(unsafe_hash=True)
class Agente(Agent):
    """
    Agente de la simulación, asociado a un rol y un lugar.
    """
    unique_id: str
    model: Model
    rol: str
    lugar_id: str
    prompt: str
    interacciones: List[Dict] = field(default_factory=list, hash=False)

@dataclass(unsafe_hash=True)
class Turista(Agent):
    """
    Representa al turista, con memoria priorizada y satisfacción.
    """
    unique_id: int
    model: Model
    nombre: str
    satisfaccion: float = 5.0
    memoria_alta: List[tuple] = field(default_factory=list, hash=False)
    memoria_media: List[tuple] = field(default_factory=list, hash=False)
    memoria_baja: List[tuple] = field(default_factory=list, hash=False)
    contexto_actual: List[Dict] = field(default_factory=list, hash=False)

    LIMITE_ALTA = 5
    LIMITE_MEDIA = 7
    LIMITE_BAJA = 10

    def agregar_experiencia(self, texto: str, impacto: float):
        """
        Agrega una experiencia a la memoria, asignando prioridad según el impacto.
        Si la cola se llena, degrada el recuerdo o lo olvida.
        """
        if abs(impacto) >= 1.2:
            cola = self.memoria_alta
            limite = self.LIMITE_ALTA
        elif abs(impacto) >= 0.5:
            cola = self.memoria_media
            limite = self.LIMITE_MEDIA
        else:
            cola = self.memoria_baja
            limite = self.LIMITE_BAJA

        cola.append((texto, impacto))
        if len(cola) > limite:
            degradado = cola.pop(0)
            if cola is self.memoria_alta:
                self.memoria_media.append(degradado)
                if len(self.memoria_media) > self.LIMITE_MEDIA:
                    degradado2 = self.memoria_media.pop(0)
                    self.memoria_baja.append(degradado2)
                    if len(self.memoria_baja) > self.LIMITE_BAJA:
                        self.memoria_baja.pop(0)
            elif cola is self.memoria_media:
                self.memoria_baja.append(degradado)
                if len(self.memoria_baja) > self.LIMITE_BAJA:
                    self.memoria_baja.pop(0)
        self.satisfaccion = max(0, min(10, self.satisfaccion + impacto))

    def recuerdos_significativos(self, n=5):
        """
        Devuelve los n recuerdos de mayor prioridad (alta > media > baja).
        """
        todos_ordenados = (
            sorted(self.memoria_alta, key=lambda x: abs(x[1]), reverse=True) +
            sorted(self.memoria_media, key=lambda x: abs(x[1]), reverse=True) +
            sorted(self.memoria_baja, key=lambda x: abs(x[1]), reverse=True)
        )
        return [texto for texto, _ in todos_ordenados[:n]]

    @property
    def memoria(self):
        """
        Devuelve todos los recuerdos actuales, ordenados por prioridad.
        """
        return [texto for texto, _ in self.memoria_alta + self.memoria_media + self.memoria_baja]

class SistemaDifusoImpacto:
    """
    Sistema difuso para calcular el impacto de una interacción.
    """
    def __init__(self):
        self.satisfaccion = ctrl.Antecedent(np.arange(0, 11, 1), 'satisfaccion')
        self.amabilidad = ctrl.Antecedent(np.arange(0, 11, 1), 'amabilidad')
        self.impacto = ctrl.Consequent(np.arange(-2, 2.1, 0.1), 'impacto')
        self._definir_variables()
        self._definir_reglas()
        self.sistema_control = ctrl.ControlSystem(self.reglas)
        self.simulador = ctrl.ControlSystemSimulation(self.sistema_control)

    def _definir_variables(self):
        """
        Define las variables lingüísticas y sus funciones de membresía.
        """
        self.satisfaccion['baja'] = fuzz.trimf(self.satisfaccion.universe, [0, 0, 5])
        self.satisfaccion['media'] = fuzz.trimf(self.satisfaccion.universe, [3, 5, 7])
        self.satisfaccion['alta'] = fuzz.trimf(self.satisfaccion.universe, [5, 10, 10])
        self.amabilidad['poca'] = fuzz.trimf(self.amabilidad.universe, [0, 0, 4])
        self.amabilidad['normal'] = fuzz.trimf(self.amabilidad.universe, [3, 5, 7])
        self.amabilidad['mucha'] = fuzz.trimf(self.amabilidad.universe, [6, 10, 10])
        self.impacto['negativo'] = fuzz.trimf(self.impacto.universe, [-2, -1, 0])
        self.impacto['neutral'] = fuzz.trimf(self.impacto.universe, [-0.5, 0, 0.5])
        self.impacto['positivo'] = fuzz.trimf(self.impacto.universe, [0, 1, 2])

    def _definir_reglas(self):
        """
        Define las reglas difusas para el sistema.
        """
        self.reglas = [
            ctrl.Rule(self.amabilidad['mucha'] & self.satisfaccion['baja'], self.impacto['positivo']),
            ctrl.Rule(self.amabilidad['poca'] & self.satisfaccion['alta'], self.impacto['negativo']),
            ctrl.Rule(self.amabilidad['normal'] & self.satisfaccion['media'], self.impacto['positivo']),
            ctrl.Rule(self.amabilidad['poca'] & self.satisfaccion['baja'], self.impacto['negativo']),
            ctrl.Rule(self.amabilidad['mucha'] & self.satisfaccion['alta'], self.impacto['positivo']),
            ctrl.Rule(self.amabilidad['normal'] & self.satisfaccion['alta'], self.impacto['neutral']),
            ctrl.Rule(self.amabilidad['normal'] & self.satisfaccion['baja'], self.impacto['positivo']),
            ctrl.Rule(self.amabilidad['poca'] & self.satisfaccion['media'], self.impacto['neutral']),
            ctrl.Rule(self.amabilidad['mucha'] & self.satisfaccion['media'], self.impacto['positivo']),
        ]

    def calcular_impacto(self, amabilidad_valor: float, satisfaccion_actual: float) -> float:
        """
        Calcula el impacto de una interacción usando lógica difusa.
        """
        try:
            self.simulador.input['amabilidad'] = amabilidad_valor
            self.simulador.input['satisfaccion'] = satisfaccion_actual
            self.simulador.compute()
            return self.simulador.output.get('impacto', 0.0)
        except Exception as e:
            print(f"Error en calcular_impacto: {e}")
            return 0.0

sistema_difuso = SistemaDifusoImpacto()

class GeneradorAgentes:
    """
    Genera agentes y sus prompts usando OpenRouter.
    """
    @staticmethod
    def generar_prompt_agente(rol: str, nodo: Nodo) -> str:
        """
        Genera un prompt para un agente usando OpenRouter.
        """
        prompt = f"Crea una descripción para un {rol} en {nodo.nombre} (2 oraciones)."
        descripcion = openrouter_client.generate(prompt)
        if descripcion == "[Respuesta no disponible]":
            return f"Un {rol} en {nodo.nombre}"
        return descripcion

    @classmethod
    def crear_agente(cls, rol: str, nodo: Nodo, model: Model) -> Agente:
        """
        Crea una instancia de Agente con su prompt generado.
        """
        descripcion = cls.generar_prompt_agente(rol, nodo)
        return Agente(
            unique_id=f"{rol}_{nodo.id}_{random.randint(1000, 9999)}",
            model=model,
            rol=rol,
            lugar_id=nodo.id,
            prompt=f"Eres un {rol} en {nodo.nombre}. {descripcion}"
        )

class SimuladorInteracciones:
    """
    Simula interacciones entre el turista y los agentes.
    """
    @staticmethod
    def interactuar(turista: Turista, agente: Agente, nodo: Nodo, max_interacciones: int = 3):
        """
        Realiza una serie de interacciones entre un turista y un agente en un nodo.
        """
        interacciones_realizadas = 0
        while interacciones_realizadas < max_interacciones:
            prompt = agente.prompt + "\n"
            for ctx in turista.contexto_actual:
                prompt += f"{ctx.get('role', '')}: {ctx.get('content', '')}\n"
            prompt += f"{turista.nombre}: [Explora {nodo.nombre}]"
            respuesta_texto = openrouter_client.generate(prompt)
            if respuesta_texto == "[Respuesta no disponible]":
                respuesta_texto = f"{agente.rol}: [Respuesta no disponible]"
                amabilidad_valor = 5.0
            else:
                amabilidad_valor = random.uniform(6, 9)
            impacto = sistema_difuso.calcular_impacto(amabilidad_valor, turista.satisfaccion)
            experiencia = f"{nodo.nombre} ({agente.rol}): {respuesta_texto}"
            turista.agregar_experiencia(experiencia, impacto)
            turista.contexto_actual.append({"role": "assistant", "content": respuesta_texto})
            agente.interacciones.append({
                "turista": turista.nombre,
                "respuesta": respuesta_texto,
                "impacto": impacto,
                "turno": interacciones_realizadas + 1
            })
            interacciones_realizadas += 1
            if random.random() > 0.6:
                break

class ModeloTurismo(Model):
    """
    Modelo principal de la simulación de turismo.
    Maneja la creación de nodos, agentes y el ciclo de simulación.
    """
    def __init__(self, lista_nodos: List[Dict], nombre_turista: str = "Turista"):
        super().__init__()
        self.grid = MultiGrid(10, 10, torus=True)
        self.schedule = RandomActivation(self)
        self.datacollector = DataCollector(
            agent_reporters={"Satisfacción": lambda a: getattr(a, 'satisfaccion', None)}
        )
        self.nodos = [Nodo(**nodo_data) for nodo_data in lista_nodos]
        self.turista = Turista(
            unique_id=0,
            model=self,
            nombre=nombre_turista
        )
        self.schedule.add(self.turista)
        for nodo in self.nodos:
            for rol in nodo.agentes:
                agente = GeneradorAgentes.crear_agente(rol, nodo, self)
                self.schedule.add(agente)

    def step(self):
        """
        Ejecuta un paso de simulación: posibles interacciones en cada nodo y recolección de datos.
        """
        for nodo in self.nodos:
            agentes_en_nodo = [a for a in self.schedule.agents if isinstance(a, Agente) and a.lugar_id == nodo.id]
            if agentes_en_nodo and random.random() > 0.3:
                agente = random.choice(agentes_en_nodo)
                SimuladorInteracciones.interactuar(self.turista, agente, nodo)
        self.datacollector.collect(self)

def ejecutar_simulaciones(n_simulaciones: int, pasos: int = 10):
    """
    Ejecuta varias simulaciones y muestra la satisfacción y recuerdos del turista.
    """
    lista_nodos = [
        {
            "id": "museo_1",
            "nombre": "Museo de Arte",
            "tipo": "museo",
            "descripcion": "Museo con obras modernas y clásicas.",
            "agentes": ["guía", "curador"]
        },
        {
            "id": "restaurante_1",
            "nombre": "Café Central",
            "tipo": "restaurante",
            "descripcion": "Famoso por su café orgánico.",
            "agentes": ["mesero"]
        }
    ]
    satisfacciones = []
    for i in range(n_simulaciones):
        modelo = ModeloTurismo(lista_nodos, nombre_turista=f"Ana_{i+1}")
        for _ in range(pasos):
            modelo.step()
        satisfaccion_final = modelo.turista.satisfaccion
        satisfacciones.append(satisfaccion_final)
        print(f"Simulación {i+1}: Satisfacción final = {satisfaccion_final:.1f}/10")
        recuerdos = modelo.turista.recuerdos_significativos()
        print("Recuerdos más significativos:")
        for rec in recuerdos:
            print(f"- {rec}")
    promedio = sum(satisfacciones) / len(satisfacciones)
    print(f"\nPromedio de satisfacción tras {n_simulaciones} simulaciones: {promedio:.2f}/10")
    print("Prueba sistema difuso:")
    print("Impacto esperado positivo:", sistema_difuso.calcular_impacto(8.0, 3.0))
    print("Impacto esperado negativo:", sistema_difuso.calcular_impacto(2.0, 8.0))
    return satisfacciones, promedio

if __name__ == "__main__":
    n_simulaciones = 5
    ejecutar_simulaciones(n_simulaciones)