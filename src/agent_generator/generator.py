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
import google.generativeai as genai
import time

GEMINI_API_KEY = 'AIzaSyCkN0mxdFQpGajEwB8sZm2fUsJzhpTCfvk'  
genai.configure(api_key=GEMINI_API_KEY)

class GeminiClient:
    """
    Cliente para interactuar con la API de Gemini.
    """
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        self.generation_config = {
            'temperature': 0.9,
            'top_p': 1,
            'top_k': 32,
            'max_output_tokens': 200,
        }
        self.max_retries = 12
        self.retry_delay = 5  # seconds

    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        """
        Genera una respuesta usando Gemini con reintentos por límite de tasa.
        """
        retries = 0
        while retries < self.max_retries:
            try:
                if system:
                    prompt = f"{system}\n{prompt}"
                
                safety_settings = [
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                ]
                
                response = self.model.generate_content(
                    prompt,
                    generation_config=self.generation_config,
                    safety_settings=safety_settings
                )
                
                if response.text:
                    return response.text.strip()
                return "[Respuesta no disponible]"
                
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    retries += 1
                    if retries < self.max_retries:
                        print(f"\nLímite de solicitudes alcanzado. Esperando {self.retry_delay} segundos...")
                        time.sleep(self.retry_delay)
                        continue
                print(f"Error con Gemini después de {retries} intentos: {e}")
                return "[Error de generación]"
            
        return "[Límite de solicitudes excedido]"

gemini_client = GeminiClient()

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

class Agente(Agent):
    """
    Agente de la simulación, asociado a un rol y un lugar.
    """
    def __init__(self, unique_id: str, model, rol: str, lugar_id: str, prompt: str):
        super().__init__(model)
        self.unique_id = unique_id
        self.rol = rol
        self.lugar_id = lugar_id
        self.prompt = prompt
        self.interacciones = []

class Turista(Agent):
    """
    Representa al turista, con memoria priorizada y satisfacción.
    """
    def __init__(self, unique_id: int, model, nombre: str):
        super().__init__(model)
        self.unique_id = unique_id
        self.nombre = nombre
        self.satisfaccion = 5.0
        self.memoria_alta = []
        self.memoria_media = []
        self.memoria_baja = []
        self.contexto_actual = []

    LIMITE_ALTA = 5
    LIMITE_MEDIA = 7
    LIMITE_BAJA = 10

    def agregar_experiencia(self, texto: str, impacto: float):
        """
        Agrega una experiencia a la memoria, asignando prioridad según el impacto.
        Si la cola se llena, degrada el recuerdo o lo olvida.
        """
        print(f"DEBUG - Agregando experiencia: impacto={impacto:.2f}, satisfacción_antes={self.satisfaccion:.2f}")
        
        if abs(impacto) >= 1.2:
            cola = self.memoria_alta
            limite = self.LIMITE_ALTA
            print(f"DEBUG - Clasificado como memoria ALTA")
        elif abs(impacto) >= 0.5:
            cola = self.memoria_media
            limite = self.LIMITE_MEDIA
            print(f"DEBUG - Clasificado como memoria MEDIA")
        else:
            cola = self.memoria_baja
            limite = self.LIMITE_BAJA
            print(f"DEBUG - Clasificado como memoria BAJA")

        cola.append((texto, impacto))
        
        # Handle memory overflow with degradation
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
        
        # Update satisfaction
        satisfaccion_anterior = self.satisfaccion
        self.satisfaccion = max(0, min(10, self.satisfaccion + impacto))
        print(f"DEBUG - Satisfacción actualizada: {satisfaccion_anterior:.2f} -> {self.satisfaccion:.2f} (cambio: {self.satisfaccion - satisfaccion_anterior:.2f})")

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
            # Ensure values are within valid ranges
            amabilidad_valor = max(0, min(10, amabilidad_valor))
            satisfaccion_actual = max(0, min(10, satisfaccion_actual))
            
            self.simulador.input['amabilidad'] = amabilidad_valor
            self.simulador.input['satisfaccion'] = satisfaccion_actual
            self.simulador.compute()
            
            impacto = self.simulador.output.get('impacto', 0.0)
            print(f"DEBUG - Fuzzy system: amabilidad={amabilidad_valor:.2f}, satisfaccion={satisfaccion_actual:.2f}, impacto={impacto:.2f}")
            return impacto
        except Exception as e:
            print(f"Error en calcular_impacto: {e}")
            # Return a random impact as fallback
            fallback_impact = random.uniform(-1.0, 1.0)
            print(f"DEBUG - Using fallback impact: {fallback_impact:.2f}")
            return fallback_impact

sistema_difuso = SistemaDifusoImpacto()

class GeneradorAgentes:
    """
    Genera agentes y sus prompts usando Gemini.
    """
    @staticmethod
    def generar_prompt_agente(rol: str, nodo: Nodo) -> str:
        """
        Genera un prompt para un agente usando Gemini.
        """
        prompt = f"Crea una descripción para un {rol} en {nodo.nombre} (2 oraciones)."
        descripcion = gemini_client.generate(prompt)
        if descripcion == "[Respuesta no disponible]":
            return f"Un {rol} en {nodo.nombre}"
        return descripcion

    @classmethod
    def crear_agente(cls, rol: str, nodo: Nodo, model: Model) -> Agente:
        """
        Crea una instancia de Agente con su prompt generado.
        """
        descripcion = cls.generar_prompt_agente(rol, nodo)
        unique_id = f"{rol}_{nodo.id}_{random.randint(1000, 9999)}"
        prompt = f"Eres un {rol} en {nodo.nombre}. {descripcion}"
        return Agente(unique_id, model, rol, nodo.id, prompt)

class SimuladorInteracciones:
    """
    Simula interacciones entre el turista y los agentes.
    """
    @staticmethod
    def interactuar(turista: Turista, agente: Agente, nodo: Nodo, max_interacciones: int = 1):
        """
        Realiza una serie de interacciones entre un turista y un agente en un nodo.
        """
        interacciones_realizadas = 0
        while interacciones_realizadas < max_interacciones:
            contexto = "\n".join([
                f"Contexto previo: {ctx['content']}" 
                for ctx in turista.contexto_actual[-2:] if ctx.get('content')
            ])
            
            prompt = (
                f"Como {agente.rol} en {nodo.nombre}, interactúa con el turista {turista.nombre} "
                f"que está explorando el lugar. {nodo.descripcion}\n"
                f"Contexto de la conversación:\n{contexto}\n"
                f"{turista.nombre}: [Explora {nodo.nombre}]\n"
                f"Responde en español con una frase corta y amigable."
            )
            
            respuesta_texto = gemini_client.generate(prompt)
            
            if not respuesta_texto or respuesta_texto in ["[Respuesta no disponible]", "[Error de generación]"]:
                respuestas_fallback = [
                    f"¡Bienvenido a {nodo.nombre}! Es un placer ayudarte.",
                    f"Te recomiendo explorar esta área, es muy interesante.",
                    f"¿Hay algo específico que te gustaría saber sobre {nodo.nombre}?",
                    f"Espero que disfrutes tu visita a {nodo.nombre}.",
                    f"Este lugar tiene mucha historia, déjame contarte algo interesante."
                ]
                respuesta_texto = random.choice(respuestas_fallback)
                print(f"API falló, usando respuesta fallback: {respuesta_texto}")
            else:
                print(f"Respuesta de API: {respuesta_texto}")
            
            amabilidad_valor = random.uniform(2, 10)
            
            print(f"DEBUG - Antes del impacto: satisfacción={turista.satisfaccion:.2f}")
            impacto = sistema_difuso.calcular_impacto(amabilidad_valor, turista.satisfaccion)
            
            if abs(impacto) < 0.1:
                if amabilidad_valor > 7:
                    impacto = random.uniform(0.3, 1.2)
                elif amabilidad_valor < 4:
                    impacto = random.uniform(-1.2, -0.3)
                else:
                    impacto = random.uniform(-0.5, 0.5)
                print(f"DEBUG - Impacto ajustado manualmente: {impacto:.2f}")
            
            experiencia = f"{nodo.nombre} ({agente.rol}): {respuesta_texto}"
            
            print(f"DEBUG - Agregando experiencia con impacto: {impacto:.2f}")
            turista.agregar_experiencia(experiencia, impacto)
            turista.contexto_actual.append({"role": "assistant", "content": respuesta_texto})
            
            agente.interacciones.append({
                "turista": turista.nombre,
                "respuesta": respuesta_texto,
                "impacto": impacto,
                "turno": interacciones_realizadas + 1
            })
            
            print(f"DEBUG - Nueva experiencia agregada: {experiencia[:80]}...")
            print(f"DEBUG - Impacto aplicado: {impacto:.2f}")
            print(f"DEBUG - Satisfacción después: {turista.satisfaccion:.2f}")
            print(f"DEBUG - Total memorias: Alta={len(turista.memoria_alta)}, Media={len(turista.memoria_media)}, Baja={len(turista.memoria_baja)}")
            
            interacciones_realizadas += 1
            if random.random() > 0.8:
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
        print(f"DEBUG - Creados {len(self.nodos)} nodos")
        
        self.turista = Turista(0, self, nombre_turista)
        self.schedule.add(self.turista)
        print(f"DEBUG - Turista agregado al schedule")
        
        agentes_creados = 0
        for nodo in self.nodos:
            print(f"DEBUG - Procesando nodo {nodo.nombre} con agentes: {nodo.agentes}")
            for rol in nodo.agentes:
                agente = GeneradorAgentes.crear_agente(rol, nodo, self)
                self.schedule.add(agente)
                agentes_creados += 1
                print(f"DEBUG - Agente creado: {agente.rol} en {nodo.nombre} (ID: {agente.unique_id})")
        
        print(f"DEBUG - Total agentes creados: {agentes_creados}")
        print(f"DEBUG - Total agentes en schedule: {len(self.schedule.agents)}")

    def step(self):
        """
        Ejecuta un paso de simulación: posibles interacciones en cada nodo y recolección de datos.
        """
        print(f"DEBUG - Iniciando paso, satisfacción actual: {self.turista.satisfaccion:.2f}")
        print(f"DEBUG - Total agentes en schedule: {len(self.schedule.agents)}")
        
        for nodo in self.nodos:
            agentes_en_nodo = [a for a in self.schedule.agents if isinstance(a, Agente) and a.lugar_id == nodo.id]
            print(f"DEBUG - Nodo {nodo.nombre}: {len(agentes_en_nodo)} agentes encontrados")
            
            if agentes_en_nodo:
                agente = random.choice(agentes_en_nodo)
                print(f"DEBUG - Interactuando en {nodo.nombre} con {agente.rol}")
                SimuladorInteracciones.interactuar(self.turista, agente, nodo)
            else:
                print(f"DEBUG - No hay agentes en {nodo.nombre}")
        
        print(f"DEBUG - Fin del paso, satisfacción final: {self.turista.satisfaccion:.2f}")
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
        print(f"\n=== INICIANDO SIMULACIÓN {i+1} ===")
        modelo = ModeloTurismo(lista_nodos, nombre_turista=f"Ana_{i+1}")
        print(f"Satisfacción inicial: {modelo.turista.satisfaccion}")
        
        for paso in range(pasos):
            print(f"\n--- Paso {paso+1} ---")
            modelo.step()
            print(f"Satisfacción después del paso {paso+1}: {modelo.turista.satisfaccion:.2f}")
        
        satisfaccion_final = modelo.turista.satisfaccion
        satisfacciones.append(satisfaccion_final)
        print(f"\nSimulación {i+1}: Satisfacción final = {satisfaccion_final:.1f}/10")
        
        # Debug memory contents
        print(f"Memorias totales: Alta={len(modelo.turista.memoria_alta)}, Media={len(modelo.turista.memoria_media)}, Baja={len(modelo.turista.memoria_baja)}")
        
        recuerdos = modelo.turista.recuerdos_significativos()
        print("Recuerdos más significativos:")
        if recuerdos:
            for rec in recuerdos:
                print(f"- {rec}")
        else:
            print("- No hay recuerdos significativos")
        print("=" * 50)
    promedio = sum(satisfacciones) / len(satisfacciones)
    print(f"\nPromedio de satisfacción tras {n_simulaciones} simulaciones: {promedio:.2f}/10")
    print("Prueba sistema difuso:")
    print("Impacto esperado positivo:", sistema_difuso.calcular_impacto(8.0, 3.0))
    print("Impacto esperado negativo:", sistema_difuso.calcular_impacto(2.0, 8.0))
    return satisfacciones, promedio

if __name__ == "__main__":
    n_simulaciones = 3
    ejecutar_simulaciones(n_simulaciones, pasos=5)