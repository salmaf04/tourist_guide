import random 
from mesa import Model
from typing import List, Dict

from mesa.space import MultiGrid
from mesa import DataCollector
import google.generativeai as genai
from .fuzzy_system import SistemaDifusoImpacto
from .client import GeminiClient
from .models import Turista, Agente, Nodo

GEMINI_API_KEY = 'AIzaSyCkN0mxdFQpGajEwB8sZm2fUsJzhpTCfvk'  
genai.configure(api_key=GEMINI_API_KEY)
sistema_difuso = SistemaDifusoImpacto()
gemini_client = GeminiClient()


# Simple scheduler to replace RandomActivation 
class SimpleScheduler:
    def __init__(self, model):
        self.model = model
        self.agents = []
    
    def add(self, agent):
        self.agents.append(agent)
    
    def step(self):
        # Shuffle agents for random activation
        agents_shuffled = self.agents.copy()
        random.shuffle(agents_shuffled)
        for agent in agents_shuffled:
            if hasattr(agent, 'step'):
                agent.step()

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
        self.schedule = SimpleScheduler(self)
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