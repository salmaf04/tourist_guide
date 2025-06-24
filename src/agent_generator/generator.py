import random 
from mesa import Model
from typing import List, Dict

from mesa.space import MultiGrid
from mesa import DataCollector
import google.generativeai as genai
from .fuzzy_system import SistemaDifusoImpacto
from .mistral_client import MistralClient
from .models import AgenteBDIBase, TuristaBDI, Nodo
from .fipa_acl import (
    ACLMessage, Performative, get_messaging_system,
    create_request_message, create_recommendation_message, create_inform_message,
    cleanup_messaging_system
)

GEMINI_API_KEY = 'AIzaSyCkN0mxdFQpGajEwB8sZm2fUsJzhpTCfvk'  
genai.configure(api_key=GEMINI_API_KEY)
sistema_difuso = SistemaDifusoImpacto()
mistral_client = MistralClient()


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
    Genera agentes BDI y sus prompts usando Gemini.
    """
    @staticmethod
    def generar_prompt_agente(rol: str, nodo: Nodo) -> str:
        """
        Genera un prompt para un agente usando Gemini.
        """
        prompt = f"Crea una descripción para un {rol} en {nodo.nombre} (2 oraciones)."
        descripcion = mistral_client.generate(prompt)
        if descripcion == "[Respuesta no disponible]":
            return f"Un {rol} en {nodo.nombre}"
        return descripcion

    @classmethod
    def crear_agente(cls, rol: str, nodo: Nodo, model: Model) -> AgenteBDIBase:
        """
        Crea una instancia de Agente con su prompt generado.
        """
        try:
            print(f"DEBUG - Generando prompt para {rol} en {nodo.nombre}")
            descripcion = cls.generar_prompt_agente(rol, nodo)
            unique_id = f"{rol}_{nodo.id}_{random.randint(1000, 9999)}"
            prompt = f"Eres un {rol} en {nodo.nombre}. {descripcion}"
            print(f"DEBUG - Creando AgenteBDIBase con unique_id={unique_id}")
            return AgenteBDIBase(unique_id, model, rol, nodo.id, prompt)
        except Exception as e:
            print(f"ERROR - Fallo en crear_agente: {str(e)}")
            print(f"ERROR - Tipo de error: {type(e)}")
            raise e

class SimuladorInteracciones:
    """
    Simula interacciones entre el turista y los agentes BDI usando protocolo FIPA-ACL.
    """
    @staticmethod
    def interactuar(turista: TuristaBDI, agente: AgenteBDIBase, nodo: Nodo, max_interacciones: int = 1):
        """
        Realiza una serie de interacciones entre un turista y un agente en un nodo usando mensajería FIPA-ACL.
        """
        print(f"DEBUG - Iniciando interacción FIPA-ACL entre {turista.nombre} y {agente.rol} en {nodo.nombre}")
        
        interacciones_realizadas = 0
        while interacciones_realizadas < max_interacciones:
            # El turista inicia la comunicación con una consulta o solicitud
            conversation_id = SimuladorInteracciones._initiate_tourist_communication(turista, agente, nodo)
            
            # El agente responde usando su sistema de mensajería BDI
            response_success = SimuladorInteracciones._process_agent_response(agente, turista, nodo, conversation_id)
            
            if response_success:
                # Calcular impacto de la interacción
                impacto = SimuladorInteracciones._calculate_interaction_impact(agente, nodo)
                
                # Crear experiencia para el turista
                experiencia = SimuladorInteracciones._create_experience_from_messaging(agente, nodo, impacto)
                
                print(f"DEBUG - Experiencia FIPA-ACL creada: {experiencia[:80]}...")
                print(f"DEBUG - Impacto calculado: {impacto:.2f}")
                
                # Agregar experiencia al turista
                turista.agregar_experiencia(experiencia, impacto)
                
                # Registrar interacción en el agente
                agente.interacciones.append({
                    "turista": turista.nombre,
                    "respuesta": experiencia,
                    "impacto": impacto,
                    "turno": interacciones_realizadas + 1,
                    "protocol": "fipa-acl",
                    "conversation_id": conversation_id
                })
                
                print(f"DEBUG - Satisfacción después de FIPA-ACL: {turista.satisfaccion:.2f}")
                print(f"DEBUG - Total memorias: Alta={len(turista.memoria_alta)}, Media={len(turista.memoria_media)}, Baja={len(turista.memoria_baja)}")
            
            interacciones_realizadas += 1
            if random.random() > 0.8:
                break
    
    @staticmethod
    def _initiate_tourist_communication(turista: TuristaBDI, agente: AgenteBDIBase, nodo: Nodo) -> str:
        """Inicia comunicación del turista usando protocolo FIPA-ACL"""
        # Determinar tipo de mensaje según el contexto
        if turista.satisfaccion < 4:
            # Turista insatisfecho busca ayuda
            message = create_request_message(
                sender=str(turista.unique_id),
                receiver=str(agente.unique_id),
                content=f"Necesito ayuda para disfrutar mejor mi visita a {nodo.nombre}",
                context={
                    'location': nodo.nombre,
                    'tourist_satisfaction': turista.satisfaccion,
                    'urgency': 'high' if turista.satisfaccion < 3 else 'medium'
                }
            )
        elif agente.rol in ['guía', 'curador']:
            # Solicitar información a agentes informativos
            message = ACLMessage(
                performative=Performative.QUERY,
                sender=str(turista.unique_id),
                receiver=str(agente.unique_id),
                content=f"¿Qué me puedes contar sobre {nodo.nombre}?",
                protocol="tourism-information",
                context={
                    'location': nodo.nombre,
                    'tourist_satisfaction': turista.satisfaccion,
                    'information_type': 'general'
                }
            )
        elif agente.rol in ['mesero', 'vendedor']:
            # Consultar servicios
            message = create_request_message(
                sender=str(turista.unique_id),
                receiver=str(agente.unique_id),
                content=f"¿Qué servicios ofrecen en {nodo.nombre}?",
                context={
                    'location': nodo.nombre,
                    'tourist_satisfaction': turista.satisfaccion,
                    'service_type': 'general'
                }
            )
        else:
            # Saludo general
            message = create_inform_message(
                sender=str(turista.unique_id),
                receiver=str(agente.unique_id),
                information=f"Hola, estoy visitando {nodo.nombre}",
                context={
                    'location': nodo.nombre,
                    'tourist_satisfaction': turista.satisfaccion,
                    'interaction_type': 'greeting'
                }
            )
        
        # Enviar mensaje usando el sistema de mensajería del turista
        success = turista.send_message(message)
        print(f"DEBUG - Mensaje turista enviado: {success}, tipo: {message.performative.value}")
        
        return message.conversation_id
    
    @staticmethod
    def _process_agent_response(agente: AgenteBDIBase, turista: TuristaBDI, nodo: Nodo, conversation_id: str) -> bool:
        """Procesa la respuesta del agente usando su sistema BDI con mensajería"""
        try:
            # Forzar procesamiento de mensajes pendientes en el agente
            agente._process_incoming_messages()
            
            # Verificar si hay mensajes para procesar
            if agente.message_queue.size() > 0:
                # El agente procesará el mensaje en su próximo ciclo BDI
                # Simular un paso del agente para procesar el mensaje
                agente.step()
                return True
            else:
                print(f"DEBUG - No hay mensajes pendientes para {agente.rol}")
                return False
                
        except Exception as e:
            print(f"ERROR - Procesando respuesta del agente: {e}")
            return False
    
    @staticmethod
    def _calculate_interaction_impact(agente: AgenteBDIBase, nodo: Nodo) -> float:
        """Calcula el impacto de la interacción usando sistema difuso"""
        # Obtener amabilidad base del agente
        base_amabilidad = SimuladorInteracciones._get_base_amabilidad(agente.rol, nodo.tipo)
        
        # Añadir variabilidad basada en el estado del agente
        agent_satisfaction = getattr(agente, 'satisfaction', 5.0)
        agent_energy = getattr(agente, 'energy', 5.0)
        
        # Ajustar amabilidad según estado del agente
        amabilidad_ajustada = base_amabilidad + (agent_satisfaction - 5) * 0.2 + (agent_energy - 5) * 0.1
        amabilidad_final = max(1, min(10, amabilidad_ajustada + random.uniform(-1.0, 1.0)))
        
        # Usar sistema difuso para calcular impacto
        impacto = sistema_difuso.calcular_impacto(
            amabilidad_valor=amabilidad_final,
            satisfaccion_actual=5.0,  # Valor neutral para el cálculo
            lugar_tipo=nodo.tipo
        )
        
        return impacto
    
    @staticmethod
    def _create_experience_from_messaging(agente: AgenteBDIBase, nodo: Nodo, impacto: float) -> str:
        """Crea experiencia basada en la interacción de mensajería"""
        # Obtener estadísticas de comunicación del agente
        comm_stats = agente.get_communication_stats()
        
        # Crear experiencia contextualizada
        if impacto > 0.5:
            experiencias_positivas = [
                f"Tuviste una excelente interacción con el {agente.rol} en {nodo.nombre} vía mensajería FIPA-ACL",
                f"El {agente.rol} te proporcionó información muy útil sobre {nodo.nombre} usando comunicación estructurada",
                f"La comunicación con el {agente.rol} en {nodo.nombre} fue muy satisfactoria y bien organizada"
            ]
            base_experience = random.choice(experiencias_positivas)
        elif impacto < -0.5:
            experiencias_negativas = [
                f"La comunicación con el {agente.rol} en {nodo.nombre} no fue muy efectiva",
                f"Hubo dificultades en la interacción con el {agente.rol} en {nodo.nombre}",
                f"La respuesta del {agente.rol} en {nodo.nombre} no fue muy útil"
            ]
            base_experience = random.choice(experiencias_negativas)
        else:
            experiencias_neutrales = [
                f"Tuviste una interacción estándar con el {agente.rol} en {nodo.nombre}",
                f"El {agente.rol} te atendió de manera profesional en {nodo.nombre}",
                f"La comunicación con el {agente.rol} en {nodo.nombre} fue correcta"
            ]
            base_experience = random.choice(experiencias_neutrales)
        
        # Añadir detalles del protocolo si la comunicación fue exitosa
        if comm_stats['successful_interactions'] > 0:
            base_experience += " (comunicación FIPA-ACL exitosa)"
        
        return base_experience

    @staticmethod
    def _get_base_amabilidad(rol: str, tipo_lugar: str) -> float:
        """
        Calcula la amabilidad base esperada según el rol del agente y tipo de lugar.
        """
        # Amabilidad base por rol
        amabilidad_por_rol = {
            'guía': 8.0,
            'mesero': 7.5,
            'vendedor': 6.5,
            'curador': 7.0,
            'chef': 6.0,
            'jardinero': 7.5,
            'historiador': 7.8,
            'sacerdote': 8.5,
            'asistente': 7.0,
            'salvavidas': 8.0,
            'fotógrafo': 6.5
        }
        
        # Modificadores por tipo de lugar
        modificadores_lugar = {
            'museo': 0.2,      # Ambiente más formal, ligeramente menos amigable
            'restaurante': 0.5, # Servicio al cliente importante
            'parque': 0.3,     # Ambiente relajado
            'monumento': 0.0,  # Neutral
            'iglesia': 0.4,    # Ambiente acogedor
            'mercado': -0.2,   # Puede ser más comercial/agresivo
            'tienda': -0.1,    # Enfoque en ventas
            'playa': 0.6,      # Ambiente muy relajado
            'mirador': 0.2,    # Ambiente positivo
            'atraccion': 0.1   # Neutral-positivo
        }
        
        base = amabilidad_por_rol.get(rol, 7.0)
        modificador = modificadores_lugar.get(tipo_lugar, 0.0)
        
        return max(3.0, min(9.5, base + modificador))

class ModeloTurismo(Model):
    """
    Modelo principal de la simulación de turismo con agentes BDI y protocolo FIPA-ACL.
    """
    def __init__(self, lista_nodos: List[Dict], nombre_turista: str = "Turista"):
        super().__init__()
        self.grid = MultiGrid(10, 10, torus=True)
        self.schedule = SimpleScheduler(self)
        self.datacollector = DataCollector(
            agent_reporters={"Satisfacción": lambda a: getattr(a, 'satisfaccion', None)}
        )

        # Inicializar sistema de mensajería FIPA-ACL
        self.dispatcher, self.conversation_manager, self.protocol_handler = get_messaging_system()
        print("DEBUG - Sistema de mensajería FIPA-ACL inicializado")

        # Convert input to Nodo objects if they aren't already
        self.nodos = []
        for nodo_data in lista_nodos:
            if isinstance(nodo_data, Nodo):
                self.nodos.append(nodo_data)
            else:
                # Create a new dictionary with default values for missing fields
                nodo_dict = {
                    'id': nodo_data.get('id', ''),
                    'nombre': nodo_data.get('nombre', ''),
                    'tipo': nodo_data.get('tipo', ''),
                    'descripcion': nodo_data.get('descripcion', ''),
                    'agentes': nodo_data.get('agentes', [])
                }
                self.nodos.append(Nodo(**nodo_dict))

        print(f"DEBUG - Creados {len(self.nodos)} nodos")

        # Initialize the tourist with proper parameters and messaging capabilities
        try:
            print(f"DEBUG - Intentando crear TuristaBDI con mensajería FIPA-ACL: {nombre_turista}")
            self.turista = TuristaBDI(unique_id=0, model=self, nombre=nombre_turista)
            print(f"DEBUG - TuristaBDI con mensajería creado exitosamente")
            self.schedule.add(self.turista)
            print(f"DEBUG - Turista BDI con mensajería agregado al schedule")
        except Exception as e:
            print(f"ERROR - Fallo al crear TuristaBDI con mensajería: {str(e)}")
            print(f"ERROR - Tipo de error: {type(e)}")
            raise e

        agentes_creados = 0
        for nodo in self.nodos:
            print(f"DEBUG - Procesando nodo {nodo.nombre} con agentes: {nodo.agentes}")
            for rol in nodo.agentes:
                try:
                    print(f"DEBUG - Intentando crear agente {rol} con mensajería FIPA-ACL en {nodo.nombre}")
                    agente = GeneradorAgentes.crear_agente(rol=rol, nodo=nodo, model=self)
                    self.schedule.add(agente)
                    agentes_creados += 1
                    print(f"DEBUG - Agente BDI con mensajería creado: {agente.rol} en {nodo.nombre} (ID: {agente.unique_id})")
                except Exception as e:
                    print(f"ERROR - Fallo al crear agente {rol} con mensajería en {nodo.nombre}: {str(e)}")
                    print(f"ERROR - Tipo de error: {type(e)}")
                    raise e

        print(f"DEBUG - Total agentes con mensajería creados: {agentes_creados}")
        print(f"DEBUG - Total agentes en schedule: {len(self.schedule.agents)}")
        
        # Configurar sistema difuso para el modelo
        self.sistema_difuso = sistema_difuso

    def step(self):
        """
        Ejecuta un paso de simulación con protocolo FIPA-ACL integrado.
        """
        print(f"DEBUG - Iniciando paso FIPA-ACL, satisfacción actual: {self.turista.satisfaccion:.2f}")
        print(f"DEBUG - Total agentes en schedule: {len(self.schedule.agents)}")
        
        # Seleccionar un nodo aleatorio para visitar en este paso
        if self.nodos:
            nodo_actual = random.choice(self.nodos)
            agentes_en_nodo = [a for a in self.schedule.agents if hasattr(a, 'lugar_id') and a.lugar_id == nodo_actual.id]
            
            print(f"DEBUG - Visitando {nodo_actual.nombre}: {len(agentes_en_nodo)} agentes encontrados")
            
            if agentes_en_nodo:
                # Seleccionar agente para interactuar usando protocolo FIPA-ACL
                agente = random.choice(agentes_en_nodo)
                print(f"DEBUG - Interacción FIPA-ACL en {nodo_actual.nombre} con {agente.rol}")
                
                # Realizar interacción usando el simulador con protocolo FIPA-ACL
                SimuladorInteracciones.interactuar(
                    turista=self.turista,
                    agente=agente,
                    nodo=nodo_actual,
                    max_interacciones=random.randint(1, 2)  # 1-2 interacciones por paso
                )
            else:
                print(f"DEBUG - No hay agentes en {nodo_actual.nombre}")
                # Experiencia sin agente (autoexploración)
                self._experiencia_autoexploracion(nodo_actual)
        
        # Ejecutar step de agentes BDI con capacidades de mensajería
        self.schedule.step()
        
        # Procesar mensajes pendientes en el sistema
        self._process_system_messages()
        
        print(f"DEBUG - Fin del paso FIPA-ACL, satisfacción final: {self.turista.satisfaccion:.2f}")
        self.datacollector.collect(self)
    
    def _process_system_messages(self):
        """Procesa mensajes pendientes en el sistema de mensajería"""
        try:
            # Obtener estadísticas del sistema de mensajería
            stats = self.dispatcher.get_stats()
            if stats['messages_sent'] > 0:
                print(f"DEBUG - Mensajes FIPA-ACL procesados: {stats['messages_sent']} enviados, {stats['messages_delivered']} entregados")
            
            # Limpiar conversaciones antiguas
            self.conversation_manager.cleanup_old_conversations(max_age_hours=1)
            
        except Exception as e:
            print(f"ERROR - Procesando mensajes del sistema: {e}")

    def _experiencia_autoexploracion(self, nodo):
        """
        Simula una experiencia de autoexploración cuando no hay agentes disponibles.
        """
        experiencias_autoexploracion = [
            f"Exploras {nodo.nombre} por tu cuenta, disfrutando del ambiente.",
            f"Te tomas tu tiempo para apreciar los detalles de {nodo.nombre}.",
            f"Caminas tranquilamente por {nodo.nombre}, observando todo a tu ritmo.",
            f"Disfrutas de un momento de paz en {nodo.nombre}.",
            f"Te sientes libre explorando {nodo.nombre} sin prisa."
        ]
        
        experiencia = random.choice(experiencias_autoexploracion)
        
        # Impacto más moderado para autoexploración
        impacto_base = random.uniform(-0.3, 0.8)
        
        # Ajustar según tipo de lugar
        if nodo.tipo in ['parque', 'playa', 'mirador']:
            impacto_base += 0.3  # Lugares naturales son mejores para autoexploración
        elif nodo.tipo in ['museo', 'monumento']:
            impacto_base -= 0.2  # Lugares que se benefician de guías
        
        impacto_final = max(-1.0, min(1.5, impacto_base))
        
        self.turista.agregar_experiencia(experiencia, impacto_final)
        print(f"DEBUG - Autoexploración en {nodo.nombre}: impacto={impacto_final:.2f}")

def ejecutar_simulaciones(n_simulaciones: int, pasos: int = 10):
    """
    Ejecuta varias simulaciones con protocolo FIPA-ACL integrado y muestra la satisfacción y recuerdos del turista.
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
        },
        {
            "id": "parque_1",
            "nombre": "Parque Central",
            "tipo": "parque",
            "descripcion": "Hermoso parque con jardines y fuentes.",
            "agentes": ["jardinero", "guía"]
        }
    ]
    satisfacciones = []
    estadisticas_comunicacion = []
    
    for i in range(n_simulaciones):
        print(f"\n=== INICIANDO SIMULACIÓN FIPA-ACL {i+1} ===")
        modelo = ModeloTurismo(lista_nodos, nombre_turista=f"Ana_{i+1}")
        print(f"Satisfacción inicial: {modelo.turista.satisfaccion}")
        print(f"Sistema de mensajería FIPA-ACL: {len(modelo.dispatcher.agent_queues)} agentes registrados")
        
        for paso in range(pasos):
            print(f"\n--- Paso FIPA-ACL {paso+1} ---")
            modelo.step()
            print(f"Satisfacción después del paso {paso+1}: {modelo.turista.satisfaccion:.2f}")
        
        satisfaccion_final = modelo.turista.satisfaccion
        satisfacciones.append(satisfaccion_final)
        
        # Obtener estadísticas de comunicación
        comm_stats = modelo.dispatcher.get_stats()
        tourist_comm_stats = modelo.turista.get_communication_stats()
        estadisticas_comunicacion.append({
            'sistema': comm_stats,
            'turista': tourist_comm_stats
        })
        
        print(f"\nSimulación FIPA-ACL {i+1}: Satisfacción final = {satisfaccion_final:.1f}/10")
        print(f"Estadísticas de comunicación:")
        print(f"  - Mensajes del sistema: {comm_stats['messages_sent']} enviados, {comm_stats['messages_delivered']} entregados")
        print(f"  - Mensajes del turista: {tourist_comm_stats['messages_sent']} enviados, {tourist_comm_stats['messages_received']} recibidos")
        print(f"  - Conversaciones iniciadas: {tourist_comm_stats['conversations_initiated']}")
        print(f"  - Interacciones exitosas: {tourist_comm_stats['successful_interactions']}")
        
        # Debug memory contents
        print(f"Memorias totales: Alta={len(modelo.turista.memoria_alta)}, Media={len(modelo.turista.memoria_media)}, Baja={len(modelo.turista.memoria_baja)}")
        
        recuerdos = modelo.turista.recuerdos_significativos()
        print("Recuerdos más significativos:")
        if recuerdos:
            for rec in recuerdos:
                print(f"- {rec}")
        else:
            print("- No hay recuerdos significativos")
        
        # Mostrar estadísticas de agentes
        print("\nEstadísticas de agentes:")
        for agent in modelo.schedule.agents:
            if hasattr(agent, 'rol') and agent.rol != 'turista':
                agent_stats = agent.get_communication_stats()
                print(f"  - {agent.rol}: {agent_stats['messages_sent']} enviados, {agent_stats['successful_interactions']} exitosas")
        
        print("=" * 60)
    
    # Estadísticas finales
    promedio = sum(satisfacciones) / len(satisfacciones)
    total_mensajes = sum(stats['sistema']['messages_sent'] for stats in estadisticas_comunicacion)
    total_conversaciones = sum(stats['turista']['conversations_initiated'] for stats in estadisticas_comunicacion)
    
    print(f"\n=== RESUMEN SIMULACIONES FIPA-ACL ===")
    print(f"Promedio de satisfacción tras {n_simulaciones} simulaciones: {promedio:.2f}/10")
    print(f"Total mensajes FIPA-ACL intercambiados: {total_mensajes}")
    print(f"Total conversaciones iniciadas: {total_conversaciones}")
    print(f"Promedio mensajes por simulación: {total_mensajes/n_simulaciones:.1f}")
    
    print("\nPrueba sistema difuso:")
    print("Impacto esperado positivo:", sistema_difuso.calcular_impacto(8.0, 3.0))
    print("Impacto esperado negativo:", sistema_difuso.calcular_impacto(2.0, 8.0))
    
    # Limpiar sistema de mensajería al final
    cleanup_messaging_system()
    print("Sistema de mensajería FIPA-ACL limpiado")
    
    return satisfacciones, promedio, estadisticas_comunicacion

if __name__ == "__main__":
    n_simulaciones = 3
    ejecutar_simulaciones(n_simulaciones, pasos=5)