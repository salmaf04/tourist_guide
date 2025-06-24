"""
Enhanced BDI Agents with FIPA-ACL Messaging Integration
Agentes BDI mejorados con integración del protocolo de mensajería FIPA-ACL
"""

import random
import time
from typing import Dict, List, Optional, Any
from .bdi import AgenteBDI, Desire, Intention, StrategyType
from .fipa_acl import (
    ACLMessage, Performative, MessageQueue, 
    get_messaging_system, create_request_message, 
    create_recommendation_message, create_inform_message
)
from .mistral_client import MistralClient


class MessagingBDIAgent(AgenteBDI):
    """
    Agente BDI con capacidades de mensajería FIPA-ACL integradas
    Combina la arquitectura BDI con comunicación estructurada
    """
    
    def __init__(self, unique_id, model, estrategia="equilibrada", agent_type="generic"):
        super().__init__(unique_id, model, estrategia)
        
        # Configuración de mensajería
        self.agent_type = agent_type
        self.dispatcher, self.conversation_manager, self.protocol_handler = get_messaging_system()
        self.message_queue = self.dispatcher.register_agent(str(unique_id))
        
        # Cache de respuestas para evitar uso excesivo del LLM
        self.response_cache = {}
        self.llm_usage_limit = 3  # Máximo 3 llamadas LLM por step
        self.llm_usage_count = 0
        
        # Configurar manejadores de mensajes
        self._setup_message_handlers()
        
        # Contexto de comunicación
        self.active_conversations = {}
        self.communication_preferences = self._get_communication_preferences()
        
        # Estadísticas de comunicación
        self.comm_stats = {
            'messages_sent': 0,
            'messages_received': 0,
            'conversations_initiated': 0,
            'successful_interactions': 0
        }
    
    def _setup_message_handlers(self):
        """Configura manejadores para diferentes tipos de mensajes"""
        self.message_queue.register_handler(Performative.REQUEST, self._handle_request)
        self.message_queue.register_handler(Performative.QUERY, self._handle_query)
        self.message_queue.register_handler(Performative.RECOMMEND, self._handle_recommendation)
        self.message_queue.register_handler(Performative.INFORM, self._handle_inform)
        self.message_queue.register_handler(Performative.ACCEPT, self._handle_accept)
        self.message_queue.register_handler(Performative.REJECT, self._handle_reject)
    
    def _get_communication_preferences(self) -> Dict[str, Any]:
        """Obtiene preferencias de comunicación según el tipo de agente"""
        preferences = {
            'guía': {
                'proactive': True,
                'helpful': 0.9,
                'informative': 0.8,
                'friendly': 0.8,
                'response_time': 'fast'
            },
            'mesero': {
                'proactive': True,
                'helpful': 0.8,
                'informative': 0.6,
                'friendly': 0.9,
                'response_time': 'fast'
            },
            'vendedor': {
                'proactive': True,
                'helpful': 0.7,
                'informative': 0.7,
                'friendly': 0.7,
                'response_time': 'medium'
            },
            'turista': {
                'proactive': False,
                'helpful': 0.5,
                'informative': 0.4,
                'friendly': 0.6,
                'response_time': 'medium'
            },
            'curador': {
                'proactive': False,
                'helpful': 0.8,
                'informative': 0.9,
                'friendly': 0.7,
                'response_time': 'slow'
            }
        }
        return preferences.get(self.agent_type, preferences['generic'] if 'generic' in preferences else {
            'proactive': False,
            'helpful': 0.6,
            'informative': 0.6,
            'friendly': 0.6,
            'response_time': 'medium'
        })
    
    def percibir(self):
        """Percepción mejorada que incluye procesamiento de mensajes"""
        # Percepción base del BDI
        super().percibir()
        
        # Procesar mensajes entrantes
        self._process_incoming_messages()
        
        # Actualizar creencias con información de comunicación
        self.beliefs.update({
            'pending_messages': self.message_queue.size(),
            'active_conversations': len(self.active_conversations),
            'communication_load': self._calculate_communication_load()
        })
    
    def deliberar(self):
        """Deliberación que considera necesidades de comunicación"""
        # Deliberación base del BDI
        super().deliberar()
        
        # Añadir deseos relacionados con comunicación
        communication_desires = self._generate_communication_desires()
        self.desires.extend(communication_desires)
        
        # Priorizar deseos según contexto de comunicación
        self._prioritize_desires_by_communication()
    
    def actuar(self):
        """Ejecución que incluye acciones de comunicación"""
        # Ejecutar acciones base del BDI
        super().actuar()
        
        # Ejecutar acciones de comunicación pendientes
        self._execute_communication_actions()
        
        # Limpiar conversaciones inactivas
        self._cleanup_inactive_conversations()
    
    def _process_incoming_messages(self):
        """Procesa mensajes entrantes de la cola"""
        processed_count = 0
        max_messages_per_step = 5  # Limitar procesamiento por step
        
        while processed_count < max_messages_per_step:
            message = self.message_queue.get_by_priority()
            if not message:
                break
            
            self.comm_stats['messages_received'] += 1
            
            # Obtener manejadores para este tipo de mensaje
            handlers = self.message_queue.get_handlers(message.performative)
            
            # Ejecutar manejadores
            for handler in handlers:
                try:
                    handler(message)
                except Exception as e:
                    print(f"Error procesando mensaje {message.performative}: {e}")
            
            # Actualizar conversación si existe
            if message.conversation_id:
                self.conversation_manager.update_conversation(message.conversation_id, message)
                self.active_conversations[message.conversation_id] = time.time()
            
            processed_count += 1
    
    def _generate_communication_desires(self) -> List[Desire]:
        """Genera deseos relacionados con comunicación"""
        desires = []
        
        # Deseo de responder mensajes pendientes
        if self.message_queue.size() > 0:
            priority = 0.8 if self.communication_preferences.get('helpful', 0.5) > 0.7 else 0.5
            desires.append(Desire(
                name="responder_mensajes",
                priority=priority,
                context={'pending_count': self.message_queue.size()}
            ))
        
        # Deseo de iniciar comunicación (para agentes proactivos)
        if (self.communication_preferences.get('proactive', False) and 
            len(self.active_conversations) < 2 and random.random() < 0.3):
            desires.append(Desire(
                name="iniciar_comunicacion",
                priority=0.6,
                context={'type': 'proactive'}
            ))
        
        # Deseo de compartir información (para agentes informativos)
        if (self.communication_preferences.get('informative', 0.5) > 0.7 and
            random.random() < 0.2):
            desires.append(Desire(
                name="compartir_informacion",
                priority=0.5,
                context={'type': 'informative'}
            ))
        
        return desires
    
    def _prioritize_desires_by_communication(self):
        """Ajusta prioridades de deseos según contexto de comunicación"""
        comm_load = self._calculate_communication_load()
        
        for desire in self.desires:
            # Reducir prioridad de deseos no comunicativos si hay alta carga
            if comm_load > 0.7 and desire.name not in ['responder_mensajes', 'iniciar_comunicacion']:
                desire.priority *= 0.8
            
            # Aumentar prioridad de respuestas si el agente es muy servicial
            elif (desire.name == 'responder_mensajes' and 
                  self.communication_preferences.get('helpful', 0.5) > 0.8):
                desire.priority *= 1.2
    
    def _execute_communication_actions(self):
        """Ejecuta acciones de comunicación basadas en intenciones"""
        for intention in self.intentions:
            if intention.name == "responder_mensajes":
                self._respond_to_pending_messages()
            elif intention.name == "iniciar_comunicacion":
                self._initiate_communication()
            elif intention.name == "compartir_informacion":
                self._share_information()
    
    def _respond_to_pending_messages(self):
        """Responde a mensajes pendientes usando protocolo y LLM selectivamente"""
        message = self.message_queue.peek()
        if not message:
            return
        
        # Usar respuesta en cache si está disponible
        cache_key = f"{message.performative.value}_{hash(message.content)}"
        if cache_key in self.response_cache:
            response_content = self.response_cache[cache_key]
        else:
            # Decidir si usar LLM o respuesta predefinida
            if self._should_use_llm_for_response(message):
                response_content = self._generate_llm_response(message)
                self.response_cache[cache_key] = response_content
            else:
                response_content = self._generate_rule_based_response(message)
        
        # Crear y enviar respuesta
        if response_content:
            response = message.create_reply(
                performative=Performative.INFORM,
                content=response_content,
                sender=str(self.unique_id),
                context={'agent_type': self.agent_type}
            )
            
            self.send_message(response)
    
    def _should_use_llm_for_response(self, message: ACLMessage) -> bool:
        """Decide si usar LLM para generar respuesta"""
        # No usar LLM si se ha alcanzado el límite
        if self.llm_usage_count >= self.llm_usage_limit:
            return False
        
        # Usar LLM para mensajes complejos o de alta prioridad
        complex_performatives = [Performative.REQUEST, Performative.QUERY, Performative.RECOMMEND]
        if message.performative in complex_performatives:
            return True
        
        # Usar LLM si el contenido es largo o complejo
        if len(message.content) > 50 or '?' in message.content:
            return True
        
        return False
    
    def _generate_llm_response(self, message: ACLMessage) -> str:
        """Genera respuesta usando LLM con contexto del agente"""
        if self.llm_usage_count >= self.llm_usage_limit:
            return self._generate_rule_based_response(message)
        
        try:
            # Construir prompt contextualizado
            context = self._build_llm_context(message)
            prompt = f"""
            Eres un {self.agent_type} en un sistema turístico.
            Contexto: {context}
            
            Mensaje recibido: "{message.content}"
            Tipo de mensaje: {message.performative.value}
            
            Responde de manera {self.communication_preferences.get('friendly', 0.6) > 0.7 and 'amigable' or 'profesional'} 
            y {'muy informativa' if self.communication_preferences.get('informative', 0.5) > 0.7 else 'concisa'}.
            
            Respuesta (máximo 2 oraciones):
            """
            
            response = self.llm.generate(prompt)
            self.llm_usage_count += 1
            
            if response and response != "[Respuesta no disponible]":
                return response
            else:
                return self._generate_rule_based_response(message)
                
        except Exception as e:
            print(f"Error generando respuesta LLM: {e}")
            return self._generate_rule_based_response(message)
    
    def _generate_rule_based_response(self, message: ACLMessage) -> str:
        """Genera respuesta basada en reglas sin usar LLM"""
        responses_by_type = {
            'guía': {
                Performative.REQUEST: [
                    "Con gusto te ayudo con esa información.",
                    "Déjame explicarte sobre eso.",
                    "Es un placer compartir mi conocimiento contigo."
                ],
                Performative.QUERY: [
                    "Según mi experiencia, puedo decirte que...",
                    "Esa es una excelente pregunta.",
                    "Te puedo proporcionar esa información."
                ],
                Performative.RECOMMEND: [
                    "Gracias por la recomendación, la tendré en cuenta.",
                    "Interesante sugerencia, me parece útil.",
                    "Aprecio tu recomendación."
                ]
            },
            'mesero': {
                Performative.REQUEST: [
                    "¡Por supuesto! Enseguida te atiendo.",
                    "Con mucho gusto, ¿qué necesitas?",
                    "Estoy aquí para servirte."
                ],
                Performative.QUERY: [
                    "Te puedo recomendar nuestras especialidades.",
                    "Déjame consultarte sobre nuestro menú.",
                    "¿Te gustaría conocer nuestros platos del día?"
                ]
            },
            'turista': {
                Performative.INFORM: [
                    "Gracias por la información.",
                    "Muy útil, lo tendré en cuenta.",
                    "Aprecio tu ayuda."
                ],
                Performative.RECOMMEND: [
                    "¡Suena interesante! Lo consideraré.",
                    "Gracias por la recomendación.",
                    "Me parece una buena opción."
                ]
            }
        }
        
        agent_responses = responses_by_type.get(self.agent_type, {})
        performative_responses = agent_responses.get(message.performative, [
            "Gracias por tu mensaje.",
            "He recibido tu comunicación.",
            "Entiendo tu solicitud."
        ])
        
        return random.choice(performative_responses)
    
    def _build_llm_context(self, message: ACLMessage) -> str:
        """Construye contexto para el LLM"""
        context_parts = []
        
        # Información del agente
        context_parts.append(f"Soy un {self.agent_type}")
        
        # Estado actual
        if hasattr(self, 'location'):
            context_parts.append(f"ubicado en {self.location}")
        
        # Información de la conversación
        if message.conversation_id and message.conversation_id in self.active_conversations:
            context_parts.append("en una conversación activa")
        
        # Contexto del mensaje
        if message.context:
            relevant_context = {k: v for k, v in message.context.items() 
                              if k in ['location', 'urgency', 'topic']}
            if relevant_context:
                context_parts.append(f"contexto adicional: {relevant_context}")
        
        return ". ".join(context_parts)
    
    def _initiate_communication(self):
        """Inicia comunicación proactiva con otros agentes"""
        if not self.communication_preferences.get('proactive', False):
            return
        
        # Buscar agentes cercanos para comunicarse
        nearby_agents = self._get_nearby_agents()
        if not nearby_agents:
            return
        
        target_agent = random.choice(nearby_agents)
        
        # Crear mensaje apropiado según el tipo de agente
        if self.agent_type == 'guía':
            message = create_inform_message(
                sender=str(self.unique_id),
                receiver=str(target_agent.unique_id),
                information="¿Necesitas información sobre este lugar?",
                context={'type': 'proactive_help', 'agent_type': self.agent_type}
            )
        elif self.agent_type == 'mesero':
            message = create_request_message(
                sender=str(self.unique_id),
                receiver=str(target_agent.unique_id),
                content="¿Te gustaría ver nuestro menú?",
                context={'type': 'service_offer', 'agent_type': self.agent_type}
            )
        else:
            message = create_inform_message(
                sender=str(self.unique_id),
                receiver=str(target_agent.unique_id),
                information="¡Hola! ¿Cómo va tu visita?",
                context={'type': 'friendly_greeting', 'agent_type': self.agent_type}
            )
        
        self.send_message(message)
        self.comm_stats['conversations_initiated'] += 1
    
    def _share_information(self):
        """Comparte información relevante con otros agentes"""
        if not self.communication_preferences.get('informative', 0.5) > 0.6:
            return
        
        # Generar información para compartir según el tipo de agente
        info_to_share = self._generate_shareable_information()
        if not info_to_share:
            return
        
        # Crear mensaje de información
        message = create_inform_message(
            sender=str(self.unique_id),
            receiver="broadcast",  # Enviar a todos
            information=info_to_share,
            context={'type': 'information_sharing', 'agent_type': self.agent_type}
        )
        
        # Enviar como broadcast
        self.dispatcher.broadcast_message(message, exclude=[str(self.unique_id)])
    
    def _generate_shareable_information(self) -> Optional[str]:
        """Genera información para compartir según el tipo de agente"""
        info_templates = {
            'guía': [
                "Dato curioso: Este lugar tiene una historia fascinante.",
                "Recomendación: El mejor momento para visitar es por la mañana.",
                "Información útil: Hay servicios disponibles cerca."
            ],
            'mesero': [
                "Oferta especial: Tenemos platos del día disponibles.",
                "Recomendación culinaria: Nuestras especialidades locales son muy populares.",
                "Información: El restaurante tiene opciones para diferentes dietas."
            ],
            'curador': [
                "Información cultural: Esta exhibición tiene un significado especial.",
                "Dato educativo: La historia detrás de estas piezas es muy interesante.",
                "Recomendación: Vale la pena dedicar tiempo a observar los detalles."
            ]
        }
        
        templates = info_templates.get(self.agent_type, [])
        return random.choice(templates) if templates else None
    
    def send_message(self, message: ACLMessage) -> bool:
        """Envía mensaje usando el dispatcher"""
        success = self.dispatcher.send_message(message)
        if success:
            self.comm_stats['messages_sent'] += 1
            if message.conversation_id:
                self.active_conversations[message.conversation_id] = time.time()
        return success
    
    def _calculate_communication_load(self) -> float:
        """Calcula la carga de comunicación actual"""
        pending_messages = self.message_queue.size()
        active_conversations = len(self.active_conversations)
        
        # Normalizar entre 0 y 1
        load = (pending_messages * 0.3 + active_conversations * 0.7) / 10
        return min(1.0, load)
    
    def _cleanup_inactive_conversations(self):
        """Limpia conversaciones inactivas"""
        current_time = time.time()
        timeout = 300  # 5 minutos
        
        inactive_conversations = [
            conv_id for conv_id, last_activity in self.active_conversations.items()
            if current_time - last_activity > timeout
        ]
        
        for conv_id in inactive_conversations:
            del self.active_conversations[conv_id]
            self.conversation_manager.end_conversation(conv_id)
    
    def step(self):
        """Ciclo BDI mejorado con comunicación integrada"""
        # Reset contador de uso de LLM
        self.llm_usage_count = 0
        
        # Ejecutar ciclo BDI base con comunicación integrada
        super().step()
    
    # Manejadores de mensajes específicos
    def _handle_request(self, message: ACLMessage):
        """Maneja mensajes de solicitud"""
        # La respuesta se manejará en _respond_to_pending_messages
        pass
    
    def _handle_query(self, message: ACLMessage):
        """Maneja mensajes de consulta"""
        # La respuesta se manejará en _respond_to_pending_messages
        pass
    
    def _handle_recommendation(self, message: ACLMessage):
        """Maneja mensajes de recomendación"""
        # Procesar recomendación y posiblemente responder
        if random.random() < self.communication_preferences.get('helpful', 0.5):
            # Aceptar o agradecer la recomendación
            response = message.create_reply(
                performative=Performative.ACCEPT,
                content="Gracias por la recomendación, la tendré en cuenta.",
                sender=str(self.unique_id)
            )
            self.send_message(response)
    
    def _handle_inform(self, message: ACLMessage):
        """Maneja mensajes informativos"""
        # Procesar información y posiblemente responder
        if message.reply_to:  # Es respuesta a algo que enviamos
            self.comm_stats['successful_interactions'] += 1
    
    def _handle_accept(self, message: ACLMessage):
        """Maneja mensajes de aceptación"""
        self.comm_stats['successful_interactions'] += 1
    
    def _handle_reject(self, message: ACLMessage):
        """Maneja mensajes de rechazo"""
        # Podríamos ajustar estrategia futura basada en rechazos
        pass
    
    def get_communication_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de comunicación del agente"""
        return {
            **self.comm_stats,
            'pending_messages': self.message_queue.size(),
            'active_conversations': len(self.active_conversations),
            'communication_load': self._calculate_communication_load(),
            'llm_usage_efficiency': self.llm_usage_count / max(1, self.comm_stats['messages_received'])
        }