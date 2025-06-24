"""
FIPA-ACL Simplified Protocol Implementation
Protocolo de Mensajería Asíncrona simplificado para agentes turísticos
"""

import json
import time
import uuid
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Callable
from collections import deque
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor


class Performative(Enum):
    """Performativas FIPA-ACL simplificadas para turismo"""
    # Básicas
    REQUEST = "request"           # Solicitar información/servicio
    INFORM = "inform"            # Informar/responder
    QUERY = "query"              # Consultar información específica
    ACCEPT = "accept"            # Aceptar propuesta/solicitud
    REJECT = "reject"            # Rechazar propuesta/solicitud
    CONFIRM = "confirm"          # Confirmar información
    
    # Específicas de turismo
    RECOMMEND = "recommend"      # Recomendar lugares/actividades
    BOOK = "book"               # Reservar servicio
    CANCEL = "cancel"           # Cancelar reserva
    GUIDE = "guide"             # Proporcionar guía/direcciones
    WARN = "warn"               # Advertir sobre algo
    INVITE = "invite"           # Invitar a actividad


@dataclass
class ACLMessage:
    """Mensaje ACL simplificado para comunicación entre agentes"""
    performative: Performative
    sender: str
    receiver: str
    content: str
    conversation_id: Optional[str] = None
    reply_to: Optional[str] = None
    language: str = "es"
    ontology: str = "tourism"
    protocol: str = "fipa-request"
    
    # Metadatos adicionales
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    priority: int = 5  # 1-10, donde 10 es máxima prioridad
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_json(self) -> str:
        """Convierte el mensaje a JSON"""
        data = asdict(self)
        data['performative'] = self.performative.value
        return json.dumps(data, ensure_ascii=False)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'ACLMessage':
        """Crea mensaje desde JSON"""
        data = json.loads(json_str)
        data['performative'] = Performative(data['performative'])
        return cls(**data)
    
    def create_reply(self, performative: Performative, content: str, 
                    sender: str, context: Dict[str, Any] = None) -> 'ACLMessage':
        """Crea mensaje de respuesta"""
        return ACLMessage(
            performative=performative,
            sender=sender,
            receiver=self.sender,
            content=content,
            conversation_id=self.conversation_id,
            reply_to=self.message_id,
            language=self.language,
            ontology=self.ontology,
            protocol=self.protocol,
            context=context or {}
        )


class MessageQueue:
    """Cola de mensajes thread-safe para agentes"""
    
    def __init__(self, max_size: int = 100):
        self.queue = deque(maxlen=max_size)
        self.lock = threading.Lock()
        self.handlers: Dict[Performative, List[Callable]] = {}
    
    def put(self, message: ACLMessage) -> bool:
        """Añade mensaje a la cola"""
        with self.lock:
            try:
                self.queue.append(message)
                return True
            except Exception as e:
                print(f"Error añadiendo mensaje a cola: {e}")
                return False
    
    def get(self) -> Optional[ACLMessage]:
        """Obtiene mensaje de la cola (FIFO)"""
        with self.lock:
            try:
                return self.queue.popleft()
            except IndexError:
                return None
    
    def get_by_priority(self) -> Optional[ACLMessage]:
        """Obtiene mensaje con mayor prioridad"""
        with self.lock:
            if not self.queue:
                return None
            
            # Encontrar mensaje con mayor prioridad
            max_priority = max(msg.priority for msg in self.queue)
            for i, msg in enumerate(self.queue):
                if msg.priority == max_priority:
                    del self.queue[i]
                    return msg
            return None
    
    def peek(self) -> Optional[ACLMessage]:
        """Ve el próximo mensaje sin removerlo"""
        with self.lock:
            try:
                return self.queue[0]
            except IndexError:
                return None
    
    def size(self) -> int:
        """Retorna tamaño de la cola"""
        with self.lock:
            return len(self.queue)
    
    def clear(self):
        """Limpia la cola"""
        with self.lock:
            self.queue.clear()
    
    def register_handler(self, performative: Performative, handler: Callable):
        """Registra manejador para tipo de mensaje"""
        if performative not in self.handlers:
            self.handlers[performative] = []
        self.handlers[performative].append(handler)
    
    def get_handlers(self, performative: Performative) -> List[Callable]:
        """Obtiene manejadores para tipo de mensaje"""
        return self.handlers.get(performative, [])


class MessageDispatcher:
    """Despachador central de mensajes para el sistema multi-agente"""
    
    def __init__(self):
        self.agent_queues: Dict[str, MessageQueue] = {}
        self.message_history: List[ACLMessage] = []
        self.max_history = 1000
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Estadísticas
        self.stats = {
            'messages_sent': 0,
            'messages_delivered': 0,
            'messages_failed': 0,
            'messages_by_type': {}
        }
    
    def register_agent(self, agent_id: str, queue_size: int = 100) -> MessageQueue:
        """Registra un agente en el sistema de mensajería"""
        with self.lock:
            if agent_id not in self.agent_queues:
                self.agent_queues[agent_id] = MessageQueue(queue_size)
            return self.agent_queues[agent_id]
    
    def unregister_agent(self, agent_id: str):
        """Desregistra un agente del sistema"""
        with self.lock:
            if agent_id in self.agent_queues:
                del self.agent_queues[agent_id]
    
    def send_message(self, message: ACLMessage) -> bool:
        """Envía mensaje a agente destinatario"""
        try:
            with self.lock:
                self.stats['messages_sent'] += 1
                
                # Actualizar estadísticas por tipo
                perf_str = message.performative.value
                if perf_str not in self.stats['messages_by_type']:
                    self.stats['messages_by_type'][perf_str] = 0
                self.stats['messages_by_type'][perf_str] += 1
                
                # Añadir a historial
                self.message_history.append(message)
                if len(self.message_history) > self.max_history:
                    self.message_history.pop(0)
            
            # Entregar mensaje
            if message.receiver in self.agent_queues:
                success = self.agent_queues[message.receiver].put(message)
                if success:
                    self.stats['messages_delivered'] += 1
                else:
                    self.stats['messages_failed'] += 1
                return success
            else:
                print(f"Agente destinatario {message.receiver} no encontrado")
                self.stats['messages_failed'] += 1
                return False
                
        except Exception as e:
            print(f"Error enviando mensaje: {e}")
            self.stats['messages_failed'] += 1
            return False
    
    def broadcast_message(self, message: ACLMessage, exclude: List[str] = None) -> int:
        """Envía mensaje a todos los agentes registrados"""
        exclude = exclude or []
        sent_count = 0
        
        for agent_id in self.agent_queues.keys():
            if agent_id not in exclude and agent_id != message.sender:
                # Crear copia del mensaje para cada destinatario
                broadcast_msg = ACLMessage(
                    performative=message.performative,
                    sender=message.sender,
                    receiver=agent_id,
                    content=message.content,
                    conversation_id=message.conversation_id,
                    language=message.language,
                    ontology=message.ontology,
                    protocol=message.protocol,
                    context=message.context.copy()
                )
                
                if self.send_message(broadcast_msg):
                    sent_count += 1
        
        return sent_count
    
    def get_agent_queue(self, agent_id: str) -> Optional[MessageQueue]:
        """Obtiene cola de mensajes de un agente"""
        return self.agent_queues.get(agent_id)
    
    def get_conversation_history(self, conversation_id: str) -> List[ACLMessage]:
        """Obtiene historial de una conversación"""
        return [msg for msg in self.message_history 
                if msg.conversation_id == conversation_id]
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del sistema de mensajería"""
        return self.stats.copy()
    
    def cleanup_old_messages(self, max_age_hours: int = 24):
        """Limpia mensajes antiguos del historial"""
        cutoff_time = time.time() - (max_age_hours * 3600)
        with self.lock:
            self.message_history = [
                msg for msg in self.message_history 
                if msg.timestamp > cutoff_time
            ]


class ConversationManager:
    """Gestor de conversaciones para mantener contexto"""
    
    def __init__(self):
        self.conversations: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()
    
    def start_conversation(self, initiator: str, participant: str, 
                          topic: str = "general") -> str:
        """Inicia nueva conversación"""
        conversation_id = str(uuid.uuid4())
        
        with self.lock:
            self.conversations[conversation_id] = {
                'id': conversation_id,
                'initiator': initiator,
                'participant': participant,
                'topic': topic,
                'started_at': time.time(),
                'last_activity': time.time(),
                'message_count': 0,
                'status': 'active',
                'context': {}
            }
        
        return conversation_id
    
    def update_conversation(self, conversation_id: str, message: ACLMessage):
        """Actualiza conversación con nuevo mensaje"""
        with self.lock:
            if conversation_id in self.conversations:
                conv = self.conversations[conversation_id]
                conv['last_activity'] = time.time()
                conv['message_count'] += 1
                
                # Actualizar contexto si el mensaje lo incluye
                if message.context:
                    conv['context'].update(message.context)
    
    def end_conversation(self, conversation_id: str):
        """Termina conversación"""
        with self.lock:
            if conversation_id in self.conversations:
                self.conversations[conversation_id]['status'] = 'ended'
    
    def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene información de conversación"""
        return self.conversations.get(conversation_id)
    
    def cleanup_old_conversations(self, max_age_hours: int = 24):
        """Limpia conversaciones antiguas"""
        cutoff_time = time.time() - (max_age_hours * 3600)
        with self.lock:
            to_remove = [
                conv_id for conv_id, conv in self.conversations.items()
                if conv['last_activity'] < cutoff_time
            ]
            for conv_id in to_remove:
                del self.conversations[conv_id]


class ProtocolHandler:
    """Manejador de protocolos específicos de turismo"""
    
    def __init__(self, dispatcher: MessageDispatcher, 
                 conversation_manager: ConversationManager):
        self.dispatcher = dispatcher
        self.conversation_manager = conversation_manager
        self.protocols = {
            'fipa-request': self._handle_request_protocol,
            'fipa-query': self._handle_query_protocol,
            'tourism-recommendation': self._handle_recommendation_protocol,
            'tourism-booking': self._handle_booking_protocol
        }
    
    def handle_message(self, message: ACLMessage) -> bool:
        """Maneja mensaje según su protocolo"""
        protocol_handler = self.protocols.get(message.protocol)
        if protocol_handler:
            return protocol_handler(message)
        else:
            print(f"Protocolo no soportado: {message.protocol}")
            return False
    
    def _handle_request_protocol(self, message: ACLMessage) -> bool:
        """Maneja protocolo de solicitud básico"""
        if message.performative == Performative.REQUEST:
            # El agente receptor debe responder con INFORM, ACCEPT o REJECT
            return True
        elif message.performative in [Performative.INFORM, Performative.ACCEPT, Performative.REJECT]:
            # Respuesta a solicitud - actualizar conversación
            if message.conversation_id:
                self.conversation_manager.update_conversation(message.conversation_id, message)
            return True
        return False
    
    def _handle_query_protocol(self, message: ACLMessage) -> bool:
        """Maneja protocolo de consulta"""
        if message.performative == Performative.QUERY:
            return True
        elif message.performative == Performative.INFORM:
            if message.conversation_id:
                self.conversation_manager.update_conversation(message.conversation_id, message)
            return True
        return False
    
    def _handle_recommendation_protocol(self, message: ACLMessage) -> bool:
        """Maneja protocolo de recomendaciones turísticas"""
        if message.performative == Performative.RECOMMEND:
            return True
        elif message.performative in [Performative.ACCEPT, Performative.REJECT]:
            if message.conversation_id:
                self.conversation_manager.update_conversation(message.conversation_id, message)
            return True
        return False
    
    def _handle_booking_protocol(self, message: ACLMessage) -> bool:
        """Maneja protocolo de reservas"""
        if message.performative == Performative.BOOK:
            return True
        elif message.performative in [Performative.CONFIRM, Performative.CANCEL]:
            if message.conversation_id:
                self.conversation_manager.update_conversation(message.conversation_id, message)
            return True
        return False


# Funciones de utilidad para crear mensajes comunes
def create_request_message(sender: str, receiver: str, content: str, 
                          context: Dict[str, Any] = None) -> ACLMessage:
    """Crea mensaje de solicitud"""
    return ACLMessage(
        performative=Performative.REQUEST,
        sender=sender,
        receiver=receiver,
        content=content,
        conversation_id=str(uuid.uuid4()),
        context=context or {}
    )


def create_recommendation_message(sender: str, receiver: str, 
                                 recommendations: List[str],
                                 context: Dict[str, Any] = None) -> ACLMessage:
    """Crea mensaje de recomendación"""
    content = "Te recomiendo: " + ", ".join(recommendations)
    return ACLMessage(
        performative=Performative.RECOMMEND,
        sender=sender,
        receiver=receiver,
        content=content,
        protocol="tourism-recommendation",
        context=context or {}
    )


def create_inform_message(sender: str, receiver: str, information: str,
                         reply_to: str = None, conversation_id: str = None,
                         context: Dict[str, Any] = None) -> ACLMessage:
    """Crea mensaje informativo"""
    return ACLMessage(
        performative=Performative.INFORM,
        sender=sender,
        receiver=receiver,
        content=information,
        reply_to=reply_to,
        conversation_id=conversation_id,
        context=context or {}
    )


# Singleton global para el sistema de mensajería
_global_dispatcher = None
_global_conversation_manager = None
_global_protocol_handler = None


def get_messaging_system():
    """Obtiene instancia global del sistema de mensajería"""
    global _global_dispatcher, _global_conversation_manager, _global_protocol_handler
    
    if _global_dispatcher is None:
        _global_dispatcher = MessageDispatcher()
        _global_conversation_manager = ConversationManager()
        _global_protocol_handler = ProtocolHandler(_global_dispatcher, _global_conversation_manager)
    
    return _global_dispatcher, _global_conversation_manager, _global_protocol_handler


def cleanup_messaging_system():
    """Limpia el sistema de mensajería global"""
    global _global_dispatcher, _global_conversation_manager, _global_protocol_handler
    
    if _global_dispatcher:
        _global_dispatcher.executor.shutdown(wait=True)
    
    _global_dispatcher = None
    _global_conversation_manager = None
    _global_protocol_handler = None