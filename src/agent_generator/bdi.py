from mesa import Agent
from .client import GeminiClient
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import json
import hashlib
import random
import time

class StrategyType(Enum):
    """Tipos de estrategia con pesos cuantificados"""
    EXPLORATORIA = "exploratoria"
    CONSERVADORA = "conservadora"
    SOCIAL = "social"
    EQUILIBRADA = "equilibrada"

@dataclass
class StrategyWeights:
    """Pesos numéricos para diferentes tipos de deseos según la estrategia"""
    explorar: float = 0.5
    socializar: float = 0.5
    descansar: float = 0.3
    aprender: float = 0.4
    comprar: float = 0.2
    fotografiar: float = 0.3
    
    @classmethod
    def get_strategy_weights(cls, strategy: StrategyType) -> 'StrategyWeights':
        """Retorna pesos específicos para cada estrategia"""
        if strategy == StrategyType.EXPLORATORIA:
            return cls(explorar=0.9, socializar=0.3, descansar=0.1, aprender=0.7, comprar=0.2, fotografiar=0.8)
        elif strategy == StrategyType.CONSERVADORA:
            return cls(explorar=0.2, socializar=0.4, descansar=0.8, aprender=0.5, comprar=0.3, fotografiar=0.4)
        elif strategy == StrategyType.SOCIAL:
            return cls(explorar=0.4, socializar=0.9, descansar=0.3, aprender=0.6, comprar=0.5, fotografiar=0.6)
        else:  # EQUILIBRADA
            return cls(explorar=0.5, socializar=0.5, descansar=0.5, aprender=0.5, comprar=0.4, fotografiar=0.5)

@dataclass
class Desire:
    """Deseo con prioridad cuantificada"""
    name: str
    priority: float
    context: Dict[str, Any] = field(default_factory=dict)
    created_at: int = 0
    
    def __post_init__(self):
        if self.created_at == 0:
            self.created_at = int(time.time())

@dataclass
class Intention:
    """Intención con plan asociado"""
    name: str
    priority: float
    plan_id: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    status: str = "active"  # active, completed, failed

class SemanticMemory:
    """Sistema de memoria semántica con embeddings simulados"""
    
    def __init__(self, max_size: int = 50):
        self.max_size = max_size
        self.memories: List[Tuple[str, np.ndarray, Dict[str, Any]]] = []
        self.embedding_cache: Dict[str, np.ndarray] = {}
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Genera embedding simulado para el texto"""
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        
        # Simulación de embedding usando hash + ruido
        hash_obj = hashlib.md5(text.encode())
        seed = int(hash_obj.hexdigest()[:8], 16)
        np.random.seed(seed)
        embedding = np.random.normal(0, 1, 64)  # Vector de 64 dimensiones
        embedding = embedding / np.linalg.norm(embedding)  # Normalizar
        
        self.embedding_cache[text] = embedding
        return embedding
    
    def add_memory(self, text: str, metadata: Dict[str, Any] = None):
        """Añade una memoria con su embedding"""
        if metadata is None:
            metadata = {}
        
        embedding = self._get_embedding(text)
        self.memories.append((text, embedding, metadata))
        
        # Mantener tamaño máximo
        if len(self.memories) > self.max_size:
            self.memories.pop(0)
    
    def search_similar(self, query: str, top_k: int = 3) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Busca memorias similares usando similitud coseno"""
        if not self.memories:
            return []
        
        query_embedding = self._get_embedding(query)
        similarities = []
        
        for text, embedding, metadata in self.memories:
            similarity = np.dot(query_embedding, embedding)
            similarities.append((text, similarity, metadata))
        
        # Ordenar por similitud descendente
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

class LLMValidator:
    """Validador para salidas del LLM"""
    
    @staticmethod
    def validate_list(text: str, expected_type: type = str) -> List[Any]:
        """Valida y parsea una lista del LLM"""
        try:
            import ast
            result = ast.literal_eval(text.strip())
            if isinstance(result, list):
                return result[:5]  # Limitar a 5 elementos máximo
            else:
                return []
        except:
            # Fallback: intentar extraer elementos de texto
            return LLMValidator._extract_list_from_text(text)
    
    @staticmethod
    def _extract_list_from_text(text: str) -> List[str]:
        """Extrae elementos de lista de texto no estructurado"""
        import re
        patterns = [
            r'^\d+\.\s*(.+)$',  # 1. item
            r'^[-*]\s*(.+)$',   # - item o * item
            r'^[•]\s*(.+)$',    # • item
        ]
        
        items = []
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            for pattern in patterns:
                match = re.match(pattern, line, re.MULTILINE)
                if match:
                    items.append(match.group(1).strip())
                    break
            else:
                if len(line) < 50 and len(line) > 3:
                    items.append(line)
        
        return items[:5]  # Limitar a 5 elementos máximo

class RuleEngine:
    """Motor de reglas para decisiones sin LLM"""
    
    def __init__(self):
        self.rules = self._initialize_tourist_rules()
    
    def _initialize_tourist_rules(self):
        """Inicializa reglas específicas de turismo"""
        return {
            'desire_generation': [
                self._energy_based_desires,
                self._social_desires,
                self._location_based_desires
            ],
            'feasibility_check': [
                self._check_energy_feasibility,
                self._check_social_feasibility
            ]
        }
    
    def _energy_based_desires(self, context):
        """Genera deseos basados en energía"""
        energy = context.get('energy', 5)
        if energy < 3:
            return [('descansar', 0.9), ('buscar_sombra', 0.7)]
        elif energy > 7:
            return [('explorar', 0.8), ('caminar', 0.6)]
        return []
    
    def _social_desires(self, context):
        """Genera deseos sociales"""
        nearby_agents = context.get('nearby_agents', [])
        satisfaction = context.get('satisfaction', 5)
        if len(nearby_agents) > 0 and satisfaction < 6:
            return [('socializar', 0.7), ('pedir_ayuda', 0.5)]
        return []
    
    def _location_based_desires(self, context):
        """Genera deseos basados en ubicación"""
        location_type = context.get('location_type', '')
        if 'museo' in location_type.lower():
            return [('aprender', 0.8), ('fotografiar', 0.6)]
        elif 'restaurante' in location_type.lower():
            return [('comer', 0.9), ('descansar', 0.5)]
        elif 'parque' in location_type.lower():
            return [('relajarse', 0.7), ('observar_naturaleza', 0.6)]
        return []
    
    def _check_energy_feasibility(self, context):
        """Verifica factibilidad basada en energía"""
        energy = context.get('energy', 5)
        desire = context.get('desire', '')
        
        if desire in ['explorar', 'caminar', 'correr'] and energy < 3:
            return False
        elif desire in ['descansar', 'sentarse'] and energy > 8:
            return False
        return True
    
    def _check_social_feasibility(self, context):
        """Verifica factibilidad social"""
        nearby_agents = context.get('nearby_agents', [])
        desire = context.get('desire', '')
        
        if desire in ['socializar', 'conversar', 'pedir_ayuda'] and len(nearby_agents) == 0:
            return False
        return True
    
    def apply_rules(self, category: str, context: Dict[str, Any]) -> List[Any]:
        """Aplica todas las reglas de una categoría"""
        results = []
        for rule in self.rules.get(category, []):
            try:
                result = rule(context)
                if result is not None:
                    results.append(result)
            except Exception as e:
                print(f"Error aplicando regla: {e}")
        return results

class AgenteBDI(Agent):
    """
    Agente BDI mejorado con menor dependencia del LLM, memoria semántica, 
    estrategias cuantificadas y validación robusta.
    """
    def __init__(self, unique_id, model, estrategia="equilibrada"):
        try:
            print(f"DEBUG - Inicializando AgenteBDI mejorado con unique_id={unique_id}, estrategia={estrategia}")
            # Call Mesa Agent constructor with only model - unique_id is auto-generated
            super().__init__(model)
            # Override the auto-generated unique_id if needed
            if unique_id is not None:
                self.unique_id = unique_id
            
            # Componentes BDI básicos mejorados
            self.beliefs = {}
            self.desires = []
            self.intentions = []
            self.plan = []
            
            # Estrategia cuantificada
            if isinstance(estrategia, str):
                strategy_map = {
                    "exploratoria": StrategyType.EXPLORATORIA,
                    "conservadora": StrategyType.CONSERVADORA,
                    "social": StrategyType.SOCIAL,
                    "equilibrada": StrategyType.EQUILIBRADA
                }
                self.strategy = strategy_map.get(estrategia, StrategyType.EQUILIBRADA)
            else:
                self.strategy = estrategia
            
            self.strategy_weights = StrategyWeights.get_strategy_weights(self.strategy)
            
            # LLM con uso limitado
            self.llm = GeminiClient()
            self.llm_call_count = 0
            self.max_llm_calls_per_step = 2  # Limitar llamadas por paso
            
            # Memoria semántica
            self.semantic_memory = SemanticMemory()
            
            # Motor de reglas
            self.rule_engine = RuleEngine()
            
            # Cache para evitar llamadas repetitivas al LLM
            self.llm_cache = {}
            
            # Memoria episódica (mantenida para compatibilidad)
            self.memoria_episodica = []
            self.estrategia = estrategia  # Para compatibilidad
            
            # Métricas de desempeño
            self.performance_metrics = {
                'decisions_made': 0,
                'successful_actions': 0,
                'failed_actions': 0,
                'llm_calls': 0
            }
            
            print(f"DEBUG - AgenteBDI mejorado inicializado exitosamente")
        except Exception as e:
            print(f"ERROR - Fallo en __init__ de AgenteBDI mejorado: {str(e)}")
            print(f"ERROR - Tipo de error: {type(e)}")
            raise e

    def percibir(self):
        """Percepción mejorada con menos dependencia del LLM"""
        # Percepción directa del entorno
        self.beliefs['location'] = getattr(self, 'location', 'unknown')
        self.beliefs['energy'] = getattr(self, 'energy', 5.0)
        self.beliefs['satisfaction'] = getattr(self, 'satisfaction', 5.0)
        
        # Percepción de otros agentes cercanos
        nearby_agents = self._get_nearby_agents()
        self.beliefs['nearby_agents'] = [agent.unique_id for agent in nearby_agents]
        
        # Solo usar LLM para análisis complejo si es necesario
        if self.llm_call_count < self.max_llm_calls_per_step and len(self.semantic_memory.memories) > 3:
            context_summary = self._get_llm_context_summary()
            if context_summary:
                self.beliefs['context_summary'] = context_summary
                self.llm_call_count += 1

    def deliberar(self):
        """Deliberación con estrategias cuantificadas y reglas"""
        # Generar deseos basados en reglas (sin LLM)
        new_desires = []
        
        # Crear contexto para reglas
        context = {
            'energy': self.beliefs.get('energy', 5),
            'satisfaction': self.beliefs.get('satisfaction', 5),
            'location_type': self.beliefs.get('location', ''),
            'nearby_agents': self.beliefs.get('nearby_agents', [])
        }
        
        # Aplicar reglas de generación de deseos
        rule_results = self.rule_engine.apply_rules('desire_generation', context)
        for result in rule_results:
            if isinstance(result, list):
                for desire_name, priority in result:
                    # Ajustar prioridad según estrategia
                    adjusted_priority = self._adjust_priority_by_strategy(desire_name, priority)
                    new_desires.append(Desire(desire_name, adjusted_priority, {'source': 'rules'}))
        
        # Complementar con LLM solo si es necesario y hay presupuesto
        if len(new_desires) < 2 and self.llm_call_count < self.max_llm_calls_per_step:
            llm_desires = self._get_llm_desires()
            new_desires.extend(llm_desires)
            if llm_desires:
                self.llm_call_count += 1
        
        # Actualizar lista de deseos
        self.desires = new_desires
        
        # Añadir a memoria semántica
        if new_desires:
            desire_summary = f"Generated desires: {[d.name for d in new_desires]}"
            self.semantic_memory.add_memory(desire_summary, {'type': 'deliberation', 'strategy': self.strategy.value})

    def filtrar_deseos(self):
        """Filtrado de deseos con priorización cuantitativa"""
        if not self.desires:
            return
        
        # Ordenar deseos por prioridad
        self.desires.sort(key=lambda d: d.priority, reverse=True)
        
        # Convertir top deseos en intenciones
        max_intentions = 3  # Limitar número de intenciones activas
        new_intentions = []
        
        context = {
            'energy': self.beliefs.get('energy', 5),
            'satisfaction': self.beliefs.get('satisfaction', 5),
            'location_type': self.beliefs.get('location', ''),
            'nearby_agents': self.beliefs.get('nearby_agents', [])
        }
        
        for desire in self.desires[:max_intentions]:
            # Validar factibilidad usando reglas
            context['desire'] = desire.name
            feasibility_results = self.rule_engine.apply_rules('feasibility_check', context)
            
            # Si alguna regla dice que no es factible, no lo es
            is_feasible = all(result != False for result in feasibility_results)
            
            if is_feasible:
                intention = Intention(
                    name=desire.name,
                    priority=desire.priority,
                    context=desire.context.copy()
                )
                new_intentions.append(intention)
        
        self.intentions = new_intentions

    def planificar(self):
        """Planificación simplificada basada en intenciones"""
        self.plan = []
        for intention in self.intentions:
            # Plan simple basado en el nombre de la intención
            if intention.name in ['descansar', 'buscar_sombra']:
                self.plan.append('encontrar_lugar_descanso')
                self.plan.append('descansar')
            elif intention.name in ['socializar', 'pedir_ayuda']:
                self.plan.append('acercarse_agente')
                self.plan.append('iniciar_conversacion')
            elif intention.name in ['explorar', 'caminar']:
                self.plan.append('elegir_direccion')
                self.plan.append('moverse')
            elif intention.name in ['aprender', 'fotografiar']:
                self.plan.append('observar_entorno')
                self.plan.append('procesar_informacion')
            else:
                self.plan.append(f'ejecutar_{intention.name}')

    def actuar(self):
        """Ejecución de acciones con seguimiento de métricas"""
        if not self.plan:
            return
        
        accion = self.plan.pop(0)
        
        # Simular ejecución de acción
        success = self._execute_action(accion)
        
        if success:
            self.performance_metrics['successful_actions'] += 1
            resultado = f"Acción '{accion}' ejecutada exitosamente"
        else:
            self.performance_metrics['failed_actions'] += 1
            resultado = f"Acción '{accion}' falló"
        
        self.beliefs['ultimo_resultado'] = resultado
        
        # Guardar en memoria semántica
        self.semantic_memory.add_memory(resultado, {'type': 'action', 'success': success})
        
        # Guardar episodio en memoria episódica (compatibilidad)
        self.memoria_episodica.append((
            self.beliefs.copy(), 
            [d.name for d in self.desires], 
            [i.name for i in self.intentions], 
            accion, 
            resultado
        ))
        
        self.performance_metrics['decisions_made'] += 1

    def reflexionar(self):
        """Reflexión con memoria semántica"""
        # Buscar patrones en memoria
        if len(self.semantic_memory.memories) > 5:
            recent_actions = self.semantic_memory.search_similar("action", top_k=3)
            
            if recent_actions:
                pattern_summary = f"Recent action patterns: {[action[0] for action in recent_actions]}"
                self.semantic_memory.add_memory(pattern_summary, {'type': 'reflection'})
        
        # Ajustar estrategia basada en desempeño (sin LLM)
        success_rate = self.performance_metrics['successful_actions'] / max(1, self.performance_metrics['decisions_made'])
        if success_rate < 0.3:  # Si el éxito es bajo, ser más conservador
            self.strategy_weights.explorar *= 0.9
            self.strategy_weights.descansar *= 1.1

    def compartir_creencias(self, otros_agentes):
        """Comunicación multi-agente simplificada"""
        if not otros_agentes or random.random() > 0.3:  # 30% probabilidad de comunicar
            return
        
        # Compartir información básica con agentes cercanos
        for agente in otros_agentes[:2]:  # Máximo 2 agentes
            if hasattr(agente, 'beliefs'):
                shared_info = {
                    'location': self.beliefs.get('location', 'unknown'),
                    'satisfaction': self.beliefs.get('satisfaction', 5.0),
                    'strategy': self.strategy.value
                }
                agente.beliefs[f'shared_{self.unique_id}'] = shared_info

    def step(self):
        """Ciclo BDI mejorado"""
        self.llm_call_count = 0  # Reset contador de llamadas LLM
        
        self.percibir()
        self.deliberar()
        self.filtrar_deseos()
        self.planificar()
        self.actuar()
        self.reflexionar()
        
        # Comunicación multi-agente
        if hasattr(self.model, 'schedule'):
            otros = [a for a in self.model.schedule.agents if a != self]
            self.compartir_creencias(otros)
        
        self.performance_metrics['llm_calls'] += self.llm_call_count

    # Métodos auxiliares mejorados
    def _get_nearby_agents(self) -> List[Agent]:
        """Obtiene agentes cercanos sin usar LLM"""
        if not hasattr(self.model, 'schedule'):
            return []
        return [a for a in self.model.schedule.agents if a != self][:3]  # Máximo 3 agentes

    def _get_llm_context_summary(self) -> Optional[str]:
        """Obtiene resumen de contexto del LLM con cache"""
        context_key = f"context_{hash(str(self.beliefs))}"
        if context_key in self.llm_cache:
            return self.llm_cache[context_key]
        
        try:
            prompt = f"Summarize this context briefly: {self.beliefs}"
            summary = self.llm.generate(prompt)
            self.llm_cache[context_key] = summary
            return summary
        except:
            return None

    def _get_llm_desires(self) -> List[Desire]:
        """Obtiene deseos del LLM con validación"""
        try:
            prompt = f"Given beliefs {self.beliefs} and strategy {self.strategy.value}, suggest 2-3 desires as a Python list."
            response = self.llm.generate(prompt)
            desire_names = LLMValidator.validate_list(response)
            
            desires = []
            for name in desire_names[:3]:  # Máximo 3 deseos del LLM
                priority = getattr(self.strategy_weights, name.lower(), 0.5)
                desires.append(Desire(name, priority, {'source': 'llm'}))
            
            return desires
        except:
            return []

    def _adjust_priority_by_strategy(self, desire_name: str, base_priority: float) -> float:
        """Ajusta prioridad según estrategia"""
        strategy_multiplier = getattr(self.strategy_weights, desire_name.lower(), 1.0)
        return min(1.0, base_priority * strategy_multiplier)

    def _execute_action(self, action: str) -> bool:
        """Simula ejecuci��n de acción"""
        # Simulación simple de éxito/fallo
        success_rates = {
            'encontrar_lugar_descanso': 0.8,
            'descansar': 0.9,
            'acercarse_agente': 0.7,
            'iniciar_conversacion': 0.6,
            'elegir_direccion': 0.9,
            'moverse': 0.8,
            'observar_entorno': 0.9,
            'procesar_informacion': 0.7
        }
        
        success_rate = success_rates.get(action, 0.7)
        return random.random() < success_rate

    # Métodos de compatibilidad con sistema original
    def _generar_contexto_percepcion(self):
        """Extrae contexto relevante del entorno/modelo"""
        return str(self.model)

    def _resumir_memoria_llm(self, n=3):
        """Resume los últimos n episodios para el LLM"""
        if not self.memoria_episodica:
            return "Sin memoria."
        resumen = "\n".join([
            f"Episodio {i+1}: percepcion={e[0]}, deseo={e[1]}, intencion={e[2]}, accion={e[3]}, resultado={e[4]}"
            for i, e in enumerate(self.memoria_episodica[-n:])
        ])
        return resumen

    def _parsear_lista_llm(self, texto_llm):
        """Parsea lista del LLM con validación"""
        return LLMValidator.validate_list(texto_llm)