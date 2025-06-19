from mesa import Model, Agent
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from .bdi import AgenteBDI, StrategyType, Desire, Intention


@dataclass
class Nodo:
    """
    Representa un nodo/lugar del recorrido turístico.
    """
    id: str = ""
    nombre: str = ""
    tipo: str = ""
    descripcion: str = ""
    agentes: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Ensure agentes is always a list"""
        if self.agentes is None:
            self.agentes = []
        elif not isinstance(self.agentes, list):
            self.agentes = list(self.agentes) if hasattr(self.agentes, '__iter__') else [self.agentes]

class AgenteBDIBase(AgenteBDI):
    """
    Agente BDI mejorado para la simulación turística.
    Integra todas las mejoras del sistema BDI.
    """
    def __init__(self, unique_id, model, rol, lugar_id, prompt, estrategia="equilibrada"):
        try:
            print(f"DEBUG - Inicializando AgenteBDIBase mejorado con unique_id={unique_id}, rol={rol}")
            # Call parent init with improved BDI system
            super().__init__(unique_id, model, estrategia)
            
            # Atributos específicos del agente turístico
            self.rol = rol
            self.lugar_id = lugar_id
            self.prompt = prompt
            self.interacciones = []
            
            # Atributos mejorados
            self.location = lugar_id
            self.energy = 7.0  # Energía inicial
            self.satisfaction = 5.0  # Satisfacción inicial
            
            # Configurar creencias iniciales específicas
            self.beliefs.update({
                'lugar_id': lugar_id,
                'rol': rol,
                'location': lugar_id,
                'energy': self.energy,
                'satisfaction': self.satisfaction
            })
            
            print(f"DEBUG - AgenteBDIBase mejorado inicializado exitosamente")
        except Exception as e:
            print(f"ERROR - Fallo en __init__ de AgenteBDIBase mejorado: {str(e)}")
            print(f"ERROR - Tipo de error: {type(e)}")
            raise e

    def percibir(self):
        """Percepción mejorada específica para agentes turísticos"""
        # Llamar percepción base mejorada
        super().percibir()
        
        # Añadir percepciones específicas del rol
        self.beliefs.update({
            'lugar_id': self.lugar_id,
            'rol': self.rol,
            'location': self.location,
            'energy': self.energy,
            'satisfaction': self.satisfaction,
            'interacciones_realizadas': len(self.interacciones)
        })
        
        # Percibir turistas cercanos si es un agente de servicio
        if self.rol in ['guía', 'mesero', 'vendedor', 'asistente']:
            turistas_cercanos = self._detectar_turistas_cercanos()
            self.beliefs['turistas_cercanos'] = len(turistas_cercanos)

    def deliberar(self):
        """Deliberación mejorada con reglas específicas del rol"""
        # Llamar deliberación base mejorada
        super().deliberar()
        
        # Añadir deseos específicos del rol
        role_desires = self._generate_role_specific_desires()
        
        # Convertir a objetos Desire y añadir
        for desire_name, priority in role_desires:
            adjusted_priority = self._adjust_priority_by_strategy(desire_name, priority)
            self.desires.append(Desire(desire_name, adjusted_priority, {'source': 'role', 'role': self.rol}))

    def actuar(self):
        """Ejecución mejorada con acciones específicas del rol"""
        # Llamar ejecución base mejorada
        super().actuar()
        
        # Procesar resultados específicos del rol
        if self.beliefs.get('ultimo_resultado'):
            self._process_role_specific_result()

    def _detectar_turistas_cercanos(self) -> List[Agent]:
        """Detecta turistas cercanos para agentes de servicio"""
        if not hasattr(self.model, 'schedule'):
            return []
        
        turistas = []
        for agent in self.model.schedule.agents:
            if hasattr(agent, 'rol') and agent.rol == 'turista':
                turistas.append(agent)
        
        return turistas[:3]  # Máximo 3 turistas cercanos

    def _generate_role_specific_desires(self) -> List[tuple]:
        """Genera deseos específicos según el rol del agente"""
        role_desires = {
            'guía': [
                ('compartir_conocimiento', 0.8),
                ('ayudar_turistas', 0.9),
                ('mantener_grupo', 0.7)
            ],
            'mesero': [
                ('atender_clientes', 0.9),
                ('tomar_pedidos', 0.8),
                ('mantener_servicio', 0.7)
            ],
            'vendedor': [
                ('mostrar_productos', 0.8),
                ('realizar_ventas', 0.9),
                ('atender_consultas', 0.7)
            ],
            'curador': [
                ('explicar_exhibiciones', 0.9),
                ('mantener_orden', 0.6),
                ('educar_visitantes', 0.8)
            ],
            'asistente': [
                ('ayudar_visitantes', 0.8),
                ('proporcionar_informacion', 0.7),
                ('mantener_area', 0.5)
            ]
        }
        
        return role_desires.get(self.rol, [('cumplir_funcion', 0.6)])

    def _process_role_specific_result(self):
        """Procesa resultados específicos del rol"""
        resultado = self.beliefs.get('ultimo_resultado', '')
        
        # Ajustar energía y satisfacción según el resultado
        if 'exitosamente' in resultado:
            self.energy = max(0, self.energy - 0.5)  # Gastar energía
            self.satisfaction = min(10, self.satisfaction + 0.3)  # Aumentar satisfacción
        elif 'falló' in resultado:
            self.energy = max(0, self.energy - 0.2)
            self.satisfaction = max(0, self.satisfaction - 0.1)
        
        # Actualizar creencias
        self.beliefs.update({
            'energy': self.energy,
            'satisfaction': self.satisfaction
        })

    def interactuar_con_turista(self, turista):
        """Método mejorado de interacción con turistas"""
        if not hasattr(turista, 'agregar_experiencia'):
            return False
        
        try:
            # Usar sistema difuso para calcular impacto
            if hasattr(self.model, 'sistema_difuso'):
                amabilidad = getattr(self, 'amabilidad', 7.0)
                impacto = self.model.sistema_difuso.calcular_impacto(
                    amabilidad, 
                    turista.satisfaccion,
                    lugar_tipo=self.lugar_id
                )
            else:
                # Fallback simple
                impacto = (self.satisfaction - 5) * 0.3
            
            # Crear experiencia basada en el rol
            experiencia = self._crear_experiencia_por_rol(turista)
            
            # Añadir experiencia al turista
            turista.agregar_experiencia(experiencia, impacto)
            
            # Registrar interacción
            self.interacciones.append({
                'turista_id': turista.unique_id,
                'experiencia': experiencia,
                'impacto': impacto,
                'timestamp': len(self.interacciones)
            })
            
            # Añadir a memoria semántica
            self.semantic_memory.add_memory(
                f"Interacted with tourist {turista.unique_id}: {experiencia}",
                {'type': 'interaction', 'role': self.rol, 'impact': impacto}
            )
            
            return True
            
        except Exception as e:
            print(f"Error en interacción: {e}")
            return False

    def _crear_experiencia_por_rol(self, turista) -> str:
        """Crea experiencia específica según el rol"""
        experiencias_por_rol = {
            'guía': [
                f"El guía te explica la historia fascinante de {self.lugar_id}",
                f"Recibes información valiosa sobre {self.lugar_id}",
                f"El guía te muestra detalles únicos del lugar"
            ],
            'mesero': [
                f"El mesero te atiende con una sonrisa en {self.lugar_id}",
                f"Recibes un servicio excelente en el restaurante",
                f"El mesero te recomienda especialidades locales"
            ],
            'vendedor': [
                f"El vendedor te muestra productos únicos de {self.lugar_id}",
                f"Descubres artesanías locales interesantes",
                f"El vendedor te cuenta sobre la tradición local"
            ],
            'curador': [
                f"El curador te explica el significado de las exhibiciones",
                f"Aprendes sobre la cultura local en {self.lugar_id}",
                f"El curador comparte conocimientos especializados"
            ]
        }
        
        experiencias = experiencias_por_rol.get(self.rol, [f"Tienes una interacción en {self.lugar_id}"])
        import random
        return random.choice(experiencias)

class TuristaBDI(AgenteBDI):
    """
    Turista con arquitectura BDI mejorada.
    Integra todas las mejoras del sistema BDI.
    """
    def __init__(self, unique_id, model, nombre, estrategia="equilibrada"):
        try:
            print(f"DEBUG - Inicializando TuristaBDI mejorado con unique_id={unique_id}, nombre={nombre}")
            # Call parent init with improved BDI system
            super().__init__(unique_id, model, estrategia)
            
            # Atributos específicos del turista
            self.nombre = nombre
            self.satisfaccion = 5.0
            self.rol = 'turista'  # Añadir rol para compatibilidad
            
            # Sistema de memoria jerárquica (mantenido para compatibilidad)
            self.memoria_alta = []
            self.memoria_media = []
            self.memoria_baja = []
            self.contexto_actual = []
            
            # Atributos mejorados
            self.location = "inicio"
            self.energy = 8.0  # Los turistas empiezan con más energía
            self.satisfaction = self.satisfaccion  # Sincronizar con satisfaccion
            
            # Configurar creencias iniciales específicas del turista
            self.beliefs.update({
                'nombre': nombre,
                'satisfaccion': self.satisfaccion,
                'rol': 'turista',
                'location': self.location,
                'energy': self.energy,
                'satisfaction': self.satisfaction
            })
            
            print(f"DEBUG - TuristaBDI mejorado inicializado exitosamente")
        except Exception as e:
            print(f"ERROR - Fallo en __init__ de TuristaBDI mejorado: {str(e)}")
            print(f"ERROR - Tipo de error: {type(e)}")
            raise e

    LIMITE_ALTA = 5
    LIMITE_MEDIA = 7
    LIMITE_BAJA = 10

    def percibir(self):
        """Percepción mejorada específica para turistas"""
        # Llamar percepción base mejorada
        super().percibir()
        
        # Añadir percepciones específicas del turista
        self.beliefs.update({
            'satisfaccion': self.satisfaccion,
            'nombre': self.nombre,
            'memoria_total': len(self.memoria_alta) + len(self.memoria_media) + len(self.memoria_baja),
            'energia_nivel': self._categorizar_energia(),
            'satisfaccion_nivel': self._categorizar_satisfaccion()
        })

    def deliberar(self):
        """Deliberación mejorada específica para turistas"""
        # Llamar deliberación base mejorada
        super().deliberar()
        
        # Añadir deseos específicos del turista basados en su estado
        tourist_desires = self._generate_tourist_specific_desires()
        
        # Convertir a objetos Desire y añadir
        for desire_name, priority in tourist_desires:
            adjusted_priority = self._adjust_priority_by_strategy(desire_name, priority)
            self.desires.append(Desire(desire_name, adjusted_priority, {'source': 'tourist_state'}))

    def actuar(self):
        """Ejecución mejorada específica para turistas"""
        # Llamar ejecución base mejorada
        super().actuar()
        
        # Sincronizar satisfacción con satisfaction
        self.satisfaccion = self.satisfaction
        
        # Procesar resultados específicos del turista
        if self.beliefs.get('ultimo_resultado'):
            self._process_tourist_specific_result()

    def _categorizar_energia(self) -> str:
        """Categoriza el nivel de energía"""
        if self.energy >= 7:
            return "alta"
        elif self.energy >= 4:
            return "media"
        else:
            return "baja"

    def _categorizar_satisfaccion(self) -> str:
        """Categoriza el nivel de satisfacción"""
        if self.satisfaccion >= 7:
            return "alta"
        elif self.satisfaccion >= 4:
            return "media"
        else:
            return "baja"

    def _generate_tourist_specific_desires(self) -> List[tuple]:
        """Genera deseos específicos del turista basados en su estado"""
        desires = []
        
        # Deseos basados en satisfacción
        if self.satisfaccion < 4:
            desires.extend([
                ('buscar_ayuda', 0.8),
                ('encontrar_algo_interesante', 0.7),
                ('mejorar_experiencia', 0.9)
            ])
        elif self.satisfaccion > 7:
            desires.extend([
                ('explorar_mas', 0.7),
                ('compartir_experiencia', 0.6),
                ('disfrutar_momento', 0.8)
            ])
        
        # Deseos basados en energía
        if self.energy < 3:
            desires.extend([
                ('descansar', 0.9),
                ('buscar_lugar_comodo', 0.8)
            ])
        elif self.energy > 7:
            desires.extend([
                ('explorar', 0.8),
                ('ser_activo', 0.7)
            ])
        
        # Deseos basados en memoria
        if len(self.memoria_alta) == 0:
            desires.append(('crear_recuerdos_memorables', 0.8))
        
        return desires

    def _process_tourist_specific_result(self):
        """Procesa resultados específicos del turista"""
        resultado = self.beliefs.get('ultimo_resultado', '')
        
        # Ajustar energía y satisfacción según el resultado
        if 'exitosamente' in resultado:
            self.energy = max(0, self.energy - 0.3)  # Los turistas gastan menos energía
            self.satisfaction = min(10, self.satisfaction + 0.5)  # Más ganancia de satisfacción
        elif 'falló' in resultado:
            self.energy = max(0, self.energy - 0.1)
            self.satisfaction = max(0, self.satisfaction - 0.3)  # Más pérdida de satisfacción
        
        # Sincronizar satisfaccion con satisfaction
        self.satisfaccion = self.satisfaction
        
        # Actualizar creencias
        self.beliefs.update({
            'energy': self.energy,
            'satisfaction': self.satisfaction,
            'satisfaccion': self.satisfaccion
        })

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