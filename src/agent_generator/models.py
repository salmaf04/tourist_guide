from mesa import Model, Agent
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from .bdi import AgenteBDI


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
    Agente BDI genérico para la simulación turística.
    Puede ser extendido para roles específicos.
    """
    def __init__(self, unique_id, model, rol, lugar_id, prompt):
        try:
            print(f"DEBUG - Inicializando AgenteBDIBase con unique_id={unique_id}, rol={rol}")
            # Call parent init with correct parameter order
            super().__init__(unique_id, model)
            self.rol = rol
            self.lugar_id = lugar_id
            self.prompt = prompt
            self.interacciones = []
            print(f"DEBUG - AgenteBDIBase inicializado exitosamente")
        except Exception as e:
            print(f"ERROR - Fallo en __init__ de AgenteBDIBase: {str(e)}")
            print(f"ERROR - Tipo de error: {type(e)}")
            raise e

    def percibir(self):
        # Ejemplo: percibir entorno, estado propio, etc.
        self.beliefs['lugar_id'] = self.lugar_id
        self.beliefs['rol'] = self.rol
        # ...agregar percepciones relevantes...

    def deliberar(self):
        # Ejemplo: generar deseos según creencias
        if 'satisfaccion' in self.beliefs and self.beliefs['satisfaccion'] < 7:
            if 'aumentar_satisfaccion' not in self.desires:
                self.desires.append('aumentar_satisfaccion')
        # ...otros deseos...

    def filtrar_deseos(self):
        # Priorizar deseos y convertir en intenciones
        if 'aumentar_satisfaccion' in self.desires and 'buscar_interaccion_positiva' not in self.intentions:
            self.intentions.append('buscar_interaccion_positiva')

    def planificar(self):
        # Generar plan concreto para cada intención
        self.plan = []
        for intention in self.intentions:
            if intention == 'buscar_interaccion_positiva':
                self.plan.append('interactuar_con_turista')
        # ...otros planes...

    def actuar(self):
        # Ejecutar la siguiente acción del plan
        if self.plan:
            accion = self.plan.pop(0)
            if accion == 'interactuar_con_turista':
                # Aquí se llamaría a la lógica de interacción
                pass
        # ...otras acciones...

class TuristaBDI(AgenteBDI):
    """
    Turista con arquitectura BDI.
    """
    def __init__(self, unique_id, model, nombre):
        try:
            print(f"DEBUG - Inicializando TuristaBDI con unique_id={unique_id}, nombre={nombre}")
            # Call parent init with correct parameter order
            super().__init__(unique_id, model)
            self.nombre = nombre
            self.satisfaccion = 5.0
            self.memoria_alta = []
            self.memoria_media = []
            self.memoria_baja = []
            self.contexto_actual = []
            print(f"DEBUG - TuristaBDI inicializado exitosamente")
        except Exception as e:
            print(f"ERROR - Fallo en __init__ de TuristaBDI: {str(e)}")
            print(f"ERROR - Tipo de error: {type(e)}")
            raise e

    LIMITE_ALTA = 5
    LIMITE_MEDIA = 7
    LIMITE_BAJA = 10

    def percibir(self):
        self.beliefs['satisfaccion'] = self.satisfaccion
        # ...otros datos del entorno...

    def deliberar(self):
        if self.beliefs['satisfaccion'] < 7 and 'aumentar_satisfaccion' not in self.desires:
            self.desires.append('aumentar_satisfaccion')

    def filtrar_deseos(self):
        if 'aumentar_satisfaccion' in self.desires and 'buscar_interaccion_positiva' not in self.intentions:
            self.intentions.append('buscar_interaccion_positiva')

    def planificar(self):
        self.plan = []
        for intention in self.intentions:
            if intention == 'buscar_interaccion_positiva':
                self.plan.append('buscar_agente_amigable')
                self.plan.append('interactuar')

    def actuar(self):
        if self.plan:
            accion = self.plan.pop(0)
            if accion == 'buscar_agente_amigable':
                # Buscar agentes con mayor amabilidad
                pass
            elif accion == 'interactuar':
                # Ejecutar interacción
                pass

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