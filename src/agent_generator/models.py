from mesa import Model, Agent
from dataclasses import dataclass, field
from typing import List, Dict, Optional


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
