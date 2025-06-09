import skfuzzy as fuzz
from skfuzzy import control as ctrl
import numpy as np
import random

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
