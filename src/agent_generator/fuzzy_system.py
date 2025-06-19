import skfuzzy as fuzz
from skfuzzy import control as ctrl
import numpy as np
import random

class SistemaDifusoImpacto:
    """
    Sistema difuso avanzado para calcular el impacto de una interacción considerando múltiples factores.
    """
    def __init__(self):
        # Variables de entrada
        self.satisfaccion = ctrl.Antecedent(np.arange(0, 11, 1), 'satisfaccion')
        self.amabilidad = ctrl.Antecedent(np.arange(0, 11, 1), 'amabilidad')
        self.calidad_lugar = ctrl.Antecedent(np.arange(0, 11, 1), 'calidad_lugar')
        self.fatiga = ctrl.Antecedent(np.arange(0, 11, 1), 'fatiga')
        
        # Variable de salida
        self.impacto = ctrl.Consequent(np.arange(-3, 3.1, 0.1), 'impacto')
        
        self._definir_variables()
        self._definir_reglas()
        self.sistema_control = ctrl.ControlSystem(self.reglas)
        self.simulador = ctrl.ControlSystemSimulation(self.sistema_control)

    def _definir_variables(self):
        """
        Define las variables lingüísticas y sus funciones de membresía mejoradas.
        """
        # Satisfacción
        self.satisfaccion['muy_baja'] = fuzz.trimf(self.satisfaccion.universe, [0, 0, 2])
        self.satisfaccion['baja'] = fuzz.trimf(self.satisfaccion.universe, [1, 3, 5])
        self.satisfaccion['media'] = fuzz.trimf(self.satisfaccion.universe, [3, 5, 7])
        self.satisfaccion['alta'] = fuzz.trimf(self.satisfaccion.universe, [5, 7, 9])
        self.satisfaccion['muy_alta'] = fuzz.trimf(self.satisfaccion.universe, [8, 10, 10])
        
        # Amabilidad
        self.amabilidad['muy_poca'] = fuzz.trimf(self.amabilidad.universe, [0, 0, 2])
        self.amabilidad['poca'] = fuzz.trimf(self.amabilidad.universe, [1, 3, 5])
        self.amabilidad['normal'] = fuzz.trimf(self.amabilidad.universe, [3, 5, 7])
        self.amabilidad['mucha'] = fuzz.trimf(self.amabilidad.universe, [5, 7, 9])
        self.amabilidad['excepcional'] = fuzz.trimf(self.amabilidad.universe, [8, 10, 10])
        
        # Calidad del lugar
        self.calidad_lugar['muy_baja'] = fuzz.trimf(self.calidad_lugar.universe, [0, 0, 2])
        self.calidad_lugar['baja'] = fuzz.trimf(self.calidad_lugar.universe, [1, 3, 5])
        self.calidad_lugar['media'] = fuzz.trimf(self.calidad_lugar.universe, [3, 5, 7])
        self.calidad_lugar['alta'] = fuzz.trimf(self.calidad_lugar.universe, [5, 7, 9])
        self.calidad_lugar['excepcional'] = fuzz.trimf(self.calidad_lugar.universe, [8, 10, 10])
        
        # Fatiga
        self.fatiga['muy_baja'] = fuzz.trimf(self.fatiga.universe, [0, 0, 2])
        self.fatiga['baja'] = fuzz.trimf(self.fatiga.universe, [1, 3, 5])
        self.fatiga['media'] = fuzz.trimf(self.fatiga.universe, [3, 5, 7])
        self.fatiga['alta'] = fuzz.trimf(self.fatiga.universe, [5, 7, 9])
        self.fatiga['muy_alta'] = fuzz.trimf(self.fatiga.universe, [8, 10, 10])
        
        # Impacto (ampliado)
        self.impacto['muy_negativo'] = fuzz.trimf(self.impacto.universe, [-3, -3, -1.5])
        self.impacto['negativo'] = fuzz.trimf(self.impacto.universe, [-2, -1, 0])
        self.impacto['neutral'] = fuzz.trimf(self.impacto.universe, [-0.5, 0, 0.5])
        self.impacto['positivo'] = fuzz.trimf(self.impacto.universe, [0, 1, 2])
        self.impacto['muy_positivo'] = fuzz.trimf(self.impacto.universe, [1.5, 3, 3])

    def _definir_reglas(self):
        """
        Define reglas difusas más complejas y realistas.
        """
        self.reglas = [
            # Reglas para experiencias excepcionales
            ctrl.Rule(self.amabilidad['excepcional'] & self.calidad_lugar['excepcional'] & self.satisfaccion['baja'], 
                     self.impacto['muy_positivo']),
            ctrl.Rule(self.amabilidad['excepcional'] & self.calidad_lugar['alta'] & self.fatiga['muy_baja'], 
                     self.impacto['muy_positivo']),
            
            # Reglas para experiencias muy positivas
            ctrl.Rule(self.amabilidad['mucha'] & self.calidad_lugar['alta'] & self.satisfaccion['baja'], 
                     self.impacto['positivo']),
            ctrl.Rule(self.amabilidad['mucha'] & self.satisfaccion['media'] & self.fatiga['baja'], 
                     self.impacto['positivo']),
            ctrl.Rule(self.calidad_lugar['excepcional'] & self.satisfaccion['media'], 
                     self.impacto['positivo']),
            
            # Reglas para experiencias negativas
            ctrl.Rule(self.amabilidad['muy_poca'] & self.satisfaccion['alta'], 
                     self.impacto['negativo']),
            ctrl.Rule(self.amabilidad['poca'] & self.calidad_lugar['muy_baja'], 
                     self.impacto['negativo']),
            ctrl.Rule(self.fatiga['muy_alta'] & self.amabilidad['poca'], 
                     self.impacto['muy_negativo']),
            ctrl.Rule(self.fatiga['alta'] & self.calidad_lugar['baja'], 
                     self.impacto['negativo']),
            
            # Reglas para experiencias neutrales
            ctrl.Rule(self.amabilidad['normal'] & self.satisfaccion['alta'] & self.fatiga['media'], 
                     self.impacto['neutral']),
            ctrl.Rule(self.calidad_lugar['media'] & self.satisfaccion['media'], 
                     self.impacto['neutral']),
            
            # Reglas específicas para diferentes combinaciones
            ctrl.Rule(self.amabilidad['mucha'] & self.satisfaccion['muy_alta'], 
                     self.impacto['positivo']),
            ctrl.Rule(self.amabilidad['normal'] & self.calidad_lugar['alta'] & self.satisfaccion['baja'], 
                     self.impacto['positivo']),
            ctrl.Rule(self.amabilidad['poca'] & self.satisfaccion['muy_baja'], 
                     self.impacto['muy_negativo']),
            
            # Reglas para fatiga
            ctrl.Rule(self.fatiga['muy_alta'] & self.satisfaccion['media'], 
                     self.impacto['negativo']),
            ctrl.Rule(self.fatiga['baja'] & self.amabilidad['mucha'], 
                     self.impacto['positivo']),
        ]

    def calcular_impacto(self, amabilidad_valor: float, satisfaccion_actual: float, 
                        calidad_lugar: float = None, fatiga: float = None, 
                        lugar_tipo: str = None) -> float:
        """
        Calcula el impacto de una interacción usando lógica difusa avanzada.
        
        :param amabilidad_valor: Nivel de amabilidad del agente (0-10)
        :param satisfaccion_actual: Satisfacción actual del turista (0-10)
        :param calidad_lugar: Calidad percibida del lugar (0-10)
        :param fatiga: Nivel de fatiga del turista (0-10)
        :param lugar_tipo: Tipo de lugar para ajustes específicos
        :return: Impacto calculado (-3 a 3)
        """
        try:
            # Asegurar valores dentro de rangos válidos
            amabilidad_valor = max(0, min(10, amabilidad_valor))
            satisfaccion_actual = max(0, min(10, satisfaccion_actual))
            
            # Calcular calidad del lugar si no se proporciona
            if calidad_lugar is None:
                calidad_lugar = self._calcular_calidad_lugar(lugar_tipo, amabilidad_valor)
            calidad_lugar = max(0, min(10, calidad_lugar))
            
            # Calcular fatiga si no se proporciona
            if fatiga is None:
                fatiga = self._calcular_fatiga(satisfaccion_actual)
            fatiga = max(0, min(10, fatiga))
            
            # Aplicar el sistema difuso
            self.simulador.input['amabilidad'] = amabilidad_valor
            self.simulador.input['satisfaccion'] = satisfaccion_actual
            self.simulador.input['calidad_lugar'] = calidad_lugar
            self.simulador.input['fatiga'] = fatiga
            
            self.simulador.compute()
            
            impacto_base = self.simulador.output.get('impacto', 0.0)
            
            # Aplicar ajustes específicos por tipo de lugar
            impacto_ajustado = self._ajustar_por_tipo_lugar(impacto_base, lugar_tipo, satisfaccion_actual)
            
            # Añadir variabilidad realista
            variabilidad = random.uniform(-0.2, 0.2)
            impacto_final = impacto_ajustado + variabilidad
            
            # Limitar el impacto final
            impacto_final = max(-3, min(3, impacto_final))
            
            print(f"DEBUG - Fuzzy avanzado: amabilidad={amabilidad_valor:.2f}, satisfaccion={satisfaccion_actual:.2f}, "
                  f"calidad={calidad_lugar:.2f}, fatiga={fatiga:.2f}, impacto={impacto_final:.2f}")
            
            return impacto_final
            
        except Exception as e:
            print(f"Error en calcular_impacto: {e}")
            # Fallback más inteligente
            fallback_impact = self._calcular_impacto_fallback(amabilidad_valor, satisfaccion_actual, lugar_tipo)
            print(f"DEBUG - Using intelligent fallback impact: {fallback_impact:.2f}")
            return fallback_impact

    def _calcular_calidad_lugar(self, lugar_tipo: str, amabilidad_base: float) -> float:
        """
        Calcula la calidad percibida del lugar basada en su tipo.
        """
        calidades_base = {
            'museo': 7.5,
            'monumento': 7.0,
            'parque': 6.5,
            'restaurante': 6.0,
            'iglesia': 6.8,
            'mercado': 5.5,
            'tienda': 5.0,
            'playa': 7.2,
            'mirador': 8.0,
            'atraccion': 6.0
        }
        
        calidad_base = calidades_base.get(lugar_tipo, 6.0)
        # Ajustar basado en la amabilidad del personal
        ajuste_amabilidad = (amabilidad_base - 5) * 0.3
        calidad_final = calidad_base + ajuste_amabilidad + random.uniform(-1, 1)
        
        return max(0, min(10, calidad_final))

    def _calcular_fatiga(self, satisfaccion_actual: float) -> float:
        """
        Calcula el nivel de fatiga basado en la satisfacción y otros factores.
        """
        # La fatiga tiende a ser inversa a la satisfacción
        fatiga_base = 10 - satisfaccion_actual
        # Añadir variabilidad
        fatiga_final = fatiga_base + random.uniform(-2, 2)
        
        return max(0, min(10, fatiga_final))

    def _ajustar_por_tipo_lugar(self, impacto_base: float, lugar_tipo: str, satisfaccion: float) -> float:
        """
        Ajusta el impacto basado en el tipo de lugar y contexto.
        """
        if lugar_tipo is None:
            return impacto_base
        
        # Multiplicadores por tipo de lugar
        multiplicadores = {
            'museo': 1.2 if satisfaccion < 6 else 1.0,  # Los museos pueden ser más impactantes si estás aburrido
            'restaurante': 1.3 if satisfaccion < 5 else 1.1,  # La comida siempre ayuda
            'parque': 1.1,  # Experiencia relajante
            'monumento': 1.0,  # Experiencia estándar
            'iglesia': 0.9,  # Puede ser menos impactante para algunos
            'mercado': 1.2,  # Experiencia cultural intensa
            'tienda': 0.8,  # Menos impacto emocional
            'playa': 1.4,  # Muy relajante y positivo
            'mirador': 1.3,  # Vistas impresionantes
            'atraccion': 1.1
        }
        
        multiplicador = multiplicadores.get(lugar_tipo, 1.0)
        return impacto_base * multiplicador

    def _calcular_impacto_fallback(self, amabilidad: float, satisfaccion: float, lugar_tipo: str) -> float:
        """
        Cálculo de impacto inteligente cuando falla el sistema difuso.
        """
        # Lógica simple pero efectiva
        if amabilidad >= 8 and satisfaccion <= 4:
            base_impact = random.uniform(1.5, 2.5)
        elif amabilidad <= 3 and satisfaccion >= 7:
            base_impact = random.uniform(-2.0, -0.5)
        elif amabilidad >= 7:
            base_impact = random.uniform(0.5, 1.5)
        elif amabilidad <= 4:
            base_impact = random.uniform(-1.5, -0.2)
        else:
            base_impact = random.uniform(-0.5, 0.8)
        
        # Ajustar por tipo de lugar
        return self._ajustar_por_tipo_lugar(base_impact, lugar_tipo, satisfaccion)
