from mesa import Agent
from .client import GeminiClient

class AgenteBDI(Agent):
    """
    Agente BDI avanzado con memoria episódica, razonamiento multi-agente, auto-reflexión y estrategias configurables, todo asistido por LLM.
    """
    def __init__(self, unique_id, model, estrategia="equilibrada"):
        try:
            print(f"DEBUG - Inicializando AgenteBDI con unique_id={unique_id}, estrategia={estrategia}")
            # Call Mesa Agent constructor with only model - unique_id is auto-generated
            super().__init__(model)
            # Override the auto-generated unique_id if needed
            if unique_id is not None:
                self.unique_id = unique_id
            self.beliefs = {}
            self.desires = []
            self.intentions = []
            self.plan = []
            self.llm = GeminiClient()
            self.memoria_episodica = []  # [(percepcion, deseo, intencion, accion, resultado)]
            self.estrategia = estrategia  # "exploratoria", "conservadora", "social", "equilibrada"
            print(f"DEBUG - AgenteBDI inicializado exitosamente")
        except Exception as e:
            print(f"ERROR - Fallo en __init__ de AgenteBDI: {str(e)}")
            print(f"ERROR - Tipo de error: {type(e)}")
            raise e

    def percibir(self):
        contexto = self._generar_contexto_percepcion()
        prompt = (
            f"Como agente BDI con memoria, analiza el contexto y resume creencias relevantes. "
            f"Estrategia: {self.estrategia}.\n"
            f"Contexto: {contexto}\n"
            f"Memoria reciente: {self._resumir_memoria_llm()}"
        )
        try:
            resumen = self.llm.generate(prompt)
            self.beliefs['llm_resumen'] = resumen
        except Exception as e:
            self.beliefs['llm_resumen'] = f"Error LLM: {e}"

    def deliberar(self):
        creencias = str(self.beliefs)
        prompt = (
            f"Dado el estado de creencias: {creencias}, y la estrategia '{self.estrategia}', "
            f"sugiere una lista de deseos/objetivos relevantes para un agente turístico. "
            f"Incluye deseos sociales si corresponde. Responde solo con una lista Python."
        )
        try:
            deseos_llm = self.llm.generate(prompt)
            self.desires = self._parsear_lista_llm(deseos_llm)
        except Exception as e:
            self.desires = []

    def filtrar_deseos(self):
        prompt = (
            f"Tengo estos deseos: {self.desires}. Según mis creencias: {self.beliefs}. "
            f"Estrategia: {self.estrategia}. ¿Cuáles son los más alcanzables y prioritarios? "
            f"Responde solo con una lista Python."
        )
        try:
            intenciones_llm = self.llm.generate(prompt)
            self.intentions = self._parsear_lista_llm(intenciones_llm)
        except Exception as e:
            self.intentions = []

    def planificar(self):
        prompt = (
            f"Para cada intención en {self.intentions}, sugiere un plan de acciones concretas (máximo 3 por intención) "
            f"en formato lista de listas Python. Estrategia: {self.estrategia}."
        )
        try:
            plan_llm = self.llm.generate(prompt)
            self.plan = self._parsear_lista_llm(plan_llm)
        except Exception as e:
            self.plan = []

    def actuar(self):
        if self.plan:
            accion = self.plan.pop(0)
            prompt = (
                f"Ejecuta la acción: {accion}. Estrategia: {self.estrategia}. "
                f"¿Qué resultado esperas y cómo deberías proceder? Responde en una frase."
            )
            try:
                resultado = self.llm.generate(prompt)
                self.beliefs['ultimo_resultado'] = resultado
            except Exception as e:
                self.beliefs['ultimo_resultado'] = f"Error LLM: {e}"
            # Guardar episodio en memoria
            self.memoria_episodica.append((self.beliefs.copy(), self.desires.copy(), self.intentions.copy(), accion, self.beliefs['ultimo_resultado']))

    def reflexionar(self):
        """Auto-reflexión asistida por LLM para ajustar deseos/intenciones futuros."""
        prompt = (
            f"Como agente BDI, reflexiona sobre tu último ciclo. "
            f"Memoria reciente: {self._resumir_memoria_llm()}\n"
            f"¿Qué aprendiste y qué deberías cambiar en tus deseos o intenciones? "
            f"Responde con sugerencias en formato lista Python."
        )
        try:
            reflexion = self.llm.generate(prompt)
            sugerencias = self._parsear_lista_llm(reflexion)
            # Opcional: aplicar sugerencias a deseos/intenciones
            if sugerencias:
                self.desires.extend([s for s in sugerencias if s not in self.desires])
        except Exception:
            pass

    def compartir_creencias(self, otros_agentes):
        """Razonamiento multi-agente: compartir creencias relevantes con otros agentes."""
        resumen = self.beliefs.get('llm_resumen', '')
        for agente in otros_agentes:
            if hasattr(agente, 'beliefs'):
                agente.beliefs[f'compartido_{self.unique_id}'] = resumen

    def step(self):
        self.percibir()
        self.deliberar()
        self.filtrar_deseos()
        self.planificar()
        self.actuar()
        self.reflexionar()
        # Ejemplo de razonamiento multi-agente (puedes personalizar la selección de agentes)
        if hasattr(self.model, 'schedule'):
            otros = [a for a in self.model.schedule.agents if a is not self]
            self.compartir_creencias(otros)

    def _generar_contexto_percepcion(self):
        # Extrae contexto relevante del entorno/modelo
        return str(self.model)

    def _resumir_memoria_llm(self, n=3):
        # Resume los últimos n episodios para el LLM
        if not self.memoria_episodica:
            return "Sin memoria."
        resumen = "\n".join([
            f"Episodio {i+1}: percepcion={e[0]}, deseo={e[1]}, intencion={e[2]}, accion={e[3]}, resultado={e[4]}"
            for i, e in enumerate(self.memoria_episodica[-n:])
        ])
        return resumen

    def _parsear_lista_llm(self, texto_llm):
        import ast
        try:
            return ast.literal_eval(texto_llm)
        except Exception:
            return []
