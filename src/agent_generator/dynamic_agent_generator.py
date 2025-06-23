"""
Generador dinámico de agentes usando LLM.
Permite crear tipos de agentes específicos para cada lugar sin mapeos predefinidos.
"""

import json
import os
from typing import List, Dict, Optional
from agent_generator.mistral_client import MistralClient


class DynamicAgentGenerator:
    """
    Generador de agentes dinámico que usa LLM para determinar qué tipos de personal
    serían apropiados para cada lugar turístico específico.
    """
    
    def __init__(self, cache_file: str = "agent_cache.json"):
        """
        Inicializa el generador dinámico de agentes.
        
        :param cache_file: Archivo para cachear respuestas del LLM
        """
        self.llm_client = MistralClient()
        self.cache_file = cache_file
        self.cache = self._load_cache()
        
    def _load_cache(self) -> Dict:
        """Carga el cache de agentes desde archivo."""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"DEBUG - Error cargando cache de agentes: {e}")
        return {}
    
    def _save_cache(self):
        """Guarda el cache de agentes en archivo."""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"DEBUG - Error guardando cache de agentes: {e}")
    
    def _create_cache_key(self, place_name: str, place_type: str, place_description: str) -> str:
        """Crea una clave única para el cache basada en los datos del lugar."""
        # Usar hash de los datos principales para crear clave única
        key_data = f"{place_name}_{place_type}_{place_description[:100]}"
        return str(hash(key_data))
    
    def generate_agents_for_place(self, place_name: str, place_type: str, 
                                place_description: str, force_refresh: bool = False) -> List[str]:
        """
        Genera tipos de agentes para un lugar específico usando LLM.
        
        :param place_name: Nombre del lugar
        :param place_type: Tipo de lugar (museo, restaurante, etc.)
        :param place_description: Descripción del lugar
        :param force_refresh: Si True, ignora el cache y genera nuevos agentes
        :return: Lista de tipos de agentes
        """
        # Verificar cache primero
        cache_key = self._create_cache_key(place_name, place_type, place_description)
        
        if not force_refresh and cache_key in self.cache:
            cached_agents = self.cache[cache_key]
            print(f"DEBUG - Agentes desde cache para {place_name}: {cached_agents}")
            return cached_agents
        
        # Generar nuevos agentes con LLM
        try:
            agents = self._generate_with_llm(place_name, place_type, place_description)
            
            if agents:
                # Guardar en cache
                self.cache[cache_key] = agents
                self._save_cache()
                print(f"DEBUG - Agentes generados y cacheados para {place_name}: {agents}")
                return agents
            else:
                print(f"DEBUG - LLM no generó agentes válidos para {place_name}")
                
        except Exception as e:
            print(f"DEBUG - Error generando agentes para {place_name}: {e}")
        
        # Fallback a agentes por defecto
        fallback_agents = self._get_fallback_agents(place_type)
        print(f"DEBUG - Usando agentes fallback para {place_name}: {fallback_agents}")
        return fallback_agents
    
    def _generate_with_llm(self, place_name: str, place_type: str, place_description: str) -> List[str]:
        """
        Genera agentes usando LLM con prompt optimizado.
        """
        prompt = self._create_optimized_prompt(place_name, place_type, place_description)
        
        try:
            response = self.llm_client.generate(prompt)
            
            if response and response not in ["[Error de generación]", "[Límite de solicitudes excedido]"]:
                agents = self._parse_and_validate_response(response)
                return agents
            else:
                print(f"DEBUG - LLM falló para {place_name}: {response}")
                return []
                
        except Exception as e:
            print(f"DEBUG - Error en llamada LLM para {place_name}: {e}")
            return []
    
    def _create_optimized_prompt(self, place_name: str, place_type: str, place_description: str) -> str:
        """
        Crea un prompt optimizado para generar agentes específicos.
        """
        prompt = f"""
Analiza este lugar turístico específico y determina qué personal trabajaría allí de forma realista.

LUGAR: {place_name}
TIPO: {place_type}
DESCRIPCIÓN: {place_description}

INSTRUCCIONES:
1. Considera el lugar específico, no solo el tipo general
2. Piensa en quién interactuaría naturalmente con los turistas
3. Incluye tanto personal obvio como roles únicos del lugar
4. Sugiere 2-3 roles máximo

CONTEXTO DE ROLES:
- Lugares culturales: guía, curador, historiador, conservador
- Lugares gastronómicos: mesero, chef, sommelier, barista
- Lugares comerciales: vendedor, artesano, comerciante
- Lugares naturales: guía, jardinero, guardaparques, biólogo
- Lugares religiosos: sacerdote, guía, organista, voluntario
- Lugares de entretenimiento: artista, músico, animador, técnico

FORMATO DE RESPUESTA:
Responde ÚNICAMENTE con los roles separados por comas, sin explicaciones.

Ejemplo: guía, curador, conservador

Respuesta:"""
        
        return prompt
    
    def _parse_and_validate_response(self, response: str) -> List[str]:
        """
        Parsea y valida la respuesta del LLM.
        """
        # Limpiar respuesta
        cleaned = response.strip()
        
        # Buscar la línea con los roles
        lines = cleaned.split('\n')
        roles_line = ""
        
        for line in lines:
            line = line.strip()
            if line and ',' in line and not line.startswith(('Ejemplo:', 'Respuesta:', 'FORMATO')):
                roles_line = line
                break
        
        if not roles_line:
            # Si no hay comas, puede ser un solo rol
            roles_line = lines[0] if lines else ""
        
        # Extraer y validar roles
        agents = []
        parts = roles_line.split(',')
        
        for part in parts:
            agent = self._clean_and_normalize_agent(part.strip())
            if self._is_valid_agent(agent) and agent not in agents:
                agents.append(agent)
        
        return agents[:3]  # Máximo 3 agentes
    
    def _clean_and_normalize_agent(self, agent: str) -> str:
        """
        Limpia y normaliza un nombre de agente.
        """
        # Convertir a minúsculas y limpiar
        agent = agent.lower().strip()
        
        # Remover caracteres especiales
        for char in ['-', '*', '•', ':', '(', ')', '[', ']']:
            agent = agent.replace(char, '')
        
        agent = agent.strip()
        
        # Normalizar roles comunes
        normalizations = {
            'guia': 'guía',
            'guía turístico': 'guía',
            'guía turistico': 'guía',
            'camarero': 'mesero',
            'mozo': 'mesero',
            'cocinero': 'chef',
            'dependiente': 'vendedor',
            'comerciante': 'vendedor',
            'recepcionista': 'recepcionista',
            'conserje': 'asistente',
            'cuidador': 'jardinero',
            'experto': 'especialista',
            'padre': 'sacerdote',
            'cura': 'sacerdote',
            'socorrista': 'salvavidas',
            'fotografo': 'fotógrafo',
            'seguridad': 'vigilante',
            'guardia': 'vigilante',
            'musico': 'músico',
            'ayudante': 'asistente',
            'conservador': 'curador',
            'guardaparques': 'guardaparque',
            'biologo': 'biólogo',
            'tecnico': 'técnico'
        }
        
        return normalizations.get(agent, agent)
    
    def _is_valid_agent(self, agent: str) -> bool:
        """
        Valida si un agente es válido.
        """
        if not agent or len(agent) < 3 or len(agent) > 25:
            return False
        
        # Lista de roles válidos conocidos
        valid_roles = {
            'guía', 'curador', 'mesero', 'chef', 'vendedor', 'recepcionista',
            'jardinero', 'historiador', 'sacerdote', 'salvavidas', 'fotógrafo',
            'asistente', 'vigilante', 'artista', 'músico', 'especialista',
            'conservador', 'guardaparque', 'biólogo', 'técnico', 'sommelier',
            'barista', 'artesano', 'comerciante', 'organista', 'voluntario',
            'animador', 'instructor', 'monitor', 'coordinador'
        }
        
        # Verificar si es un rol conocido o si parece válido
        if agent in valid_roles:
            return True
        
        # Verificar patrones válidos
        if any(pattern in agent for pattern in ['guía', 'ista', 'ero', 'or', 'nte']):
            return True
        
        return False
    
    def _get_fallback_agents(self, place_type: str) -> List[str]:
        """
        Agentes de fallback cuando falla la generación dinámica.
        """
        fallback_mapping = {
            'museo': ['guía', 'curador'],
            'restaurante': ['mesero', 'chef'],
            'parque': ['guía', 'jardinero'],
            'monumento': ['guía', 'historiador'],
            'iglesia': ['guía', 'sacerdote'],
            'mercado': ['vendedor', 'guía'],
            'tienda': ['vendedor', 'asistente'],
            'playa': ['salvavidas', 'guía'],
            'mirador': ['guía', 'fotógrafo'],
            'atraccion': ['guía', 'asistente'],
            'galeria': ['curador', 'artista'],
            'teatro': ['guía', 'técnico'],
            'biblioteca': ['bibliotecario', 'asistente'],
            'universidad': ['guía', 'profesor'],
            'hospital': ['recepcionista', 'asistente'],
            'estacion': ['asistente', 'vigilante']
        }
        
        return fallback_mapping.get(place_type, ['guía', 'asistente'])
    
    def get_agent_statistics(self) -> Dict:
        """
        Obtiene estadísticas del cache de agentes.
        """
        if not self.cache:
            return {'total_places': 0, 'unique_agents': set(), 'most_common_agents': {}}
        
        all_agents = []
        for agents_list in self.cache.values():
            all_agents.extend(agents_list)
        
        unique_agents = set(all_agents)
        agent_counts = {}
        for agent in all_agents:
            agent_counts[agent] = agent_counts.get(agent, 0) + 1
        
        # Ordenar por frecuencia
        most_common = sorted(agent_counts.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'total_places': len(self.cache),
            'unique_agents': unique_agents,
            'most_common_agents': dict(most_common[:10]),
            'total_agent_instances': len(all_agents)
        }
    
    def clear_cache(self):
        """Limpia el cache de agentes."""
        self.cache = {}
        if os.path.exists(self.cache_file):
            os.remove(self.cache_file)
        print("DEBUG - Cache de agentes limpiado")


# Instancia global para uso en el proyecto
dynamic_agent_generator = DynamicAgentGenerator()