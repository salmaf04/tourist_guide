"""
Sistema de conteo de llamadas a la API de LLM (Gemini/Mistral)
Intercepta y cuenta todas las llamadas realizadas durante la ejecución de la aplicación
"""

import time
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from threading import Lock
import functools

@dataclass
class APICall:
    """Representa una llamada individual a la API"""
    timestamp: str
    prompt_length: int
    response_length: int
    system_prompt: Optional[str]
    duration_seconds: float
    success: bool
    error_message: Optional[str] = None
    caller_module: Optional[str] = None
    caller_function: Optional[str] = None

class GeminiAPICounter:
    """
    Contador global de llamadas a la API de Gemini
    Singleton para mantener el estado global
    """
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.calls: List[APICall] = []
            self.session_start = datetime.now()
            self.total_calls = 0
            self.successful_calls = 0
            self.failed_calls = 0
            self.total_prompt_chars = 0
            self.total_response_chars = 0
            self.total_duration = 0.0
            self._initialized = True
    
    def record_call(self, 
                   prompt: str, 
                   response: str, 
                   system_prompt: Optional[str] = None,
                   duration: float = 0.0,
                   success: bool = True,
                   error_message: Optional[str] = None,
                   caller_info: Optional[Dict[str, str]] = None) -> None:
        """Registra una llamada a la API"""
        
        call = APICall(
            timestamp=datetime.now().isoformat(),
            prompt_length=len(prompt) if prompt else 0,
            response_length=len(response) if response else 0,
            system_prompt=system_prompt,
            duration_seconds=duration,
            success=success,
            error_message=error_message,
            caller_module=caller_info.get('module') if caller_info else None,
            caller_function=caller_info.get('function') if caller_info else None
        )
        
        with self._lock:
            self.calls.append(call)
            self.total_calls += 1
            
            if success:
                self.successful_calls += 1
            else:
                self.failed_calls += 1
            
            self.total_prompt_chars += call.prompt_length
            self.total_response_chars += call.response_length
            self.total_duration += duration
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas completas de las llamadas"""
        with self._lock:
            session_duration = (datetime.now() - self.session_start).total_seconds()
            
            stats = {
                'session_info': {
                    'start_time': self.session_start.isoformat(),
                    'current_time': datetime.now().isoformat(),
                    'session_duration_seconds': session_duration,
                    'session_duration_formatted': self._format_duration(session_duration)
                },
                'call_counts': {
                    'total_calls': self.total_calls,
                    'successful_calls': self.successful_calls,
                    'failed_calls': self.failed_calls,
                    'success_rate': (self.successful_calls / max(1, self.total_calls)) * 100
                },
                'data_usage': {
                    'total_prompt_characters': self.total_prompt_chars,
                    'total_response_characters': self.total_response_chars,
                    'total_characters': self.total_prompt_chars + self.total_response_chars,
                    'average_prompt_length': self.total_prompt_chars / max(1, self.total_calls),
                    'average_response_length': self.total_response_chars / max(1, self.total_calls)
                },
                'timing': {
                    'total_api_time_seconds': self.total_duration,
                    'total_api_time_formatted': self._format_duration(self.total_duration),
                    'average_call_duration': self.total_duration / max(1, self.total_calls),
                    'calls_per_minute': (self.total_calls / max(1, session_duration)) * 60
                },
                'caller_breakdown': self._get_caller_breakdown()
            }
            
            return stats
    
    def _get_caller_breakdown(self) -> Dict[str, Dict[str, int]]:
        """Analiza las llamadas por m��dulo y función"""
        breakdown = {}
        
        for call in self.calls:
            module = call.caller_module or 'unknown'
            function = call.caller_function or 'unknown'
            
            if module not in breakdown:
                breakdown[module] = {}
            
            if function not in breakdown[module]:
                breakdown[module][function] = 0
            
            breakdown[module][function] += 1
        
        return breakdown
    
    def _format_duration(self, seconds: float) -> str:
        """Formatea duración en formato legible"""
        if seconds < 60:
            return f"{seconds:.2f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"
    
    def get_detailed_calls(self) -> List[Dict[str, Any]]:
        """Obtiene lista detallada de todas las llamadas"""
        with self._lock:
            return [asdict(call) for call in self.calls]
    
    def export_to_json(self, filepath: str) -> None:
        """Exporta estadísticas y llamadas a archivo JSON"""
        data = {
            'stats': self.get_stats(),
            'detailed_calls': self.get_detailed_calls()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def reset(self) -> None:
        """Reinicia el contador"""
        with self._lock:
            self.calls.clear()
            self.session_start = datetime.now()
            self.total_calls = 0
            self.successful_calls = 0
            self.failed_calls = 0
            self.total_prompt_chars = 0
            self.total_response_chars = 0
            self.total_duration = 0.0
    
    def print_summary(self) -> None:
        """Imprime un resumen de las estadísticas"""
        stats = self.get_stats()
        
        print("\n" + "="*60)
        print("📊 RESUMEN DE LLAMADAS A LA API DE LLM")
        print("="*60)
        
        print(f"🕐 Sesión iniciada: {stats['session_info']['start_time']}")
        print(f"⏱️  Duración de sesión: {stats['session_info']['session_duration_formatted']}")
        
        print(f"\n📞 LLAMADAS:")
        print(f"   Total: {stats['call_counts']['total_calls']}")
        print(f"   Exitosas: {stats['call_counts']['successful_calls']}")
        print(f"   Fallidas: {stats['call_counts']['failed_calls']}")
        print(f"   Tasa de éxito: {stats['call_counts']['success_rate']:.1f}%")
        
        print(f"\n📊 USO DE DATOS:")
        print(f"   Caracteres enviados: {stats['data_usage']['total_prompt_characters']:,}")
        print(f"   Caracteres recibidos: {stats['data_usage']['total_response_characters']:,}")
        print(f"   Total caracteres: {stats['data_usage']['total_characters']:,}")
        print(f"   Promedio por llamada: {stats['data_usage']['average_prompt_length']:.0f} → {stats['data_usage']['average_response_length']:.0f}")
        
        print(f"\n⏰ TIEMPOS:")
        print(f"   Tiempo total en API: {stats['timing']['total_api_time_formatted']}")
        print(f"   Promedio por llamada: {stats['timing']['average_call_duration']:.2f}s")
        print(f"   Llamadas por minuto: {stats['timing']['calls_per_minute']:.1f}")
        
        print(f"\n🔍 DESGLOSE POR MÓDULO:")
        for module, functions in stats['caller_breakdown'].items():
            print(f"   {module}:")
            for function, count in functions.items():
                print(f"     └─ {function}: {count} llamadas")
        
        print("="*60)

def get_caller_info() -> Dict[str, str]:
    """Obtiene información del caller usando el stack de llamadas"""
    import inspect
    
    # Buscar en el stack el primer frame que no sea de este módulo
    current_frame = inspect.currentframe()
    try:
        frame = current_frame.f_back.f_back  # Saltar este frame y el del decorator
        while frame:
            module_name = frame.f_globals.get('__name__', 'unknown')
            function_name = frame.f_code.co_name
            
            # Saltar frames internos del sistema de conteo
            if not module_name.endswith('gemini_api_counter') and not module_name.endswith('client'):
                return {
                    'module': module_name,
                    'function': function_name
                }
            frame = frame.f_back
        
        return {'module': 'unknown', 'function': 'unknown'}
    finally:
        del current_frame

def count_gemini_calls(func):
    """
    Decorator para contar automáticamente las llamadas a métodos de LLM (GeminiClient/MistralClient)
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        counter = GeminiAPICounter()
        start_time = time.time()
        
        # Obtener información del caller
        caller_info = get_caller_info()
        
        try:
            # Ejecutar la función original
            result = func(*args, **kwargs)
            
            # Extraer información de la llamada
            prompt = ""
            system_prompt = None
            
            if len(args) > 1:  # args[0] es self, args[1] debería ser prompt
                prompt = str(args[1])
            if len(args) > 2:  # args[2] podría ser system prompt
                system_prompt = str(args[2])
            elif 'system' in kwargs:
                system_prompt = str(kwargs['system'])
            
            duration = time.time() - start_time
            
            # Determinar si fue exitosa
            success = result not in ["[Error de generación]", "[Límite de solicitudes excedido]", "[Respuesta no disponible]"]
            error_message = result if not success else None
            
            # Registrar la llamada
            counter.record_call(
                prompt=prompt,
                response=result if success else "",
                system_prompt=system_prompt,
                duration=duration,
                success=success,
                error_message=error_message,
                caller_info=caller_info
            )
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            
            # Registrar llamada fallida
            prompt = str(args[1]) if len(args) > 1 else ""
            system_prompt = str(args[2]) if len(args) > 2 else kwargs.get('system')
            
            counter.record_call(
                prompt=prompt,
                response="",
                system_prompt=system_prompt,
                duration=duration,
                success=False,
                error_message=str(e),
                caller_info=caller_info
            )
            
            raise e
    
    return wrapper

# Instancia global del contador
api_counter = GeminiAPICounter()