"""
Script para mostrar un resumen final de la sesión de API de Gemini
Puede ser llamado al final de la ejecución de la aplicación
"""

import atexit
import os
from .gemini_api_counter import api_counter

def print_session_summary():
    """Imprime un resumen final de la sesión"""
    try:
        stats = api_counter.get_stats()
        
        if stats['call_counts']['total_calls'] > 0:
            print("\n" + "="*80)
            print("🎯 RESUMEN FINAL DE LA SESIÓN - API GEMINI")
            print("="*80)
            
            print(f"📊 ESTADÍSTICAS GENERALES:")
            print(f"   • Total de llamadas realizadas: {stats['call_counts']['total_calls']}")
            print(f"   • Llamadas exitosas: {stats['call_counts']['successful_calls']}")
            print(f"   • Llamadas fallidas: {stats['call_counts']['failed_calls']}")
            print(f"   • Tasa de éxito: {stats['call_counts']['success_rate']:.1f}%")
            
            print(f"\n💾 USO DE DATOS:")
            print(f"   • Caracteres enviados: {stats['data_usage']['total_prompt_characters']:,}")
            print(f"   • Caracteres recibidos: {stats['data_usage']['total_response_characters']:,}")
            print(f"   • Total de caracteres: {stats['data_usage']['total_characters']:,}")
            
            print(f"\n⏱️ TIEMPOS:")
            print(f"   • Tiempo total en API: {stats['timing']['total_api_time_formatted']}")
            print(f"   • Duración de la sesión: {stats['session_info']['session_duration_formatted']}")
            print(f"   • Promedio por llamada: {stats['timing']['average_call_duration']:.2f}s")
            print(f"   • Llamadas por minuto: {stats['timing']['calls_per_minute']:.1f}")
            
            print(f"\n🔍 DESGLOSE POR COMPONENTE:")
            for module, functions in stats['caller_breakdown'].items():
                module_name = module.split('.')[-1] if '.' in module else module
                total_module_calls = sum(functions.values())
                print(f"   • {module_name}: {total_module_calls} llamadas")
                for function, count in functions.items():
                    if count > 0:
                        print(f"     └─ {function}: {count}")
            
            # Calcular costo estimado (aproximado)
            total_chars = stats['data_usage']['total_characters']
            estimated_tokens = total_chars / 4  # Aproximación: 1 token ≈ 4 caracteres
            
            print(f"\n💰 ESTIMACIÓN DE COSTO (APROXIMADO):")
            print(f"   • Tokens estimados: {estimated_tokens:,.0f}")
            print(f"   • Modelo usado: Gemini 2.0 Flash")
            print(f"   • Nota: Para costo exacto, consulta tu dashboard de Google AI Studio")
            
            print("="*80)
            print("✅ Sesión completada. ¡Gracias por usar el Planificador Turístico!")
            print("="*80)
            
    except Exception as e:
        print(f"Error al generar resumen de sesión: {e}")

def export_session_data(filepath: str = None):
    """Exporta los datos de la sesión a un archivo JSON"""
    try:
        if filepath is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"gemini_session_{timestamp}.json"
        
        api_counter.export_to_json(filepath)
        print(f"📄 Datos de la sesión exportados a: {filepath}")
        
    except Exception as e:
        print(f"Error al exportar datos de sesión: {e}")

def setup_session_cleanup():
    """Configura la limpieza automática al final de la sesión"""
    atexit.register(print_session_summary)

# Configurar automáticamente la limpieza de sesión
setup_session_cleanup()