#!/usr/bin/env python3
"""
Script independiente para verificar el uso de la API de Gemini
Ejecuta este script para ver estadísticas actuales de las llamadas a la API
"""

import sys
import os

# Añadir el directorio src al path para importar módulos
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from utils.gemini_api_counter import api_counter
from utils.api_session_summary import export_session_data
import argparse

def main():
    parser = argparse.ArgumentParser(description='Monitor de uso de API de Gemini')
    parser.add_argument('--export', '-e', type=str, help='Exportar datos a archivo JSON')
    parser.add_argument('--reset', '-r', action='store_true', help='Resetear contador de API')
    parser.add_argument('--quiet', '-q', action='store_true', help='Modo silencioso (solo exportar)')
    
    args = parser.parse_args()
    
    if args.reset:
        api_counter.reset()
        if not args.quiet:
            print("✅ Contador de API reseteado.")
        return
    
    if not args.quiet:
        # Mostrar estadísticas completas
        api_counter.print_summary()
    
    if args.export:
        export_session_data(args.export)
        if not args.quiet:
            print(f"📄 Datos exportados a: {args.export}")

if __name__ == "__main__":
    main()