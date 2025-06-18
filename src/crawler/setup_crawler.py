#!/usr/bin/env python3
"""
Script de configuración para el crawler turístico.
Verifica e instala las dependencias necesarias.
"""

import subprocess
import sys
import os
from pathlib import Path


def check_python_version():
    """Verifica que la versión de Python sea compatible."""
    if sys.version_info < (3, 8):
        print("❌ Error: Se requiere Python 3.8 o superior")
        print(f"   Versión actual: {sys.version}")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} - Compatible")
    return True


def install_package(package):
    """Instala un paquete usando pip."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False


def check_and_install_dependencies():
    """Verifica e instala las dependencias necesarias."""
    required_packages = [
        "scrapy>=2.5.0",
        "beautifulsoup4>=4.12.0",
        "sentence-transformers>=2.2.2",
        "chromadb>=0.4.0",
        "numpy>=1.24.0",
        "google-generativeai>=0.3.0",
        "python-dotenv>=1.0.0",
        "requests>=2.31.0",
        "urllib3>=2.0.0"
    ]
    
    print("🔍 Verificando dependencias...")
    
    missing_packages = []
    
    for package in required_packages:
        package_name = package.split(">=")[0]
        try:
            __import__(package_name.replace("-", "_"))
            print(f"✅ {package_name} - Instalado")
        except ImportError:
            print(f"❌ {package_name} - No encontrado")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n📦 Instalando {len(missing_packages)} paquetes faltantes...")
        for package in missing_packages:
            print(f"   Instalando {package}...")
            if install_package(package):
                print(f"   ✅ {package} instalado correctamente")
            else:
                print(f"   ❌ Error instalando {package}")
                return False
    
    print("✅ Todas las dependencias están instaladas")
    return True


def check_environment_variables():
    """Verifica las variables de entorno necesarias."""
    print("\n🔧 Verificando variables de entorno...")
    
    env_file = Path(__file__).parent.parent / ".env"
    
    if not env_file.exists():
        print("⚠️  Archivo .env no encontrado")
        print("💡 Creando archivo .env de ejemplo...")
        
        env_content = """# Configuración del crawler turístico
# API Key para Google Gemini (requerida)
GEMINI_API_KEY=tu_api_key_aqui

# Configuración de ChromaDB (opcional)
CHROMA_DB_PATH=./db/chroma_db

# Configuración de logging (opcional)
LOG_LEVEL=INFO
"""
        
        with open(env_file, 'w', encoding='utf-8') as f:
            f.write(env_content)
        
        print(f"📝 Archivo .env creado en: {env_file}")
        print("⚠️  IMPORTANTE: Configura tu GEMINI_API_KEY en el archivo .env")
        return False
    
    # Verificar si existe la API key
    from dotenv import load_dotenv
    load_dotenv(env_file)
    
    gemini_key = os.getenv('GEMINI_API_KEY')
    if not gemini_key or gemini_key == 'tu_api_key_aqui':
        print("⚠️  GEMINI_API_KEY no configurada correctamente")
        print(f"💡 Edita el archivo {env_file} y configura tu API key")
        return False
    
    print("✅ Variables de entorno configuradas correctamente")
    return True


def check_directory_structure():
    """Verifica que la estructura de directorios sea correcta."""
    print("\n📁 Verificando estructura de directorios...")
    
    base_dir = Path(__file__).parent
    required_dirs = [
        base_dir / "db",
        base_dir.parent / "utils",
        base_dir.parent / "agent_generator"
    ]
    
    for dir_path in required_dirs:
        if not dir_path.exists():
            print(f"📁 Creando directorio: {dir_path}")
            dir_path.mkdir(parents=True, exist_ok=True)
        else:
            print(f"✅ {dir_path.name} - Existe")
    
    return True


def run_basic_tests():
    """Ejecuta pruebas básicas del sistema."""
    print("\n🧪 Ejecutando pruebas básicas...")
    
    try:
        # Test de importaciones
        from crawler.tourist_spider import TouristSpider
        from crawler.config import CRAWLER_CONFIG
        print("✅ Importaciones del spider - OK")
        
        # Test de configuración
        if CRAWLER_CONFIG.get('max_pages'):
            print("✅ Configuración del crawler - OK")
        
        # Test de ChromaDB
        import chromadb
        print("✅ ChromaDB - OK")
        
        # Test de Sentence Transformers
        from sentence_transformers import SentenceTransformer
        print("✅ Sentence Transformers - OK")
        
        print("✅ Todas las pruebas básicas pasaron")
        return True
        
    except Exception as e:
        print(f"❌ Error en las pruebas: {e}")
        return False


def main():
    """Función principal de configuración."""
    print("🚀 Configurando el crawler turístico...")
    print("=" * 50)
    
    # Verificar versión de Python
    if not check_python_version():
        sys.exit(1)
    
    # Verificar e instalar dependencias
    if not check_and_install_dependencies():
        print("❌ Error instalando dependencias")
        sys.exit(1)
    
    # Verificar estructura de directorios
    check_directory_structure()
    
    # Verificar variables de entorno
    env_ok = check_environment_variables()
    
    # Ejecutar pruebas básicas
    if not run_basic_tests():
        print("❌ Las pruebas básicas fallaron")
        sys.exit(1)
    
    print("\n" + "=" * 50)
    if env_ok:
        print("🎉 ¡Configuración completada exitosamente!")
        print("\n💡 Ahora puedes ejecutar el crawler con:")
        print("   python main.py --help")
        print("   python main.py --list-cities")
        print("   python main.py -c Barcelona")
    else:
        print("⚠️  Configuración parcialmente completada")
        print("🔧 Configura las variables de entorno antes de ejecutar el crawler")


if __name__ == "__main__":
    main()