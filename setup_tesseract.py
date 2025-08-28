"""
Script para configurar Tesseract OCR en Windows
"""
import os
import sys
import urllib.request
import subprocess
from pathlib import Path

def check_tesseract():
    """Verifica si Tesseract está instalado y accesible"""
    try:
        result = subprocess.run(['tesseract', '--version'],
                              capture_output=True, text=True, check=True)
        print("✅ Tesseract ya está instalado:")
        print(result.stdout.split('\n')[0])
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Tesseract no está instalado o no está en PATH")
        return False

def configure_pytesseract():
    """Configura pytesseract para usar Tesseract en Windows"""
    # Rutas comunes donde se instala Tesseract en Windows
    possible_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        r"C:\Users\{}\AppData\Local\Tesseract-OCR\tesseract.exe".format(os.getenv('USERNAME')),
        r"C:\tesseract\tesseract.exe"
    ]

    tesseract_path = None
    for path in possible_paths:
        if os.path.exists(path):
            tesseract_path = path
            break

    if tesseract_path:
        print(f"✅ Tesseract encontrado en: {tesseract_path}")

        # Crear archivo de configuración para pytesseract
        config_content = f'''
# Configuración automática para pytesseract
import pytesseract

# Configurar ruta de Tesseract
pytesseract.pytesseract.tesseract_cmd = r"{tesseract_path}"

print("Tesseract configurado correctamente")
'''

        with open('tesseract_config.py', 'w') as f:
            f.write(config_content)

        print("✅ Archivo de configuración creado: tesseract_config.py")
        return tesseract_path
    else:
        print("❌ Tesseract no encontrado en rutas comunes")
        return None

def main():
    print("🔍 Verificando instalación de Tesseract OCR...")

    if check_tesseract():
        print("\n✅ Tesseract está correctamente configurado")
        return

    print("\n📥 Tesseract no está instalado. Opciones:")
    print("1. Descargar e instalar manualmente desde:")
    print("   https://github.com/UB-Mannheim/tesseract/wiki")
    print("\n2. Instalar usando Chocolatey (si lo tienes):")
    print("   choco install tesseract")
    print("\n3. Instalar usando winget:")
    print("   winget install UB-Mannheim.TesseractOCR")

    # Intentar encontrar Tesseract si ya está instalado
    tesseract_path = configure_pytesseract()

    if not tesseract_path:
        print("\n⚠️  Para usar OCR, instala Tesseract y vuelve a ejecutar este script")
        print("Mientras tanto, el sistema funcionará sin OCR")

if __name__ == "__main__":
    main()
