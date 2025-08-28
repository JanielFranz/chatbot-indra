
# Configuración automática para pytesseract
import pytesseract

# Configurar ruta de Tesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

print("Tesseract configurado correctamente")
