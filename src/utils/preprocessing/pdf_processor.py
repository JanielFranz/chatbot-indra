import os
import re
import fitz  # PyMuPDF
from PIL import Image
import io
from typing import List, Dict, Any, Tuple
import logging

from src.app.ingestion.models import ExtractedContent

# Para OCR cuando el PDF no tiene texto extraíble
try:
    import pytesseract
    # Configurar ruta de Tesseract en Windows
    tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(tesseract_path):
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
    HAS_OCR = True
except ImportError:
    pytesseract = None
    HAS_OCR = False
    print("⚠️  pytesseract no está disponible. PDFs con solo imágenes no podrán procesarse con OCR.")


class PDFPreprocessor:
    """
    Clase para extraer y procesar contenido de PDFs para RAG multimodal.
    Extrae texto e imágenes, genera chunks y prepara embeddings.
    Incluye soporte para OCR cuando el PDF contiene solo imágenes.
    """

    def __init__(self,
                 chunk_size: int = 500,
                 chunk_overlap: int = 50,
                 use_ocr: bool = True):
        """
        Inicializa el preprocesador de PDFs.

        Args:
            chunk_size (int): Tamaño máximo de cada chunk de texto
            chunk_overlap (int): Solapamiento entre chunks
            use_ocr (bool): Si usar OCR para PDFs con solo imágenes
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_ocr = use_ocr and HAS_OCR

        # Configurar logging
        self.logger = logging.getLogger(__name__)

        # Directorio para guardar imágenes extraídas
        self.images_dir = "extracted_images"
        os.makedirs(self.images_dir, exist_ok=True)

    def extract_content_from_pdf(self, pdf_path: str) -> ExtractedContent:
        """
        Extrae todo el contenido (texto e imágenes) de un PDF.
        Si el PDF no tiene texto extraíble, usa OCR.

        Args:
            pdf_path (str): Ruta al archivo PDF

        Returns:
            ExtractedContent: Contenido extraído y procesado
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Archivo PDF no encontrado: {pdf_path}")

        self.logger.info(f"Extrayendo contenido de: {pdf_path}")

        # Abrir PDF
        doc = fitz.open(pdf_path)

        # Extraer texto e imágenes
        all_text = []
        images = []
        page_texts = {}
        total_text_length = 0

        for page_num in range(len(doc)):
            page = doc[page_num]

            # Extraer texto de la página
            page_text = page.get_text()
            total_text_length += len(page_text.strip())
            self.logger.info(f"Texto extraído sin OCR: {page_text.strip()}")

            # Si no hay texto y OCR está habilitado, intentar OCR
            if len(page_text.strip()) == 0:
                self.logger.info(f"Página {page_num} sin texto extraíble, intentando OCR...")
                page_text = self._extract_text_with_ocr(page)
                total_text_length += len(page_text.strip())

            page_texts[page_num] = page_text
            all_text.append(page_text)

            # Extraer imágenes de la página
            page_images = self._extract_images_from_page(page, page_num)
            images.extend(page_images)

        doc.close()

        self.logger.info(f"Total de texto extraído: {total_text_length} caracteres")

        # Si no se extrajo texto, crear contenido básico con descripciones de imágenes
        if total_text_length == 0:
            self.logger.warning("No se pudo extraer texto del PDF. Creando contenido basado en imágenes.")
            text_chunks, metadata = self._create_image_based_content(images)
        else:
            # Procesar texto en chunks
            text_chunks, metadata = self._create_text_chunks(all_text, page_texts)

        # Asociar imágenes con chunks de texto más relevantes
        self._associate_images_with_chunks(text_chunks, images, metadata)

        self.logger.info(f"Extraídos {len(text_chunks)} chunks de texto y {len(images)} imágenes")

        return ExtractedContent(
            text_chunks=text_chunks,
            images=images,
            metadata=metadata
        )

    def _extract_text_with_ocr(self, page) -> str:
        """Extrae texto de una página usando OCR."""
        if not self.use_ocr or not HAS_OCR:
            return ""

        try:
            # Convertir página a imagen
            mat = fitz.Matrix(2.0, 2.0)  # Aumentar resolución para mejor OCR
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            pil_image = Image.open(io.BytesIO(img_data))

            # Aplicar OCR
            text = pytesseract.image_to_string(pil_image, lang='eng+spa')

            pix = None  # Liberar memoria

            return text.strip()

        except Exception as e:
            self.logger.warning(f"Error en OCR: {e}")
            return ""

    def _create_image_based_content(self, images: List[Dict[str, Any]]) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Crea contenido de texto basado en las imágenes cuando no hay texto extraíble."""
        chunks = []
        metadata = []

        if not images:
            # Si no hay imágenes ni texto, crear contenido mínimo
            chunks = ["Este es un documento PDF que no contiene texto extraíble ni imágenes procesables."]
            metadata = [{
                "page_number": 0,
                "chunk_id": "default_chunk_0",
                "chunk_index": 0,
                "total_chunks_in_page": 1,
                "associated_images": []
            }]
            return chunks, metadata

        # Agrupar imágenes por página
        pages_with_images = {}
        for img in images:
            page_num = img["page"]
            if page_num not in pages_with_images:
                pages_with_images[page_num] = []
            pages_with_images[page_num].append(img)

        # Crear un chunk por página que contiene imágenes
        for page_num, page_images in pages_with_images.items():
            chunk_text = f"Página {page_num + 1} contiene {len(page_images)} imagen(es). "
            chunk_text += f"Esta página forma parte de un documento PDF con contenido visual. "
            chunk_text += f"Las imágenes pueden contener información importante como gráficos, diagramas, o texto escaneado."

            chunks.append(chunk_text)
            metadata.append({
                "page_number": page_num,
                "chunk_id": f"image_page_{page_num}_chunk_0",
                "chunk_index": 0,
                "total_chunks_in_page": 1,
                "associated_images": page_images
            })

        return chunks, metadata

    def _extract_images_from_page(self, page, page_num: int) -> List[Dict[str, Any]]:
        """Extrae todas las imágenes de una página específica."""
        images = []
        image_list = page.get_images()

        for img_index, img in enumerate(image_list):
            try:
                # Obtener datos de la imagen
                xref = img[0]
                pix = fitz.Pixmap(page.parent, xref)

                # Convertir a PIL Image
                if pix.n - pix.alpha < 4:  # Solo RGB o escala de grises
                    img_data = pix.tobytes("png")
                    pil_image = Image.open(io.BytesIO(img_data))

                    # Guardar imagen en disco
                    img_filename = f"page_{page_num}_img_{img_index}.png"
                    img_path = os.path.join(self.images_dir, img_filename)
                    pil_image.save(img_path)

                    # Intentar obtener bbox de la imagen en la página
                    try:
                        bbox = page.get_image_bbox(img)
                    except:
                        bbox = (0, 0, pil_image.width, pil_image.height)

                    images.append({
                        "image": pil_image,
                        "page": page_num,
                        "bbox": bbox,
                        "image_path": img_path,
                        "filename": img_filename,
                        "size": pil_image.size
                    })

                pix = None  # Liberar memoria

            except Exception as e:
                self.logger.warning(f"Error extrayendo imagen {img_index} de página {page_num}: {e}")
                continue

        return images

    def _create_text_chunks(self, all_text: List[str], page_texts: Dict[int, str]) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Divide el texto en chunks manejables con metadatos."""
        chunks = []
        metadata = []

        for page_num, page_text in page_texts.items():
            if not page_text.strip():
                continue

            # Limpiar texto
            cleaned_text = self._clean_text(page_text)

            # Dividir en chunks
            page_chunks = self._split_text_into_chunks(cleaned_text)

            for chunk_idx, chunk in enumerate(page_chunks):
                if len(chunk.strip()) < 50:  # Ignorar chunks muy cortos
                    continue

                chunks.append(chunk)
                metadata.append({
                    "page_number": page_num,
                    "chunk_id": f"page_{page_num}_chunk_{chunk_idx}",
                    "chunk_index": chunk_idx,
                    "total_chunks_in_page": len(page_chunks),
                    "associated_images": []  # Se llenará después
                })

                self.logger.info(f"Chunk creado: {metadata[-1]['chunk_id']} (Página {page_num}, Índice {chunk_idx})")

        return chunks, metadata

    def _clean_text(self, text: str) -> str:
        """Limpia y normaliza el texto extraído."""
        # Remover caracteres de control y espacios extras
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\"\'\/]', '', text)

        # Remover líneas muy cortas (probablemente headers/footers)
        lines = text.split('\n')
        cleaned_lines = [line.strip() for line in lines if len(line.strip()) > 3]

        return '\n'.join(cleaned_lines).strip()

    def _split_text_into_chunks(self, text: str) -> List[str]:
        """Divide texto en chunks con solapamiento."""
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size

            if end >= len(text):
                chunks.append(text[start:])
                break

            # Buscar el último punto o salto de línea antes del límite
            chunk_end = end
            for i in range(end, start + self.chunk_size // 2, -1):
                if text[i] in '.!?\n':
                    chunk_end = i + 1
                    break

            chunks.append(text[start:chunk_end].strip())
            start = chunk_end - self.chunk_overlap

        return [chunk for chunk in chunks if chunk.strip()]

    def _associate_images_with_chunks(self,
                                    text_chunks: List[str],
                                    images: List[Dict[str, Any]],
                                    metadata: List[Dict[str, Any]]):
        """Asocia imágenes con chunks de texto basándose en la página."""
        # Crear mapeo de página a imágenes
        page_to_images = {}
        for img in images:
            page_num = img["page"]
            if page_num not in page_to_images:
                page_to_images[page_num] = []
            page_to_images[page_num].append(img)

        # Asociar imágenes con chunks de la misma página
        for i, meta in enumerate(metadata):
            page_num = meta["page_number"]
            if page_num in page_to_images:
                meta["associated_images"] = page_to_images[page_num]

    def get_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas del preprocessor."""
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "use_ocr": self.use_ocr,
            "has_ocr": HAS_OCR,
            "images_directory": self.images_dir
        }
