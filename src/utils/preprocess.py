import os
import re
import fitz  # PyMuPDF
from PIL import Image
import io
from typing import List, Dict, Any, Tuple
import logging
from sentence_transformers import SentenceTransformer

from src.ingestion.model import ExtractedContent


class PDFPreprocessor:
    """
    Clase para extraer y procesar contenido de PDFs para RAG multimodal.
    Extrae texto e imágenes, genera chunks y prepara embeddings.
    """

    def __init__(self,
                 chunk_size: int = 500,
                 chunk_overlap: int = 50,
                 embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Inicializa el preprocesador de PDFs.

        Args:
            chunk_size (int): Tamaño máximo de cada chunk de texto
            chunk_overlap (int): Solapamiento entre chunks
            embedding_model (str): Modelo para generar embeddings
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model_name = embedding_model

        # Inicializar modelo de embeddings
        self.embedding_model = SentenceTransformer(embedding_model)

        # Configurar logging
        self.logger = logging.getLogger(__name__)

        # Directorio para guardar imágenes extraídas
        self.images_dir = "extracted_images"
        os.makedirs(self.images_dir, exist_ok=True)

    def extract_content_from_pdf(self, pdf_path: str) -> ExtractedContent:
        """
        Extrae todo el contenido (texto e imágenes) de un PDF.

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

        for page_num in range(len(doc)):
            page = doc[page_num]

            # Extraer texto de la página
            page_text = page.get_text()
            page_texts[page_num] = page_text
            all_text.append(page_text)

            # Extraer imágenes de la página
            page_images = self._extract_images_from_page(page, page_num)
            images.extend(page_images)

        doc.close()

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

    def _extract_images_from_page(self, page, page_num: int) -> List[Dict[str, Any]]:
        """
        Extrae todas las imágenes de una página específica.

        Args:
            page: Objeto página de PyMuPDF
            page_num (int): Número de página

        Returns:
            List[Dict]: Lista de imágenes con metadatos
        """
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

                    # Obtener bbox de la imagen en la página
                    bbox = page.get_image_bbox(img)

                    images.append({
                        "image": pil_image,
                        "page": page_num,
                        "bbox": bbox,
                        "path": img_path,
                        "filename": img_filename,
                        "size": pil_image.size
                    })

                pix = None  # Liberar memoria

            except Exception as e:
                self.logger.warning(f"Error extrayendo imagen {img_index} de página {page_num}: {e}")
                continue

        return images

    def _create_text_chunks(self, all_text: List[str], page_texts: Dict[int, str]) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Divide el texto en chunks manejables con metadatos.

        Args:
            all_text (List[str]): Texto de todas las páginas
            page_texts (Dict[int, str]): Texto por página

        Returns:
            Tuple[List[str], List[Dict]]: Chunks de texto y metadatos
        """
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

        return chunks, metadata

    def _clean_text(self, text: str) -> str:
        """
        Limpia y normaliza el texto extraído.

        Args:
            text (str): Texto crudo

        Returns:
            str: Texto limpio
        """
        # Remover caracteres de control y espacios extras
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\"\'\/]', '', text)

        # Remover líneas muy cortas (probablemente headers/footers)
        lines = text.split('\n')
        cleaned_lines = [line.strip() for line in lines if len(line.strip()) > 3]

        return '\n'.join(cleaned_lines).strip()

    def _split_text_into_chunks(self, text: str) -> List[str]:
        """
        Divide texto en chunks con solapamiento.

        Args:
            text (str): Texto a dividir

        Returns:
            List[str]: Lista de chunks
        """
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
        """
        Asocia imágenes con chunks de texto basándose en la página.

        Args:
            text_chunks (List[str]): Chunks de texto
            images (List[Dict]): Imágenes extraídas
            metadata (List[Dict]): Metadatos de chunks
        """
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
        """
        Retorna estadísticas del modelo de embeddings.

        Returns:
            Dict[str, Any]: Estadísticas del preprocessor
        """
        return {
            "embedding_model": self.embedding_model_name,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "embedding_dimension": self.embedding_model.get_sentence_embedding_dimension(),
            "images_directory": self.images_dir
        }
