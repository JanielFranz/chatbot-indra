import faiss
import numpy as np
import pickle
import os
from typing import List, Dict, Any, Tuple, Optional
import logging


class FAISSVectorStore:
    """
    Clase para manejar la base de datos vectorial FAISS para el chatbot RAG multimodal.
    Almacena embeddings de texto y metadatos que vinculan texto con imágenes extraídas de PDFs.
    """

    def __init__(self, dimension: int = 384, index_type: str = "flat"):
        """
        Inicializa la instancia de FAISS.

        Args:
            dimension (int): Dimensión de los embeddings (por defecto 384 para all-MiniLM-L6-v2)
            index_type (str): Tipo de índice FAISS ('flat', 'ivf', 'hnsw')
        """
        self.dimension = dimension
        self.index_type = index_type
        self.index = None
        self.metadata = []  # Lista para almacenar metadatos de cada embedding
        self.id_to_index = {}  # Mapeo de ID personalizado a índice FAISS
        self.next_id = 0

        # Configurar logging ANTES de inicializar el índice
        self.logger = logging.getLogger(__name__)

        self._initialize_index()

    def _initialize_index(self):
        """Inicializa el índice FAISS según el tipo especificado."""
        if self.index_type == "flat":
            # Índice plano (L2 distance) - más preciso pero más lento para grandes datasets
            self.index = faiss.IndexFlatL2(self.dimension)
        elif self.index_type == "ivf":
            # Índice IVF (Inverted File) - más rápido para datasets grandes
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)  # 100 clusters
        elif self.index_type == "hnsw":
            # HNSW (Hierarchical Navigable Small World) - muy rápido y preciso
            self.index = faiss.IndexHNSWFlat(self.dimension, 32)
        else:
            raise ValueError(f"Tipo de índice no soportado: {self.index_type}")

        self.logger.info(f"Índice FAISS inicializado: {self.index_type}, dimensión: {self.dimension}")

    def add_embeddings(self,
                      embeddings: Optional[np.ndarray] = None,
                      text_chunks: Optional[List[str]] = None,
                      metadata: Optional[List[Dict[str, Any]]] = None,
                      images: Optional[List[Dict[str, Any]]] = None,
                      processed_data: Optional[Dict[str, Any]] = None) -> List[int]:
        """
        Agrega embeddings de texto al índice FAISS junto con sus metadatos.

        Puede recibir parámetros individuales O un diccionario processed_data del IngestionService.

        Args:
            embeddings (np.ndarray, optional): Array de embeddings de forma (n_samples, dimension)
            text_chunks (List[str], optional): Lista de textos correspondientes a cada embedding
            metadata (List[Dict[str, Any]], optional): Metadatos para cada chunk
            images (List[Dict[str, Any]], optional): Lista de imágenes del PDF
            processed_data (Dict[str, Any], optional): Diccionario con toda la información procesada

        Returns:
            List[int]: Lista de IDs asignados a cada embedding
        """
        # Si se proporciona processed_data, extraer los valores de ahí
        if processed_data is not None:
            embeddings = processed_data.get("embeddings")
            text_chunks = processed_data.get("text_chunks", [])
            metadata = processed_data.get("metadata", [])
            images = processed_data.get("images", [])

        # Validaciones
        if embeddings is None or text_chunks is None:
            raise ValueError("Se requieren embeddings y text_chunks (ya sea como parámetros o en processed_data)")

        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Dimensión de embeddings ({embeddings.shape[1]}) no coincide con la esperada ({self.dimension})")

        if len(embeddings) != len(text_chunks):
            raise ValueError("El número de embeddings debe coincidir con el número de textos")

        # Si no hay metadatos, crear metadatos básicos
        if not metadata:
            metadata = [{"chunk_id": f"chunk_{i}", "page_number": None, "chunk_index": i}
                       for i in range(len(text_chunks))]

        # Normalizar embeddings para búsqueda por cosine similarity
        faiss.normalize_L2(embeddings)

        # Entrenar el índice si es necesario (para IVF)
        if self.index_type == "ivf" and not self.index.is_trained:
            self.logger.info("Entrenando índice IVF...")
            self.index.train(embeddings)

        # Agregar embeddings al índice
        start_idx = len(self.metadata)
        self.index.add(embeddings)

        # Generar IDs y agregar metadatos enriquecidos
        assigned_ids = []
        for i, (text, chunk_metadata) in enumerate(zip(text_chunks, metadata)):
            doc_id = chunk_metadata.get("chunk_id", f"doc_{self.next_id}")
            assigned_ids.append(self.next_id)

            # Buscar imágenes asociadas a este chunk por página
            page_number = chunk_metadata.get("page_number")
            #self.logger.info(f"Documento! {doc_id} ({page_number}) {text}")
            associated_images = []
            if images and page_number is not None:
                #self.logger.info(f"Imagenes del PDF de acuerdo a la pag: {images[page_number]}")
                associated_images = [img for img in images if img.get("page") == page_number]

            # Crear metadatos enriquecidos para FAISS
            faiss_metadata = {
                "id": doc_id,
                "text": text,
                "page_number": page_number,
                "chunk_index": chunk_metadata.get("chunk_index", i),
                "faiss_index": start_idx + i,
                "associated_images": len(associated_images),
                "image_info": associated_images[:3] if associated_images else []  # Limitar a 3 imágenes max
            }

            #self.logger.info(f"El texto: {faiss_metadata['text']}")
            #self.logger.info(f"page number: {faiss_metadata['page_number']} - asociated images: {faiss_metadata["associated_images"]} - image info: {faiss_metadata['image_info']}")
            self.metadata.append(faiss_metadata)
            self.id_to_index[self.next_id] = start_idx + i
            self.next_id += 1

        self.logger.info(f"Agregados {len(embeddings)} embeddings al índice FAISS")
        return assigned_ids

    def search(self,
               query_embedding: np.ndarray,
               k: int = 5,
               return_metadata: bool = True) -> Tuple[List[float], List[Dict[str, Any]]]:
        """
        Busca los k embeddings más similares al query.

        Args:
            query_embedding (np.ndarray): Embedding de la consulta (1, dimension)
            k (int): Número de resultados a retornar
            return_metadata (bool): Si retornar metadatos completos

        Returns:
            Tuple[List[float], List[Dict[str, Any]]]: (distancias, metadatos)
        """
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)

        if query_embedding.shape[1] != self.dimension:
            raise ValueError(f"Dimensión del query ({query_embedding.shape[1]}) no coincide con la esperada ({self.dimension})")

        # Normalizar query embedding
        faiss.normalize_L2(query_embedding)

        # Realizar búsqueda
        distances, indices = self.index.search(query_embedding, k)

        # Obtener metadatos
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.metadata):  # -1 indica que no se encontraron suficientes resultados
                if return_metadata:
                    result = self.metadata[idx].copy()
                    result["distance"] = float(distances[0][i])
                    result["similarity"] = 1.0 / (1.0 + float(distances[0][i]))  # Convertir distancia a similitud
                else:
                    result = {
                        "id": self.metadata[idx]["id"],
                        "text": self.metadata[idx]["text"],
                        "distance": float(distances[0][i]),
                        "similarity": 1.0 / (1.0 + float(distances[0][i]))
                    }
                results.append(result)
        self.logger.info(f"resultados de busqueda: {distances[0].tolist(), results}")
        return distances[0].tolist(), results

    def get_by_id(self, doc_id: int) -> Optional[Dict[str, Any]]:
        """
        Obtiene un documento por su ID.

        Args:
            doc_id (int): ID del documento

        Returns:
            Optional[Dict[str, Any]]: Metadatos del documento o None si no existe
        """
        if doc_id in self.id_to_index:
            faiss_idx = self.id_to_index[doc_id]
            return self.metadata[faiss_idx]
        return None

    def save_index(self, filepath: str):
        """
        Guarda el índice FAISS y metadatos en disco.

        Args:
            filepath (str): Ruta base para guardar los archivos (sin extensión)
        """
        # Guardar índice FAISS
        faiss.write_index(self.index, f"{filepath}.faiss")

        # Guardar metadatos y mapeos
        metadata_dict = {
            "metadata": self.metadata,
            "id_to_index": self.id_to_index,
            "next_id": self.next_id,
            "dimension": self.dimension,
            "index_type": self.index_type
        }

        with open(f"{filepath}_metadata.pkl", "wb") as f:
            pickle.dump(metadata_dict, f)

        self.logger.info(f"Índice FAISS guardado en: {filepath}")

    def load_index(self, filepath: str):
        """
        Carga el índice FAISS y metadatos desde disco.

        Args:
            filepath (str): Ruta base de los archivos (sin extensión)
        """
        if not os.path.exists(f"{filepath}.faiss"):
            raise FileNotFoundError(f"Archivo de índice no encontrado: {filepath}.faiss")

        if not os.path.exists(f"{filepath}_metadata.pkl"):
            raise FileNotFoundError(f"Archivo de metadatos no encontrado: {filepath}_metadata.pkl")

        # Cargar índice FAISS
        self.index = faiss.read_index(f"{filepath}.faiss")

        # Cargar metadatos
        with open(f"{filepath}_metadata.pkl", "rb") as f:
            metadata_dict = pickle.load(f)

        self.metadata = metadata_dict["metadata"]
        self.id_to_index = metadata_dict["id_to_index"]
        self.next_id = metadata_dict["next_id"]
        self.dimension = metadata_dict["dimension"]
        self.index_type = metadata_dict["index_type"]

        self.logger.info(f"Índice FAISS cargado desde: {filepath}")

    def get_stats(self) -> Dict[str, Any]:
        """
        Retorna estadísticas del índice FAISS.

        Returns:
            Dict[str, Any]: Estadísticas del índice
        """
        return {
            "total_vectors": self.index.ntotal,
            "dimension": self.dimension,
            "index_type": self.index_type,
            "is_trained": getattr(self.index, 'is_trained', True),
            "metadata_count": len(self.metadata)
        }

    def clear(self):
        """Limpia todos los datos del índice."""
        self._initialize_index()
        self.metadata = []
        self.id_to_index = {}
        self.next_id = 0
        self.logger.info("Índice FAISS limpiado")


# Función de conveniencia para crear una instancia
def create_faiss_store(dimension: int = 384, index_type: str = "flat") -> FAISSVectorStore:
    """
    Función de conveniencia para crear una instancia de FAISSVectorStore.

    Args:
        dimension (int): Dimensión de los embeddings
        index_type (str): Tipo de índice FAISS

    Returns:
        FAISSVectorStore: Instancia configurada
    """
    return FAISSVectorStore(dimension=dimension, index_type=index_type)
