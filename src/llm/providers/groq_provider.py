"""
Proveedor de LLM para Groq API.
"""
import os
import logging
from typing import Optional
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()


class GroqProvider:
    """
    Proveedor para interactuar con la API de Groq.
    """

    def __init__(self, model: str = "llama-3.1-8b-instant", temperature: float = 0.7,
                 api_key: Optional[str] = None, logger: Optional[logging.Logger] = None):
        """
        Inicializa el proveedor de Groq.

        Args:
            model: Modelo a utilizar
            temperature: Temperatura para la generación
            api_key: API key de Groq (opcional, se puede obtener de variables de entorno)
            logger: Logger para registrar eventos
        """
        self.model = model
        self.temperature = temperature
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.logger = logger or logging.getLogger(__name__)

        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY no está configurada. "
                "Configúrala como variable de entorno o pásala como parámetro."
            )

        self._client = None

    def _get_client(self) -> ChatGroq:
        """
        Obtiene o crea el cliente de Groq de forma lazy.

        Returns:
            Cliente de ChatGroq configurado
        """
        if self._client is None:
            try:
                self._client = ChatGroq(
                    model=self.model,
                    temperature=self.temperature,
                    api_key=self.api_key
                )
                self.logger.info(f"Cliente Groq inicializado con modelo: {self.model}")
            except Exception as e:
                self.logger.error(f"Error inicializando cliente Groq: {e}")
                raise

        return self._client

    def generate_response(self, prompt: str) -> str:
        """
        Genera una respuesta usando el modelo de Groq.

        Args:
            prompt: Prompt para el modelo

        Returns:
            Respuesta generada por el modelo

        Raises:
            Exception: Si hay un error al generar la respuesta
        """
        try:
            client = self._get_client()

            self.logger.debug(f"Enviando prompt a Groq: {prompt[:100]}...")

            response = client.invoke(prompt)

            # Extraer contenido de la respuesta
            if hasattr(response, 'content'):
                answer = response.content
            else:
                answer = str(response)

            self.logger.info(f"Respuesta generada exitosamente (longitud: {len(answer)})")

            return answer

        except Exception as e:
            self.logger.error(f"Error generando respuesta con Groq: {e}")
            raise Exception(f"Error del proveedor Groq: {str(e)}")

    def is_available(self) -> bool:
        """
        Verifica si el proveedor está disponible.

        Returns:
            True si el proveedor está disponible, False en caso contrario
        """
        try:
            client = self._get_client()
            return client is not None
        except Exception:
            return False
