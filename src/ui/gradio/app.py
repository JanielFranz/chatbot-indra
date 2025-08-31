import gradio as gr
import requests

API_URL = "http://127.0.0.1:8000/chatbot/ask"

def chat_with_backend(message, history):
    """
    message: user input
    history: list of (user, assistant) messages maintained by Gradio
    """
    try:
        # Send user message to FastAPI backend
        response = requests.post(API_URL, json={"question": message})
        response.raise_for_status()  # Raise an exception for bad status codes

        data = response.json()
        text = data.get("answer", "No response received")
        images = data.get("images", [])

        # Always return just the text for ChatInterface
        # ChatInterface handles multimodal differently than we initially thought
        return text

    except requests.exceptions.ConnectionError:
        return "❌ Error: No se puede conectar con el backend. Asegúrate de que FastAPI esté corriendo en http://127.0.0.1:8000"
    except requests.exceptions.RequestException as e:
        return f"❌ Error de conexión: {str(e)}"
    except Exception as e:
        return f"❌ Error inesperado: {str(e)}"

with gr.Blocks() as demo:
    gr.ChatInterface(
        fn=chat_with_backend,
        title="📚 Multimodal RAG Chatbot",
        description="Ask questions about the document. Responses may include text and images.",
        # Removed multimodal=True as it's causing issues with return type inference
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
