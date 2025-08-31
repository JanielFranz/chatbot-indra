import gradio as gr
import requests
import time
from components.html_components import (
    get_header_html,
    get_features_html,
    get_tips_html,
    get_footer_html,
    get_status_html
)
from utils.style_loader import load_css, get_theme_config

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
        return "🔌 **Connection Error**: Cannot connect to the backend server. Please make sure FastAPI is running on http://127.0.0.1:8000"
    except requests.exceptions.RequestException as e:
        return f"🌐 **Network Error**: {str(e)}"
    except Exception as e:
        return f"⚠️ **Unexpected Error**: {str(e)}"

def check_backend_status():
    """Check if the backend is running"""
    try:
        response = requests.get("http://127.0.0.1:8000/docs", timeout=5)
        return get_status_html(True, "Backend Status: Connected and running")
    except:
        return get_status_html(False, "Backend Status: Disconnected - Please start your FastAPI server")

# Load external CSS
custom_css = load_css()

# Create the improved interface
with gr.Blocks(
    css=custom_css,
    title="📚 Multimodal RAG Assistant",
    theme=gr.themes.Soft(**get_theme_config())
) as demo:

    # Header section
    with gr.Row():
        with gr.Column():
            gr.HTML(get_header_html())

    # Status and features section
    with gr.Row():
        with gr.Column(scale=2):
            # Status indicator
            status_display = gr.HTML(check_backend_status(), elem_classes=["status-indicator"])

            # Refresh status button
            refresh_btn = gr.Button(
                "🔄 Refresh Status",
                variant="secondary",
                size="sm",
                elem_classes=["refresh-btn"]
            )
            refresh_btn.click(fn=check_backend_status, outputs=status_display)

        with gr.Column(scale=3):
            gr.HTML(get_features_html())

    # Main chat interface
    with gr.Row():
        with gr.Column():
            gr.HTML('<div class="chat-container">')

            # Chat interface with improved styling
            chat_interface = gr.ChatInterface(
                fn=chat_with_backend,
                title="💬 Chat with Your Documents",
                description="Start a conversation by typing your question below. I can help you find information from your documents!",
                examples=[
                    "What is this document about?",
                    "Can you summarize the main points?",
                    "Show me any charts or graphs",
                    "What are the key findings?",
                    "Explain the methodology used"
                ],
                retry_btn="🔄 Retry",
                undo_btn="↩️ Undo",
                clear_btn="🗑️ Clear Chat",
                submit_btn="📤 Send",
                stop_btn="⏹️ Stop"
            )

            gr.HTML('</div>')

    # Footer
    with gr.Row():
        with gr.Column():
            gr.HTML(get_footer_html())

if __name__ == "__main__":
    print("🚀 Starting Multimodal RAG Assistant...")
    print("📍 Interface will be available at: http://localhost:7860")
    print("🔗 Make sure your FastAPI backend is running on: http://localhost:8000")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_api=False,
        share=False,
        inbrowser=True  # Automatically open in browser
    )
