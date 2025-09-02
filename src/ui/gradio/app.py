import gradio as gr
import requests
import os
from components.html_components import (
    get_header_html,
    get_features_html,
    get_footer_html,
    get_status_html
)
from utils.style_loader import load_css, get_theme_config

API_URL = "http://127.0.0.1:8000/chatbot/ask"

# Global variable to store current images
current_images = []

def handle_user_message(message, history):
    """
    Adds the user's message to the chat history.
    The history is a list of lists, where each inner list has two elements: [user_message, bot_response].
    """
    history.append([message, None])
    return "", history

def get_bot_response(history):
    """
    Gets the bot's response from the backend and updates the last message in the history.
    The bot's response can be a string or a list containing text and image paths.
    """
    global current_images
    user_message = history[-1][0]

    try:
        response = requests.post(API_URL, json={"question": user_message})
        response.raise_for_status()

        data = response.json()
        text = data.get("answer", "No response received")
        images = data.get("images", [])

        # Update current images for the gallery
        current_images = []
        if images:
            for image_path in images:
                abs_path = os.path.abspath(image_path)
                if os.path.exists(abs_path):
                    current_images.append(abs_path)

        # Always just show the text response - images will be displayed in the gallery
        history[-1][1] = text

    except requests.exceptions.ConnectionError:
        history[-1][1] = "🔌 **Connection Error**: Cannot connect to the backend server."
    except requests.exceptions.RequestException as e:
        history[-1][1] = f"🌐 **Network Error**: {str(e)}"
    except Exception as e:
        history[-1][1] = f"⚠️ **Unexpected Error**: {str(e)}"

    return history

def check_backend_status():
    """Checks if the backend server is running."""
    try:
        requests.get("http://127.0.0.1:8000/docs", timeout=5)
        return get_status_html(True, "Backend Status: Connected and running")
    except requests.exceptions.ConnectionError:
        return get_status_html(False, "Backend Status: Disconnected - Please start your FastAPI server")

def get_current_images():
    """Return current images for the gallery"""
    global current_images
    return current_images if current_images else []

# Load external CSS
custom_css = load_css()

# Create the improved interface using gr.Blocks
with gr.Blocks(
    css=custom_css,
    title="📚 Multimodal RAG Assistant",
    theme=gr.themes.Soft(**get_theme_config())
) as demo:

    # Header section
    gr.HTML(get_header_html())

    # Status and features section
    with gr.Row():
        with gr.Column(scale=2):
            status_display = gr.HTML(check_backend_status())
            refresh_btn = gr.Button("🔄 Refresh Status", variant="secondary", size="sm")
            refresh_btn.click(fn=check_backend_status, outputs=status_display)
        with gr.Column(scale=3):
            gr.HTML(get_features_html())

    # Custom chat interface implementation
    with gr.Column():
        # Chatbot component to display the conversation
        chatbot = gr.Chatbot(
            label="Multimodal Chat",
            bubble_full_width=False,
            height=500  # Slightly reduced to make room for image gallery
        )
        # Textbox for user input
        with gr.Row():
            msg_input = gr.Textbox(
                placeholder="Type your question here and press Enter...",
                container=False,
                scale=7,
            )
            send_btn = gr.Button("Send", variant="primary", scale=1)

    # Image gallery for displaying related images
    with gr.Column():
        gr.HTML('<h3>📸 Related Images</h3>')
        image_gallery = gr.Gallery(
            label="Images from Document",
            show_label=False,
            elem_id="image_gallery",
            columns=4,
            rows=2,
            height=300,
            allow_preview=True,
            show_share_button=False
        )

    # Footer section
    gr.HTML(get_footer_html())

    # Event handlers for sending messages
    # This handles the case where the user clicks the "Send" button
    send_btn.click(
        fn=handle_user_message,
        inputs=[msg_input, chatbot],
        outputs=[msg_input, chatbot]
    ).then(
        fn=get_bot_response,
        inputs=chatbot,
        outputs=chatbot
    ).then(
        fn=get_current_images,
        outputs=image_gallery
    )

    # This handles the case where the user presses Enter in the textbox
    msg_input.submit(
        fn=handle_user_message,
        inputs=[msg_input, chatbot],
        outputs=[msg_input, chatbot]
    ).then(
        fn=get_bot_response,
        inputs=chatbot,
        outputs=chatbot
    ).then(
        fn=get_current_images,
        outputs=image_gallery
    )

if __name__ == "__main__":
    print("🚀 Starting Multimodal RAG Assistant...")
    print("📍 Interface will be available at: http://localhost:7860")
    print("🔗 Make sure your FastAPI backend is running on: http://localhost:8000")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_api=False,
        share=False,
        inbrowser=True
    )
