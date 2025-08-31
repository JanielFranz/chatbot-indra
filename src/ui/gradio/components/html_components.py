"""
HTML templates for the Gradio interface components
"""

def get_header_html():
    """Return the header section HTML"""
    return """
    <div class="header-section">
        <h1 class="header-title">🤖 Intelligent Document Assistant</h1>
        <p class="header-subtitle">
            Indra AI Chatbot Test
        </p>
        <p class="header-description">
            Ask questions about your documents and get intelligent responses with relevant context
        </p>
    </div>
    """

def get_features_html():
    """Return the features section HTML"""
    return """
    <div class="feature-card">
        <h3>✨ Features</h3>
        <ul class="feature-list">
            <li>🔍 Advanced document search and retrieval</li>
            <li>🖼️ Multimodal support (text + images)</li>
            <li>🧠 AI-powered intelligent responses</li>
        </ul>
    </div>
    """

def get_tips_html():
    """Return the tips section HTML"""
    return """
    <div class="tips-card">
        <h3>💡 Tips for Better Results</h3>
        <ul class="tips-list">
            <li><strong>Be specific:</strong> Ask detailed questions for more accurate responses</li>
            <li><strong>Context matters:</strong> Reference specific sections or topics when possible</li>
            <li><strong>Multiple queries:</strong> Feel free to ask follow-up questions</li>
            <li><strong>Visual content:</strong> Ask about charts, images, or diagrams in the document</li>
            <li><strong>Follow-up:</strong> Build on previous answers for deeper insights</li>
        </ul>
    </div>
    """

def get_footer_html():
    """Return the footer section HTML"""
    return """
    <div class="footer-section">
        <p>🚀 Built with Gradio, FastAPI, and FAISS | Developed by Janiel Franz Escalante</p>
        <p class="footer-version">
            Version 1.0 
        </p>
    </div>
    """

def get_status_html(is_connected: bool, message: str = ""):
    """Return the status indicator HTML with dynamic styling"""
    status_class = "status-connected" if is_connected else "status-disconnected"
    icon = "🟢" if is_connected else "🔴"

    if not message:
        message = "Backend Status: Connected and running" if is_connected else "Backend Status: Disconnected - Please start your FastAPI server"

    return f"""
    <div class="status-indicator {status_class}">
        {icon} <strong>{message}</strong>
    </div>
    """
