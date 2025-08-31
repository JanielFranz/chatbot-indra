"""
CSS loader and styling utilities for the Gradio interface
"""

import os
from pathlib import Path

def load_css() -> str:
    """Load the CSS file and return its content"""
    css_path = Path(__file__).parent.parent.parent / "static" / "css" / "styles.css"

    try:
        with open(css_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        # Fallback CSS if file not found
        return """
        .gradio-container {
            background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        """

def get_theme_config():
    """Return Gradio theme configuration"""
    return {
        "primary_hue": "slate",
        "secondary_hue": "gray",
        "neutral_hue": "gray",
        "spacing_size": "md",
        "radius_size": "md",
        "text_size": "md",
        "font": ["system-ui", "sans-serif"],
        "font_mono": ["ui-monospace", "monospace"]
    }
