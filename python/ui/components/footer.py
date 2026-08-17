"""
Footer Component
================
Application footer with version and links.
"""

import gradio as gr


VERSION = "1.0.0"


def create_footer():
    """Create the application footer."""
    gr.Markdown(f"""
    ---
    
    **CRAM Generator v{VERSION}** | 
    [Documentation](https://github.com/your-repo/cram-generator) | 
    [Report Issue](https://github.com/your-repo/cram-generator/issues)
    
    *Built with PyTorch, RDKit, and Gradio*
    """)


def create_compact_footer():
    """Create a compact footer."""
    gr.Markdown(f"---\n*CRAM Generator v{VERSION}*")
