"""
Header Component
================
Application header with title and description.
"""

import gradio as gr


def create_header():
    """Create the application header."""
    gr.Markdown("""
    # 🧪 CRAM Generator
    
    **Deep Learning Toolkit for Marine DOM Molecule Generation**
    
    Generate and validate Carboxyl-Rich Alicyclic Molecules (CRAM) using 
    property-conditioned variational autoencoders and SELFIES mutation.
    
    ---
    """)


def create_compact_header():
    """Create a compact header for embedded use."""
    gr.Markdown("""
    # 🧪 CRAM Generator
    *Marine DOM Molecule Generation Toolkit*
    """)
