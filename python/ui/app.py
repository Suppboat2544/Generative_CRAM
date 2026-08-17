"""
CRAM Generator - Modular Web Interface
======================================

Streamlined Gradio interface using modular tab components.

Developed by:
    KLABKLAYDEE SUPAPORN
    Fujii Laboratory
    Tokyo Institute of Technology

Supervised by:
    Assoc. Prof. Manabu Fujii

Contact:
    klabklaydee.s.aa@m.titech.ac.jp
"""

import gradio as gr
import os
import sys

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modular components
from .tabs import (
    create_validation_tab,
    create_batch_tab,
    create_generate_tab,
    create_cvae_tab,
    create_about_tab,
)
from .components import (
    create_header,
    create_footer,
)


# =============================================================================
# THEME AND STYLING
# =============================================================================

CUSTOM_CSS = """
/* Main container */
.gradio-container {
    max-width: 1400px !important;
    margin: auto !important;
    font-family: 'Inter', system-ui, sans-serif !important;
}

/* Header styling */
.header-container {
    background: linear-gradient(135deg, #1a365d 0%, #2d5a87 100%);
    color: white;
    padding: 30px;
    border-radius: 16px;
    margin-bottom: 20px;
    text-align: center;
}

/* Card styling */
.card {
    background: white;
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    border: 1px solid #e5e7eb;
}

/* Status badges */
.badge-success { 
    background: #10b981; 
    color: white; 
    padding: 4px 12px; 
    border-radius: 20px;
    font-weight: 600;
}
.badge-error { 
    background: #ef4444; 
    color: white; 
    padding: 4px 12px; 
    border-radius: 20px;
    font-weight: 600;
}

/* Tab styling */
.tabs { border: none !important; }
.tab-nav { 
    background: #f3f4f6 !important;
    border-radius: 8px !important;
    padding: 4px !important;
}

/* Button styling */
.primary-btn {
    background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
    border: none !important;
    font-weight: 600 !important;
}

/* Molecule display */
.molecule-image {
    border: 2px solid #e5e7eb;
    border-radius: 8px;
    padding: 10px;
    background: white;
}

/* Property table */
.property-table {
    font-size: 14px;
    width: 100%;
}
.property-table th {
    background: #f3f4f6;
    padding: 8px;
    text-align: left;
}
.property-table td {
    padding: 8px;
    border-bottom: 1px solid #e5e7eb;
}

/* Score indicators */
.score-high { color: #10b981; font-weight: bold; }
.score-medium { color: #f59e0b; font-weight: bold; }
.score-low { color: #ef4444; font-weight: bold; }
"""


# =============================================================================
# MAIN INTERFACE
# =============================================================================

def create_interface() -> gr.Blocks:
    """
    Create the main Gradio interface with modular tabs.
    
    Returns:
        gr.Blocks: The complete Gradio interface
    """
    with gr.Blocks(
        title="CRAM Generator",
        css=CUSTOM_CSS,
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="gray",
        )
    ) as interface:
        
        # Header
        create_header()
        
        # Main tabs
        with gr.Tabs():
            # Validation tab
            create_validation_tab()
            
            # Batch validation tab
            create_batch_tab()
            
            # Mutation generation tab
            create_generate_tab()
            
            # CVAE generation tab
            create_cvae_tab()
            
            # About tab
            create_about_tab()
        
        # Footer
        create_footer()
    
    return interface


def launch_app(
    share: bool = False,
    server_name: str = "0.0.0.0",
    server_port: int = 7860,
    debug: bool = False
):
    """
    Launch the Gradio application.
    
    Args:
        share: Whether to create a public link
        server_name: Server hostname
        server_port: Server port
        debug: Enable debug mode
    """
    interface = create_interface()
    interface.launch(
        share=share,
        server_name=server_name,
        server_port=server_port,
        debug=debug
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="CRAM Generator Web Interface")
    parser.add_argument("--share", action="store_true", help="Create public link")
    parser.add_argument("--port", type=int, default=7860, help="Server port")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    launch_app(
        share=args.share,
        server_port=args.port,
        debug=args.debug
    )
