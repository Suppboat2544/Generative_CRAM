"""
UI Tabs Package
===============
Modular tab components for Gradio interface.
"""

from .validation_tab import create_validation_tab
from .cvae_tab import create_cvae_tab
from .batch_tab import create_batch_tab
from .generate_tab import create_generate_tab
from .about_tab import create_about_tab

__all__ = [
    "create_validation_tab",
    "create_cvae_tab",
    "create_batch_tab",
    "create_generate_tab",
    "create_about_tab",
]
