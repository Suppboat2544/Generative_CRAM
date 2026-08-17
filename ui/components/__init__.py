"""
UI Components Package
=====================
Reusable UI components for Gradio interface.
"""

from .header import create_header, create_compact_header
from .footer import create_footer, create_compact_footer
from .examples import (
    EXAMPLE_MOLECULES,
    GENERATION_SEEDS,
    MARINE_DOM_EXAMPLES,
    get_example_by_name,
    get_all_example_smiles,
    get_example_names,
)
from .molecule_display import (
    draw_molecule_image,
    format_molecule_card,
    create_property_badges,
    create_molecule_gallery,
)

__all__ = [
    # Header/Footer
    "create_header",
    "create_compact_header",
    "create_footer",
    "create_compact_footer",
    # Examples
    "EXAMPLE_MOLECULES",
    "GENERATION_SEEDS",
    "MARINE_DOM_EXAMPLES",
    "get_example_by_name",
    "get_all_example_smiles",
    "get_example_names",
    # Molecule Display
    "draw_molecule_image",
    "format_molecule_card",
    "create_property_badges",
    "create_molecule_gallery",
]
