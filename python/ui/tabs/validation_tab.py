"""
Validation Tab
==============
Single molecule validation tab.
"""

import gradio as gr
from .utils import draw_molecule, format_validation


def create_validation_tab(validator):
    """Create the validation tab interface."""
    
    with gr.TabItem("Validate", id="validate"):
        gr.Markdown("""
        ### Single Molecule Validation
        Enter a SMILES string to validate if it's a CRAM molecule.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                smiles_input = gr.Textbox(
                    label="SMILES",
                    placeholder="Enter SMILES string...",
                    lines=2
                )
                
                with gr.Row():
                    validate_btn = gr.Button("Validate", variant="primary", scale=2)
                    clear_btn = gr.Button("Clear", scale=1)
                
                gr.Markdown("""
                ---
                ### CRAM Criteria
                | Criterion | Requirement |
                |-----------|-------------|
                | Carboxyl Groups | ≥ 2 (COOH/COO⁻) |
                | Ring Type | Alicyclic (non-aromatic) |
                | Aliphatic Carbon | > 60% |
                | Molecular Weight | 100-750 Da |
                """)
            
            with gr.Column(scale=1):
                gr.Markdown("### Molecule Structure")
                mol_image = gr.Image(label="", height=350, show_label=False)
            
            with gr.Column(scale=1):
                gr.Markdown("### Validation Results")
                validation_md = gr.Markdown("Enter a SMILES string to validate")
        
        def on_validate(smiles):
            if not smiles:
                return None, "Please enter a SMILES"
            
            result = validator.validate_smiles(smiles)
            image = draw_molecule(smiles)
            text = format_validation(result)
            return image, text
        
        validate_btn.click(
            fn=on_validate,
            inputs=[smiles_input],
            outputs=[mol_image, validation_md]
        )
        
        clear_btn.click(
            fn=lambda: ("", None, "Enter a SMILES string to validate"),
            outputs=[smiles_input, mol_image, validation_md]
        )
