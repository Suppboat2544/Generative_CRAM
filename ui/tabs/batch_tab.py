"""
Batch Validation Tab
====================
Tab for validating multiple SMILES at once.
"""

import gradio as gr
from typing import List, Tuple
import pandas as pd

from ..utils import format_validation, format_properties


def validate_batch(smiles_text: str) -> Tuple[str, str]:
    """Validate multiple SMILES strings."""
    from src.validator import CRAMValidator
    from src.features import DeepCRAMFeatureExtractor
    
    if not smiles_text.strip():
        return "No SMILES provided", ""
    
    # Parse SMILES (one per line or comma-separated)
    smiles_list = []
    for line in smiles_text.strip().split('\n'):
        for smi in line.split(','):
            smi = smi.strip()
            if smi:
                smiles_list.append(smi)
    
    if not smiles_list:
        return "No valid SMILES found", ""
    
    # Initialize validator
    validator = CRAMValidator()
    
    results = []
    valid_count = 0
    
    for i, smiles in enumerate(smiles_list[:100]):  # Limit to 100
        result = validator.validate(smiles)
        is_valid = result.get('valid', False)
        if is_valid:
            valid_count += 1
        
        results.append({
            'Index': i + 1,
            'SMILES': smiles[:50] + '...' if len(smiles) > 50 else smiles,
            'Valid': '✓' if is_valid else '✗',
            'Score': f"{result.get('cram_score', 0):.3f}",
            'MW': f"{result.get('mw', 0):.1f}",
            'Message': result.get('reason', '')[:30]
        })
    
    # Create summary
    summary = f"""
## Batch Validation Results

- **Total SMILES:** {len(smiles_list)}
- **Validated:** {min(len(smiles_list), 100)}
- **Valid CRAM:** {valid_count}
- **Invalid:** {min(len(smiles_list), 100) - valid_count}
- **Success Rate:** {100 * valid_count / min(len(smiles_list), 100):.1f}%
"""
    
    # Create table
    df = pd.DataFrame(results)
    table_md = df.to_markdown(index=False)
    
    return summary, table_md


def export_results(smiles_text: str) -> str:
    """Export validation results as CSV."""
    from src.validator import CRAMValidator
    
    if not smiles_text.strip():
        return ""
    
    smiles_list = [s.strip() for s in smiles_text.strip().split('\n') if s.strip()]
    validator = CRAMValidator()
    
    rows = []
    for smiles in smiles_list[:500]:
        result = validator.validate(smiles)
        rows.append({
            'smiles': smiles,
            'valid': result.get('valid', False),
            'cram_score': result.get('cram_score', 0),
            'mw': result.get('mw', 0),
            'reason': result.get('reason', '')
        })
    
    df = pd.DataFrame(rows)
    csv_path = '/tmp/cram_validation_results.csv'
    df.to_csv(csv_path, index=False)
    return csv_path


def create_batch_tab():
    """Create the batch validation tab."""
    with gr.Tab("Batch Validation"):
        gr.Markdown("""
        ## Batch SMILES Validation
        Validate multiple SMILES at once (up to 100 for display, 500 for export).
        Enter one SMILES per line or comma-separated.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                batch_input = gr.Textbox(
                    label="SMILES (one per line)",
                    placeholder="Enter SMILES here...\nC(=O)O\nCCO\n...",
                    lines=10
                )
                
                with gr.Row():
                    validate_btn = gr.Button("Validate All", variant="primary")
                    export_btn = gr.Button("Export CSV")
                
                # Example batch
                gr.Markdown("### Example Batch")
                example_batch = """C(=O)O
CCO
c1ccccc1
CC(=O)O
CCCC(=O)O
OC(=O)CCC(=O)O"""
                load_example_btn = gr.Button("Load Examples")
            
            with gr.Column(scale=2):
                summary_output = gr.Markdown(label="Summary")
                table_output = gr.Markdown(label="Results Table")
                csv_output = gr.File(label="Download CSV", visible=False)
        
        # Event handlers
        validate_btn.click(
            fn=validate_batch,
            inputs=[batch_input],
            outputs=[summary_output, table_output]
        )
        
        load_example_btn.click(
            fn=lambda: example_batch,
            outputs=[batch_input]
        )
        
        def do_export(text):
            path = export_results(text)
            return gr.File(value=path, visible=True) if path else gr.File(visible=False)
        
        export_btn.click(
            fn=do_export,
            inputs=[batch_input],
            outputs=[csv_output]
        )
    
    return batch_input, validate_btn
