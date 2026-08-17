"""
CVAE Generation Tab
===================
Property-conditioned molecule generation.
"""

import gradio as gr
import random
import tempfile
from datetime import datetime
import numpy as np
import pandas as pd

from .utils import draw_molecule


def create_cvae_tab():
    """Create the CVAE generation tab interface."""
    
    with gr.TabItem("🎯 CVAE Generate", id="cvae_generate"):
        gr.Markdown("""
        ### Property-Conditioned CVAE Generation
        
        Generate CRAM molecules with **specific property targets** using the 
        Conditional Variational Autoencoder (CVAE).
        
        ⚠️ **Requires trained CVAE checkpoint**
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🎛️ Property Conditions")
                
                cvae_count = gr.Slider(
                    minimum=10, maximum=100000, value=1000, step=100,
                    label="Target Count"
                )
                
                gr.Markdown("#### Carboxylic Acid Groups")
                with gr.Row():
                    cvae_cooh_min = gr.Slider(0, 10, value=2, step=1, label="Min COOH")
                    cvae_cooh_max = gr.Slider(0, 10, value=5, step=1, label="Max COOH")
                
                cvae_ring_class = gr.Dropdown(
                    choices=["none", "aliphatic", "aromatic", "mixed"],
                    value="aliphatic",
                    label="Ring Class"
                )
                
                gr.Markdown("#### Molecular Weight (Da)")
                with gr.Row():
                    cvae_mw_min = gr.Slider(100, 750, value=150, step=10, label="Min MW")
                    cvae_mw_max = gr.Slider(100, 750, value=500, step=10, label="Max MW")
                
                gr.Markdown("#### Aliphatic Carbon %")
                with gr.Row():
                    cvae_aliph_min = gr.Slider(0, 100, value=40, step=5, label="Min %")
                    cvae_aliph_max = gr.Slider(0, 100, value=90, step=5, label="Max %")
                
                cvae_benzene = gr.Checkbox(label="Target Benzene-Containing", value=False)
                
                with gr.Accordion("⚙️ Advanced", open=False):
                    cvae_temperature = gr.Slider(0.1, 2.0, value=0.8, step=0.1, label="Temperature")
                    cvae_batch_size = gr.Slider(32, 512, value=256, step=32, label="Batch Size")
                    cvae_checkpoint = gr.Textbox(label="Checkpoint Path", placeholder="models/cvae_best.pt")
                
                cvae_btn = gr.Button("🚀 Generate", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                gr.Markdown("### 📊 Results")
                cvae_results = gr.Markdown("Configure and click Generate...")
                cvae_download = gr.File(label="Download CSV", visible=True)
                
                gr.Markdown("### 🧬 Samples")
                cvae_gallery = gr.Image(label="", height=350, show_label=False)
        
        def generate_cvae(count, cooh_min, cooh_max, ring_class, mw_min, mw_max,
                         benzene, aliph_min, aliph_max, temperature, checkpoint, batch_size):
            """Generate molecules with CVAE."""
            try:
                import torch
                from src.models import PropertyConditionedCRAMVAE, create_cvae_model
                from src.tokenizer import SELFIESTokenizer
                from src.generators import FastPropertyConditionedGenerator, get_device
                import os
                
                # Check checkpoint
                if not checkpoint or not os.path.exists(checkpoint):
                    default_paths = ['models/cvae_best.pt', 'outputs/cvae_checkpoint.pt']
                    for path in default_paths:
                        if os.path.exists(path):
                            checkpoint = path
                            break
                
                if not checkpoint or not os.path.exists(checkpoint):
                    return "❌ No checkpoint found", None, None
                
                device = get_device()
                ckpt = torch.load(checkpoint, map_location=device)
                
                vocab = ckpt.get('vocab')
                if not vocab:
                    return "❌ Invalid checkpoint", None, None
                
                tokenizer = SELFIESTokenizer(max_len=100)
                tokenizer.token_to_idx = vocab
                tokenizer.idx_to_token = {v: k for k, v in vocab.items()}
                
                sos_idx = vocab.get('<SOS>', vocab.get('[SOS]', 3))
                eos_idx = vocab.get('<EOS>', vocab.get('[EOS]', 1))
                
                model = create_cvae_model(vocab_size=len(vocab))
                model.load_state_dict(ckpt['model_state_dict'])
                model.to(device)
                model.eval()
                
                generator = FastPropertyConditionedGenerator(
                    model, tokenizer, device, sos_idx, eos_idx
                )
                
                ring_map = {'none': 0, 'aliphatic': 1, 'aromatic': 2, 'mixed': 3}
                ring_idx = ring_map.get(ring_class.lower(), 1)
                
                def sampler():
                    return {
                        'n_cooh': random.randint(cooh_min, cooh_max),
                        'ring_class': ring_idx,
                        'aliphatic_pct': random.uniform(aliph_min, aliph_max),
                        'mw': random.uniform(mw_min, mw_max),
                        'has_benzene': 1 if benzene else 0
                    }
                
                df = generator.generate_unique(count, sampler, temperature, batch_size)
                
                if df is None or len(df) == 0:
                    return "❌ No valid molecules generated", None, None
                
                # Save CSV
                temp_file = tempfile.NamedTemporaryFile(
                    mode='w', suffix='.csv', delete=False,
                    prefix=f'cvae_{datetime.now().strftime("%Y%m%d_%H%M%S")}_'
                )
                df.to_csv(temp_file.name, index=False)
                
                # Gallery
                images = [draw_molecule(s, (250, 250)) for s in df['smiles'].head(6)]
                while len(images) < 6:
                    images.append(np.ones((250, 250, 3), dtype=np.uint8) * 248)
                row1 = np.hstack(images[:3])
                row2 = np.hstack(images[3:6])
                gallery = np.vstack([row1, row2])
                
                summary = f"## ✅ Generated {len(df)} molecules\n\n" \
                         f"COOH: {cooh_min}-{cooh_max}, Ring: {ring_class}, MW: {mw_min}-{mw_max}"
                
                return summary, temp_file.name, gallery
                
            except Exception as e:
                return f"❌ Error: {str(e)}", None, None
        
        cvae_btn.click(
            fn=generate_cvae,
            inputs=[
                cvae_count, cvae_cooh_min, cvae_cooh_max, cvae_ring_class,
                cvae_mw_min, cvae_mw_max, cvae_benzene, cvae_aliph_min,
                cvae_aliph_max, cvae_temperature, cvae_checkpoint, cvae_batch_size
            ],
            outputs=[cvae_results, cvae_download, cvae_gallery]
        )
