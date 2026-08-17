"""
Molecule Display Component
==========================
Components for displaying molecule structures and properties.
"""

import gradio as gr
from typing import Optional, Dict, Any
import io
import base64


def draw_molecule_image(smiles: str, size: tuple = (300, 300)) -> Optional[str]:
    """
    Draw molecule and return as base64 image.
    
    Args:
        smiles: SMILES string
        size: Image size tuple
    
    Returns:
        Base64 encoded image string or None
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        img = Draw.MolToImage(mol, size=size)
        
        # Convert to base64
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.read()).decode()
        
        return f"data:image/png;base64,{img_base64}"
    except Exception:
        return None


def format_molecule_card(
    smiles: str,
    properties: Dict[str, Any],
    show_image: bool = True
) -> str:
    """
    Format molecule information as a Markdown card.
    
    Args:
        smiles: SMILES string
        properties: Dictionary of molecule properties
        show_image: Whether to include molecule image
    
    Returns:
        Markdown formatted string
    """
    card = f"### Molecule\n\n`{smiles}`\n\n"
    
    if properties:
        card += "| Property | Value |\n|----------|-------|\n"
        for key, value in properties.items():
            if isinstance(value, float):
                card += f"| {key} | {value:.3f} |\n"
            else:
                card += f"| {key} | {value} |\n"
    
    return card


def create_property_badges(properties: Dict[str, Any]) -> str:
    """
    Create property badges as Markdown.
    
    Args:
        properties: Dictionary of properties
    
    Returns:
        Markdown string with badges
    """
    badges = []
    
    if 'mw' in properties:
        mw = properties['mw']
        color = "green" if 200 <= mw <= 800 else "orange"
        badges.append(f"**MW:** {mw:.1f}")
    
    if 'cram_score' in properties:
        score = properties['cram_score']
        badges.append(f"**Score:** {score:.3f}")
    
    if 'valid' in properties:
        valid = properties['valid']
        status = " Valid" if valid else " Invalid"
        badges.append(status)
    
    return " | ".join(badges)


def create_molecule_gallery(smiles_list: list, columns: int = 4) -> str:
    """
    Create a gallery view of molecules.
    
    Args:
        smiles_list: List of SMILES strings
        columns: Number of columns
    
    Returns:
        HTML string for gallery
    """
    html = "<div style='display: grid; grid-template-columns: repeat({}, 1fr); gap: 10px;'>".format(columns)
    
    for i, smiles in enumerate(smiles_list[:20]): # Limit to 20
        img = draw_molecule_image(smiles, size=(150, 150))
        if img:
            html += f"""
            <div style='border: 1px solid #ddd; padding: 5px; text-align: center;'>
                <img src='{img}' style='max-width: 100%;'>
                <p style='font-size: 10px; word-break: break-all;'>{smiles[:30]}...</p>
            </div>
            """
    
    html += "</div>"
    return html
