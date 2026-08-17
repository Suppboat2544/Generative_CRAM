"""
UI Utility Functions
====================
Drawing and formatting utilities.
"""

import numpy as np
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.Draw import rdMolDraw2D


def draw_molecule(smiles: str, size: tuple = (400, 400), highlight_cram: bool = True) -> np.ndarray:
    """
    Draw molecule with advanced visualization.
    
    Args:
        smiles: SMILES string
        size: Image dimensions
        highlight_cram: Highlight CRAM-relevant atoms
        
    Returns:
        NumPy array of molecule image
    """
    if not smiles or not smiles.strip():
        img = Image.new('RGB', size, color='#f8fafc')
        return np.array(img)
    
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            img = Image.new('RGB', size, color='#fef2f2')
            return np.array(img)
        
        AllChem.Compute2DCoords(mol)
        
        drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
        
        if highlight_cram:
            # Find carboxyl atoms
            cooh_pattern = Chem.MolFromSmarts('[CX3](=O)[O;H1,-1]')
            highlight_atoms = set()
            if cooh_pattern:
                matches = mol.GetSubstructMatches(cooh_pattern)
                for match in matches:
                    highlight_atoms.update(match)
            
            if highlight_atoms:
                colors = {i: (0.2, 0.7, 0.3) for i in highlight_atoms}
                drawer.DrawMolecule(mol, highlightAtoms=list(highlight_atoms), 
                                   highlightAtomColors=colors)
            else:
                drawer.DrawMolecule(mol)
        else:
            drawer.DrawMolecule(mol)
        
        drawer.FinishDrawing()
        
        import io
        png_data = drawer.GetDrawingText()
        img = Image.open(io.BytesIO(png_data))
        return np.array(img)
        
    except Exception as e:
        img = Image.new('RGB', size, color='#fef2f2')
        return np.array(img)


def format_validation(result: dict) -> str:
    """Format validation result as Markdown."""
    if result.get('is_cram'):
        status = "**Valid CRAM**"
    else:
        status = "**Not CRAM**"
    
    lines = [
        "## Validation Result",
        "",
        status,
        "",
        "---",
        "",
        "### Criteria Checks",
        "",
    ]
    
    checks = result.get('checks', {})
    for name, passed in checks.items():
        mark = "pass" if passed else "fail"
        lines.append(f"- [{mark}] {name}")
    
    return "\n".join(lines)


def format_properties(result: dict) -> str:
    """Format molecular properties as Markdown."""
    props = result.get('properties', {})
    if not props:
        return ""
    
    lines = [
        "### Molecular Properties",
        "",
        "| Property | Value |",
        "|----------|-------|",
    ]
    
    for key, value in props.items():
        if isinstance(value, float):
            lines.append(f"| {key} | {value:.2f} |")
        else:
            lines.append(f"| {key} | {value} |")
    
    return "\n".join(lines)
