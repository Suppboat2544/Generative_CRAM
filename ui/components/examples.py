"""
Example Molecules Component
===========================
Pre-defined example molecules for the UI.
"""

# Example CRAM-like molecules for validation
EXAMPLE_MOLECULES = [
    {
        "name": "Acetic Acid",
        "smiles": "CC(=O)O",
        "description": "Simple carboxylic acid"
    },
    {
        "name": "Benzoic Acid",
        "smiles": "c1ccccc1C(=O)O",
        "description": "Aromatic carboxylic acid"
    },
    {
        "name": "Succinic Acid",
        "smiles": "OC(=O)CCC(=O)O",
        "description": "Dicarboxylic acid"
    },
    {
        "name": "Citric Acid",
        "smiles": "OC(=O)CC(O)(CC(=O)O)C(=O)O",
        "description": "Tricarboxylic acid with hydroxyl"
    },
    {
        "name": "Glutaric Acid",
        "smiles": "OC(=O)CCCC(=O)O",
        "description": "5-carbon dicarboxylic acid"
    },
    {
        "name": "Adipic Acid",
        "smiles": "OC(=O)CCCCC(=O)O",
        "description": "6-carbon dicarboxylic acid"
    },
    {
        "name": "Malic Acid",
        "smiles": "OC(=O)CC(O)C(=O)O",
        "description": "Hydroxy dicarboxylic acid"
    },
    {
        "name": "Tartaric Acid",
        "smiles": "OC(=O)C(O)C(O)C(=O)O",
        "description": "Dihydroxy dicarboxylic acid"
    },
    {
        "name": "Cyclohexanecarboxylic Acid",
        "smiles": "C1CCCCC1C(=O)O",
        "description": "Alicyclic carboxylic acid"
    },
    {
        "name": "Adamantane-1-carboxylic Acid",
        "smiles": "C1C2CC3CC1CC(C2)(C3)C(=O)O",
        "description": "Polycyclic carboxylic acid"
    },
]

# Example seed molecules for generation
GENERATION_SEEDS = [
    {
        "name": "Simple Acid",
        "smiles": "CC(=O)O",
        "mutation_rate": 0.3
    },
    {
        "name": "Medium Acid",
        "smiles": "CCCC(=O)O",
        "mutation_rate": 0.4
    },
    {
        "name": "Cyclic Acid",
        "smiles": "C1CCCCC1C(=O)O",
        "mutation_rate": 0.3
    },
    {
        "name": "Diacid",
        "smiles": "OC(=O)CCC(=O)O",
        "mutation_rate": 0.35
    },
]

# Marine DOM reference molecules
MARINE_DOM_EXAMPLES = [
    {
        "name": "CRAM Reference 1",
        "smiles": "CC1(C)CCC2(C(=O)O)CCC1C2",
        "h_c_ratio": 1.6,
        "o_c_ratio": 0.3
    },
    {
        "name": "CRAM Reference 2",
        "smiles": "CC(C)(C(=O)O)CC(C(=O)O)C",
        "h_c_ratio": 1.8,
        "o_c_ratio": 0.4
    },
]


def get_example_by_name(name: str) -> dict:
    """Get an example molecule by name."""
    for mol in EXAMPLE_MOLECULES:
        if mol["name"].lower() == name.lower():
            return mol
    return None


def get_all_example_smiles() -> list:
    """Get list of all example SMILES."""
    return [mol["smiles"] for mol in EXAMPLE_MOLECULES]


def get_example_names() -> list:
    """Get list of all example names."""
    return [mol["name"] for mol in EXAMPLE_MOLECULES]
