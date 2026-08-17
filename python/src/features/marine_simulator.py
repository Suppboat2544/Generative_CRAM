"""Notebook MarineEnvironment and EnhancedMarineDOMSimulator."""
from __future__ import annotations
from typing import List, Optional, Dict
import numpy as np
from rdkit import Chem

# Enhanced Marine DOM Simulation with Environmental Factors

import numpy as np
from scipy.integrate import odeint
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class MarineEnvironment:
    """Marine environment parameters"""
    salinity: float = 35.0 # psu (practical salinity units)
    temperature: float = 20.0 # °C
    ph: float = 8.1 # pH
    uv_intensity: float = 1.0 # Relative UV intensity (0-1)
    oxygen_saturation: float = 0.9 # Fraction (0-1)
    bacterial_activity: float = 0.7 # Relative activity (0-1)
    season: str = "summer" # summer, winter, spring, fall


class EnhancedMarineDOMSimulator:
    """
    Comprehensive Marine DOM Degradation Simulator

    Simulates multiple degradation pathways:
    1. Photo-oxidation (UV-mediated)
    2. Bacterial enzymatic degradation
    3. Auto-oxidation (O2-mediated)
    4. Salinity-dependent chemistry
    """

    def __init__(self, environment: MarineEnvironment = None):
        self.env = environment or MarineEnvironment()

        # Degradation rate constants (day^-1)
        self.base_rates = {
            'photo_oxidation': 0.05, # UV-mediated
            'bacterial_hydrolysis': 0.03, # Enzyme-mediated
            'auto_oxidation': 0.01, # O2-mediated
            'condensation': 0.002, # Polymerization
        }

        # Enzyme markers for bacterial activity
        self.enzyme_markers = {
            'beta_glucosidase': 0.0, # Carbohydrate degradation
            'aminopeptidase': 0.0, # Protein degradation
            'phosphatase': 0.0, # P-containing compound degradation
            'lipase': 0.0, # Lipid degradation
        }

        print(" Enhanced Marine DOM Simulator initialized")
        print(f" - Salinity: {self.env.salinity} psu")
        print(f" - Temperature: {self.env.temperature} °C")
        print(f" - pH: {self.env.ph}")
        print(f" - UV intensity: {self.env.uv_intensity}")
        print(f" - Season: {self.env.season}")

    def calculate_salinity_effect(self, salinity: float) -> float:
        """
        Calculate salinity effect on oxidation rates
        Higher salinity generally increases ionic strength and can affect rates

        Returns: Rate modification factor (0.5 - 1.5)
        """
        # Normalized to seawater (35 psu)
        normalized_salinity = salinity / 35.0

        # Salinity increases ionic strength, affecting reaction kinetics
        # Empirical relationship: rate increases with √(ionic strength)
        effect = 0.7 + 0.8 * np.sqrt(normalized_salinity)

        return np.clip(effect, 0.5, 1.5)

    def calculate_uv_degradation(self, mol, time_days: float) -> Dict[str, float]:
        """
        Simulate UV-mediated photo-oxidation over time

        UV-A (315-400 nm): Indirect photochemistry via ROS
        UV-B (280-315 nm): Direct photolysis

        Returns: Degradation products and rates
        """
        # Calculate chromophores (aromatic systems absorb UV)
        n_aromatic = sum(1 for atom in mol.GetAtoms()
                        if atom.GetIsAromatic())
        n_conjugated = Descriptors.NumAromaticRings(mol)

        # UV absorption cross-section proxy
        uv_absorption = (n_aromatic * 0.3 + n_conjugated * 0.5)

        # Photo-oxidation rate (enhanced by UV intensity)
        k_photo = self.base_rates['photo_oxidation'] * self.env.uv_intensity
        k_photo *= (1.0 + uv_absorption) # Chromophores enhance absorption

        # Seasonal modulation
        seasonal_factors = {
            'summer': 1.5, # High UV
            'spring': 1.2,
            'fall': 0.8,
            'winter': 0.5 # Low UV
        }
        k_photo *= seasonal_factors.get(self.env.season, 1.0)

        # Calculate degradation extent
        remaining_fraction = np.exp(-k_photo * time_days)
        degraded_fraction = 1.0 - remaining_fraction

        # Photo-oxidation products
        products = {
            'carboxylic_acids': degraded_fraction * 0.4, # COOH formation
            'quinones': degraded_fraction * 0.2, # Oxidized aromatics
            'peroxides': degraded_fraction * 0.15, # Intermediate ROS
            'co2': degraded_fraction * 0.25, # Complete mineralization
        }

        return {
            'remaining': remaining_fraction,
            'products': products,
            'rate_constant': k_photo
        }

    def calculate_bacterial_degradation(self, mol, time_days: float) -> Dict[str, float]:
        """
        Simulate bacterial enzymatic degradation

        Key enzymes:
        - β-glucosidase: Breaks glycosidic bonds
        - Aminopeptidase: Cleaves peptide bonds
        - Phosphatase: Removes phosphate groups
        - Lipase: Hydrolyzes ester bonds
        """
        # Identify degradable functional groups
        patterns = {
            'glycosidic': Chem.MolFromSmarts('COC'), # C-O-C
            'ester': Chem.MolFromSmarts('C(=O)O'), # Ester
            'amide': Chem.MolFromSmarts('C(=O)N'), # Amide/peptide
            'phosphate': Chem.MolFromSmarts('[P](=O)([O])[O]'), # Phosphate
        }

        substrate_scores = {}
        for name, pattern in patterns.items():
            if pattern:
                matches = mol.GetSubstructMatches(pattern)
                substrate_scores[name] = len(matches)
            else:
                substrate_scores[name] = 0

        # Calculate enzyme-specific activities
        self.enzyme_markers['beta_glucosidase'] = substrate_scores['glycosidic'] * self.env.bacterial_activity
        self.enzyme_markers['lipase'] = substrate_scores['ester'] * self.env.bacterial_activity
        self.enzyme_markers['aminopeptidase'] = substrate_scores['amide'] * self.env.bacterial_activity
        self.enzyme_markers['phosphatase'] = substrate_scores['phosphate'] * self.env.bacterial_activity

        # Overall bacterial degradation rate
        k_bacterial = self.base_rates['bacterial_hydrolysis'] * self.env.bacterial_activity

        # Temperature dependence (Q10 = 2 for biological processes)
        temp_factor = 2.0 ** ((self.env.temperature - 20.0) / 10.0)
        k_bacterial *= temp_factor

        # Calculate degradation
        remaining_fraction = np.exp(-k_bacterial * time_days)
        degraded_fraction = 1.0 - remaining_fraction

        return {
            'remaining': remaining_fraction,
            'degraded': degraded_fraction,
            'rate_constant': k_bacterial,
            'enzyme_activities': self.enzyme_markers.copy()
        }

    def calculate_auto_oxidation(self, mol, time_days: float) -> Dict[str, float]:
        """
        Simulate auto-oxidation (O2-mediated) in seawater

        Depends on:
        - Oxygen saturation
        - Oxidizable sites (aliphatic C-H, benzylic positions)
        - Salinity (ionic strength effects)
        """
        # Count oxidizable positions
        n_aliphatic_ch = sum(1 for atom in mol.GetAtoms()
                            if atom.GetSymbol() == 'C' and
                            atom.GetHybridization() == Chem.HybridizationType.SP3 and
                            atom.GetTotalNumHs() > 0)

        # Auto-oxidation rate
        k_auto = self.base_rates['auto_oxidation'] * self.env.oxygen_saturation
        k_auto *= (1.0 + n_aliphatic_ch * 0.1) # More C-H = faster oxidation

        # Salinity effect
        salinity_factor = self.calculate_salinity_effect(self.env.salinity)
        k_auto *= salinity_factor

        # Calculate oxidation extent
        remaining_fraction = np.exp(-k_auto * time_days)
        oxidized_fraction = 1.0 - remaining_fraction

        return {
            'remaining': remaining_fraction,
            'oxidized': oxidized_fraction,
            'rate_constant': k_auto
        }

    def simulate_temporal_degradation(self, mol, time_points: List[float]) -> Dict[str, np.ndarray]:
        """
        Simulate degradation over time with coupled processes

        Args:
            mol: RDKit molecule
            time_points: List of time points (days)

        Returns:
            Dictionary with temporal profiles
        """
        results = {
            'time': np.array(time_points),
            'remaining_mass': [],
            'photo_oxidation': [],
            'bacterial_degradation': [],
            'auto_oxidation': [],
            'oxidation_state': [],
            'carboxyl_content': [],
        }

        # Initial properties
        initial_cooh = len(mol.GetSubstructMatches(Chem.MolFromSmarts('C(=O)[OH]')))
        initial_carbon = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'C')
        initial_oxygen = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'O')

        for t in time_points:
            # Calculate each degradation pathway
            photo_result = self.calculate_uv_degradation(mol, t)
            bacterial_result = self.calculate_bacterial_degradation(mol, t)
            auto_result = self.calculate_auto_oxidation(mol, t)

            # Combined remaining mass (competitive pathways)
            remaining = (photo_result['remaining'] *
                        bacterial_result['remaining'] *
                        auto_result['remaining'])

            # Oxidation state increases with degradation
            # Photo-oxidation and auto-oxidation increase O/C ratio
            oxidation_increase = (1.0 - remaining) * 0.3 # Up to 30% increase
            current_oxidation = (initial_oxygen + oxidation_increase * initial_carbon) / initial_carbon

            # COOH groups increase during oxidation
            cooh_increase = (1.0 - remaining) * 2.0 # Add ~2 COOH per molecule
            current_cooh = initial_cooh + cooh_increase

            results['remaining_mass'].append(remaining)
            results['photo_oxidation'].append(1.0 - photo_result['remaining'])
            results['bacterial_degradation'].append(bacterial_result['degraded'])
            results['auto_oxidation'].append(auto_result['oxidized'])
            results['oxidation_state'].append(current_oxidation)
            results['carboxyl_content'].append(current_cooh)

        # Convert to arrays
        for key in ['remaining_mass', 'photo_oxidation', 'bacterial_degradation',
                   'auto_oxidation', 'oxidation_state', 'carboxyl_content']:
            results[key] = np.array(results[key])

        return results

    def analyze_redox_chemistry(self, mol) -> Dict[str, float]:
        """
        Analyze redox-active functional groups and redox potential
        """
        # Redox-active groups
        quinone_pattern = Chem.MolFromSmarts('C1=CC(=O)C=CC1=O')
        phenol_pattern = Chem.MolFromSmarts('c[OH]')
        thiol_pattern = Chem.MolFromSmarts('[SH]')
        disulfide_pattern = Chem.MolFromSmarts('SS')

        n_quinone = len(mol.GetSubstructMatches(quinone_pattern)) if quinone_pattern else 0
        n_phenol = len(mol.GetSubstructMatches(phenol_pattern)) if phenol_pattern else 0
        n_thiol = len(mol.GetSubstructMatches(thiol_pattern)) if thiol_pattern else 0
        n_disulfide = len(mol.GetSubstructMatches(disulfide_pattern)) if disulfide_pattern else 0

        # Estimate redox potential (relative, arbitrary units)
        # Quinones: strong electron acceptors (positive potential)
        # Phenols: moderate electron donors (negative potential)
        # Thiols: strong electron donors (very negative potential)

        redox_score = (n_quinone * 0.5 - n_phenol * 0.2 - n_thiol * 0.8)

        return {
            'quinones': n_quinone,
            'phenols': n_phenol,
            'thiols': n_thiol,
            'disulfides': n_disulfide,
            'redox_potential': redox_score,
            'electron_donating_capacity': n_phenol + n_thiol * 2,
            'electron_accepting_capacity': n_quinone * 2,
        }


def run_marine_dom_simulation(mol, smiles: str,
                              salinity: float = 35.0,
                              uv_intensity: float = 1.0,
                              season: str = "summer",
                              time_days: np.ndarray = None):
    """
    Run comprehensive marine DOM simulation for a molecule

    Args:
        mol: RDKit molecule
        smiles: SMILES string
        salinity: Salinity in psu
        uv_intensity: Relative UV intensity (0-1)
        season: Season for UV modulation
        time_days: Time points for simulation (default: 0-180 days)

    Returns:
        Complete simulation results
    """
    if time_days is None:
        time_days = np.array([0, 1, 7, 14, 30, 60, 90, 120, 150, 180])

    # Create environment
    env = MarineEnvironment(
        salinity=salinity,
        uv_intensity=uv_intensity,
        season=season
    )

    # Initialize simulator
    simulator = EnhancedMarineDOMSimulator(environment=env)

    print(f"\n Simulating Marine DOM Degradation")
    print(f" Molecule: {smiles[:60]}...")

    # Run temporal simulation
    temporal_results = simulator.simulate_temporal_degradation(mol, time_days.tolist())

    # Analyze redox chemistry
    redox_results = simulator.analyze_redox_chemistry(mol)

    # Print summary
    print(f"\n Degradation Summary (180 days):")
    print(f" - Remaining mass: {temporal_results['remaining_mass'][-1]:.2%}")
    print(f" - Photo-oxidation: {temporal_results['photo_oxidation'][-1]:.2%}")
    print(f" - Bacterial degradation: {temporal_results['bacterial_degradation'][-1]:.2%}")
    print(f" - Auto-oxidation: {temporal_results['auto_oxidation'][-1]:.2%}")
    print(f" - Final O/C ratio: {temporal_results['oxidation_state'][-1]:.3f}")
    print(f" - Final COOH count: {temporal_results['carboxyl_content'][-1]:.1f}")

    print(f"\n Redox Chemistry:")
    print(f" - Redox potential: {redox_results['redox_potential']:.3f}")
    print(f" - Electron donors: {redox_results['electron_donating_capacity']}")
    print(f" - Electron acceptors: {redox_results['electron_accepting_capacity']}")

    return {
        'temporal': temporal_results,
        'redox': redox_results,
        'environment': env
    }


# Example: Simulate with different conditions
def compare_environmental_conditions(mol, smiles: str):
    """Compare degradation under different marine conditions"""

    conditions = [
        {'name': 'Tropical (High UV)', 'salinity': 36.0, 'uv_intensity': 1.5, 'season': 'summer'},
        {'name': 'Temperate (Medium UV)', 'salinity': 35.0, 'uv_intensity': 1.0, 'season': 'spring'},
        {'name': 'Arctic (Low UV)', 'salinity': 32.0, 'uv_intensity': 0.3, 'season': 'winter'},
        {'name': 'Coastal (Variable)', 'salinity': 30.0, 'uv_intensity': 1.2, 'season': 'summer'},
    ]

    print(f"\n Comparing Environmental Conditions")
    print(f" Molecule: {smiles[:60]}...")

    all_results = []

    for cond in conditions:
        print(f"\n{'='*60}")
        print(f" {cond['name']}")
        print(f"{'='*60}")

        result = run_marine_dom_simulation(
            mol, smiles,
            salinity=cond['salinity'],
            uv_intensity=cond['uv_intensity'],
            season=cond['season']
        )

        all_results.append({
            'condition': cond['name'],
            'results': result
        })

    return all_results
