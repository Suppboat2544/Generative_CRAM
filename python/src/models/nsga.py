"""Notebook NSGA-II molecular optimization (optional pymoo)."""
from __future__ import annotations
from typing import List, Optional, Dict
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

try:
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import Problem
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.rnd import FloatRandomSampling
    from pymoo.optimize import minimize
    from pymoo.termination import get_termination
except ImportError: # pragma: no cover
    NSGA2 = SBX = PM = FloatRandomSampling = minimize = get_termination = None
    class Problem: # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError("pymoo is required for MolecularNSGAIIProblem")

class MolecularNSGAIIProblem(Problem):
    """
    Multi-Objective Optimization Problem for CRAM molecule generation

    Objectives (to minimize):
    1. Negative diversity (maximize diversity)
    2. Synthetic accessibility score (minimize difficulty)
    3. Negative CRAM compliance (maximize compliance)
    4. Deviation from target oxidation state
    5. Negative novelty (maximize uniqueness)
    """

    def __init__(self, model, device, latent_dim=128, n_samples=100,
                 target_oxidation_state=0.5, reference_molecules=None):
        self.model = model
        self.device = device
        self.latent_dim = latent_dim
        self.n_samples = n_samples
        self.target_oxidation_state = target_oxidation_state
        self.reference_molecules = reference_molecules or []

        # Define bounds for latent space optimization
        super().__init__(
            n_var=latent_dim, # Number of variables (latent dimensions)
            n_obj=5, # Number of objectives
            n_constr=0, # No explicit constraints
            xl=-3.0, # Lower bound for latent variables
            xu=3.0 # Upper bound for latent variables
        )

        print(f" NSGA-II Problem initialized:")
        print(f" - Latent dimensions: {latent_dim}")
        print(f" - Sample size: {n_samples}")
        print(f" - Target oxidation: {target_oxidation_state}")
        print(f" - Reference mols: {len(self.reference_molecules)}")

    def _evaluate(self, X, out, *args, **kwargs):
        """
        Evaluate objectives for population X (batch of latent vectors)
        X shape: (n_individuals, latent_dim)
        """
        n_individuals = X.shape[0]
        objectives = np.zeros((n_individuals, 5))

        for i, latent_vector in enumerate(X):
            try:
                # Convert to tensor and generate molecule
                z = torch.FloatTensor(latent_vector).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    # Generate SELFIES
                    generated_selfies = self.model.decode(z)
                    if isinstance(generated_selfies, list):
                        selfies = generated_selfies[0]
                    else:
                        selfies = generated_selfies

                    # Convert to molecule
                    smiles = sf.decoder(selfies)
                    mol = Chem.MolFromSmiles(smiles)

                    if mol is None:
                        # Penalty for invalid molecules
                        objectives[i] = [10.0, 10.0, 10.0, 10.0, 10.0]
                        continue

                    # Objective 1: Negative Diversity (maximize diversity = minimize negative)
                    diversity_score = self._calculate_diversity(mol)
                    objectives[i, 0] = -diversity_score

                    # Objective 2: Synthetic Accessibility (minimize)
                    sa_score = self._calculate_sa_score(mol)
                    objectives[i, 1] = sa_score

                    # Objective 3: Negative CRAM Compliance (maximize compliance)
                    cram_score = self._calculate_cram_compliance(mol)
                    objectives[i, 2] = -cram_score

                    # Objective 4: Oxidation State Deviation (minimize)
                    ox_deviation = self._calculate_oxidation_deviation(mol)
                    objectives[i, 3] = ox_deviation

                    # Objective 5: Negative Novelty (maximize novelty)
                    novelty_score = self._calculate_novelty(mol)
                    objectives[i, 4] = -novelty_score

            except Exception as e:
                # Handle errors with penalty
                objectives[i] = [10.0, 10.0, 10.0, 10.0, 10.0]

        out["F"] = objectives

    def _calculate_diversity(self, mol):
        """Calculate molecular diversity using Tanimoto distance"""
        if len(self.reference_molecules) == 0:
            return 0.5 # Neutral score if no references

        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)

        max_distance = 0.0
        for ref_mol in self.reference_molecules[:50]: # Limit for speed
            try:
                ref_fp = AllChem.GetMorganFingerprintAsBitVect(ref_mol, radius=2, nBits=2048)
                similarity = DataStructs.TanimotoSimilarity(fp, ref_fp)
                distance = 1.0 - similarity
                max_distance = max(max_distance, distance)
            except:
                continue

        return max_distance

    def _calculate_sa_score(self, mol):
        """Calculate synthetic accessibility score (1-10, lower is better)"""
        try:
            from rdkit.Chem import rdMolDescriptors
            # Simplified SA score based on complexity
            sa = rdMolDescriptors.CalcNumRotatableBonds(mol) / 10.0
            sa += rdMolDescriptors.CalcNumRings(mol) / 5.0
            sa += mol.GetNumAtoms() / 50.0
            return min(sa, 10.0)
        except:
            return 5.0 # Neutral score

    def _calculate_cram_compliance(self, mol):
        """Calculate CRAM compliance score (0-1)"""
        score = 0.0

        # Check COOH groups (essential for CRAM)
        cooh_pattern = Chem.MolFromSmarts('C(=O)[OH]')
        n_cooh = len(mol.GetSubstructMatches(cooh_pattern)) if cooh_pattern else 0
        score += min(n_cooh / 3.0, 1.0) * 0.3 # Ideal: 2-4 COOH groups

        # Check aliphatic carbons
        n_aliphatic = sum(1 for atom in mol.GetAtoms()
                         if atom.GetSymbol() == 'C' and atom.GetHybridization() == Chem.HybridizationType.SP3)
        aliphatic_ratio = n_aliphatic / max(mol.GetNumAtoms(), 1)
        score += (1.0 - abs(0.6 - aliphatic_ratio)) * 0.3 # Target: 60% aliphatic

        # Check molecular weight (CRAM typically 300-800 Da)
        mw = Descriptors.MolWt(mol)
        if 300 <= mw <= 800:
            score += 0.2

        # Check O/C ratio (CRAM typically 0.3-0.6)
        n_oxygen = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'O')
        n_carbon = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'C')
        if n_carbon > 0:
            oc_ratio = n_oxygen / n_carbon
            if 0.3 <= oc_ratio <= 0.6:
                score += 0.2

        return score

    def _calculate_oxidation_deviation(self, mol):
        """Calculate deviation from target oxidation state"""
        n_oxygen = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'O')
        n_carbon = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'C')

        if n_carbon == 0:
            return 1.0

        oxidation_state = n_oxygen / n_carbon
        deviation = abs(oxidation_state - self.target_oxidation_state)

        return deviation

    def _calculate_novelty(self, mol):
        """Calculate novelty compared to reference molecules"""
        if len(self.reference_molecules) == 0:
            return 1.0 # Maximally novel if no references

        # Use Murcko scaffold for novelty
        try:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            scaffold_fp = AllChem.GetMorganFingerprintAsBitVect(scaffold, radius=2, nBits=1024)

            max_similarity = 0.0
            for ref_mol in self.reference_molecules[:50]:
                try:
                    ref_scaffold = MurckoScaffold.GetScaffoldForMol(ref_mol)
                    ref_fp = AllChem.GetMorganFingerprintAsBitVect(ref_scaffold, radius=2, nBits=1024)
                    similarity = DataStructs.TanimotoSimilarity(scaffold_fp, ref_fp)
                    max_similarity = max(max_similarity, similarity)
                except:
                    continue

            novelty = 1.0 - max_similarity
            return novelty
        except:
            return 0.5 # Neutral novelty


class MolecularCrossover:
    """Custom crossover operator for latent space interpolation"""

    def __init__(self, prob=0.9, eta=15):
        self.prob = prob
        self.eta = eta

    def do(self, problem, X, **kwargs):
        """
        Perform crossover on parent population X
        """
        n_parents, n_var = X.shape
        n_offsprings = n_parents

        Y = np.full((n_offsprings, n_var), np.nan)

        for i in range(0, n_offsprings, 2):
            if i + 1 < n_offsprings:
                # Select two parents
                parent1 = X[i]
                parent2 = X[i + 1]

                if np.random.random() < self.prob:
                    # Latent space interpolation
                    alpha = np.random.beta(2, 2) # Bias toward middle
                    offspring1 = alpha * parent1 + (1 - alpha) * parent2
                    offspring2 = (1 - alpha) * parent1 + alpha * parent2
                else:
                    # No crossover
                    offspring1 = parent1.copy()
                    offspring2 = parent2.copy()

                Y[i] = offspring1
                Y[i + 1] = offspring2

        return Y


class MolecularMutation:
    """Custom mutation operator for latent space perturbation"""

    def __init__(self, prob=0.1, eta=20):
        self.prob = prob
        self.eta = eta

    def do(self, problem, X, **kwargs):
        """
        Perform mutation on population X
        """
        X_mutated = X.copy()

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                if np.random.random() < self.prob:
                    # Gaussian perturbation
                    perturbation = np.random.normal(0, 1.0 / self.eta)
                    X_mutated[i, j] += perturbation

                    # Clip to bounds
                    X_mutated[i, j] = np.clip(X_mutated[i, j], problem.xl, problem.xu)

        return X_mutated


def run_nsga2_optimization(model, device, reference_molecules,
                          n_generations=100, population_size=100,
                          target_oxidation_state=0.5):
    """
    Run NSGA-II multi-objective optimization

    Args:
        model: Trained PropertyConditionedTransformerVAE
        device: torch device
        reference_molecules: List of RDKit molecules for diversity calculation
        n_generations: Number of generations
        population_size: Population size
        target_oxidation_state: Target O/C ratio

    Returns:
        result: pymoo optimization result with Pareto front
    """

    print(" Starting NSGA-II Multi-Objective Optimization")
    print(f" - Population size: {population_size}")
    print(f" - Generations: {n_generations}")
    print(f" - Reference molecules: {len(reference_molecules)}")
    print(f" - Target oxidation: {target_oxidation_state}")

    # Create problem
    problem = MolecularNSGAIIProblem(
        model=model,
        device=device,
        latent_dim=128,
        n_samples=population_size,
        target_oxidation_state=target_oxidation_state,
        reference_molecules=reference_molecules
    )

    # Initialize NSGA-II algorithm
    algorithm = NSGA2(
        pop_size=population_size,
        sampling=FloatRandomSampling(), # Random initial population in latent space
        crossover=SBX(prob=0.9, eta=15), # Simulated Binary Crossover
        mutation=PM(prob=0.1, eta=20), # Polynomial Mutation
        eliminate_duplicates=True
    )

    # Termination criteria
    termination = get_termination("n_gen", n_generations)

    # Run optimization
    print("\n Running NSGA-II optimization...")
    result = minimize(
        problem,
        algorithm,
        termination,
        seed=42,
        verbose=True,
        save_history=True
    )

    print(f" - Solutions in Pareto front: {len(result.X)}")
    print(f" - Objective space coverage:")

    # Print objective statistics
    obj_names = ["Diversity", "SA Score", "CRAM Compliance", "Ox Deviation", "Novelty"]
    for i, name in enumerate(obj_names):
        obj_values = result.F[:, i]
        print(f" {name}: [{obj_values.min():.3f}, {obj_values.max():.3f}]")

    return result


def analyze_pareto_front(result, model, device):
    """
    Analyze and decode molecules from Pareto front

    Args:
        result: pymoo optimization result
        model: Trained model
        device: torch device

    Returns:
        pareto_molecules: List of (mol, objectives, latent_vector)
    """

    print("\n Analyzing Pareto Front Solutions")

    pareto_molecules = []

    for i, (latent_vector, objectives) in enumerate(zip(result.X, result.F)):
        try:
            # Decode latent vector
            z = torch.FloatTensor(latent_vector).unsqueeze(0).to(device)

            with torch.no_grad():
                generated_selfies = model.decode(z)
                if isinstance(generated_selfies, list):
                    selfies = generated_selfies[0]
                else:
                    selfies = generated_selfies

                smiles = sf.decoder(selfies)
                mol = Chem.MolFromSmiles(smiles)

                if mol is not None:
                    pareto_molecules.append((mol, objectives, latent_vector))
        except:
            continue


    # Print top 5 solutions for each objective
    obj_names = ["Diversity", "SA Score", "CRAM Compliance", "Ox Deviation", "Novelty"]

    for obj_idx, obj_name in enumerate(obj_names):
        print(f"\n Top 5 molecules by {obj_name}:")

        # Sort by this objective (handle negative objectives)
        if obj_idx in [0, 2, 4]: # Negative objectives (we want to minimize negative = maximize positive)
            sorted_mols = sorted(pareto_molecules, key=lambda x: x[1][obj_idx])
        else: # Positive objectives (minimize directly)
            sorted_mols = sorted(pareto_molecules, key=lambda x: x[1][obj_idx])

        for rank, (mol, obj, _) in enumerate(sorted_mols[:5], 1):
            smiles = Chem.MolToSmiles(mol)
            print(f" {rank}. {smiles[:50]}... | Objectives: {obj}")

    return pareto_molecules


# Store for later use
