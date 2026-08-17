"""Notebook curriculum learning classes."""
from __future__ import annotations
from typing import List, Optional, Dict
import numpy as np
from rdkit import Chem
import torch
import torch.nn as nn

# ══════════════════════════════════════════════════════════════════════════════
# CURRICULUM LEARNING FOR CRAM
# ══════════════════════════════════════════════════════════════════════════════
#
# Train from simple to complex CRAM molecules:
# Stage 1: Simple aliphatic chains + 2 COOH (easiest)
# Stage 2: Single alicyclic ring + 2 COOH
# Stage 3: Ring+chain hybrids + 2-3 COOH
# Stage 4: Complex multi-ring + multiple COOH (hardest)
# ══════════════════════════════════════════════════════════════════════════════

class CRAMComplexityAnalyzer:
    """
    Analyze CRAM molecule complexity for curriculum learning.
    
    Complexity factors:
    1. Number of atoms
    2. Number of rings (especially alicyclic)
    3. Number of COOH groups
    4. Ring-chain connectivity
    5. Branching degree
    """
    
    def __init__(self):
        """Initialize complexity patterns."""
        self.cooh_pattern = Chem.MolFromSmarts('[CX3](=O)[OX2H1,OX1-]')
        self.alicyclic_pattern = Chem.MolFromSmarts('[R;!a]')
    
    def get_complexity_score(self, mol):
        """
        Calculate complexity score for a molecule.
        
        Returns: (score, stage, details)
        - score: 0-100 complexity score
        - stage: 1-4 curriculum stage
        - details: Dict with complexity breakdown
        """
        if mol is None:
            return 0, 1, {}
        
        details = {}
        
        # 1. Basic size
        num_atoms = mol.GetNumAtoms()
        num_heavy = mol.GetNumHeavyAtoms()
        details['num_atoms'] = num_atoms
        details['num_heavy'] = num_heavy
        
        # 2. COOH count
        cooh_matches = mol.GetSubstructMatches(self.cooh_pattern) if self.cooh_pattern else []
        num_cooh = len(cooh_matches)
        details['num_cooh'] = num_cooh
        
        # 3. Ring analysis
        ring_info = mol.GetRingInfo()
        num_rings = ring_info.NumRings()
        details['num_rings'] = num_rings
        
        # Count alicyclic rings
        num_alicyclic = 0
        ring_sizes = []
        for ring in ring_info.AtomRings():
            ring_sizes.append(len(ring))
            is_aromatic = any(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring)
            if not is_aromatic:
                num_alicyclic += 1
        details['num_alicyclic'] = num_alicyclic
        details['ring_sizes'] = ring_sizes
        
        # 4. Branching
        max_degree = 0
        branch_points = 0
        for atom in mol.GetAtoms():
            degree = atom.GetDegree()
            if degree > 2:
                branch_points += 1
            max_degree = max(max_degree, degree)
        details['branch_points'] = branch_points
        details['max_degree'] = max_degree
        
        # 5. Calculate complexity score
        score = 0
        
        # Size contribution (0-20)
        score += min(num_heavy / 50 * 20, 20)
        
        # Ring contribution (0-30)
        score += min(num_rings * 7, 21)
        score += min(num_alicyclic * 3, 9)
        
        # COOH contribution (0-20)
        score += min(num_cooh * 5, 20)
        
        # Branching contribution (0-15)
        score += min(branch_points * 3, 15)
        
        # Ring-chain hybrid bonus (0-15)
        if num_alicyclic > 0 and branch_points > 0:
            score += 10
        if num_rings > 1:
            score += 5
        
        score = min(score, 100)
        details['score'] = score
        
        # 6. Determine curriculum stage
        if score < 25:
            stage = 1 # Simple aliphatic + 2 COOH
        elif score < 45:
            stage = 2 # Single ring + 2 COOH
        elif score < 70:
            stage = 3 # Ring+chain + 2-3 COOH
        else:
            stage = 4 # Complex multi-ring
        
        details['stage'] = stage
        
        return score, stage, details
    
    def classify_molecule(self, smiles):
        """Classify a SMILES string."""
        mol = Chem.MolFromSmiles(smiles)
        return self.get_complexity_score(mol)


def create_curriculum(dataset_smiles, num_stages=4):
    """
    Create curriculum stages from a dataset.
    
    Args:
        dataset_smiles: List of SMILES strings
        num_stages: Number of curriculum stages (default 4)
    
    Returns:
        Dict with stages, each containing list of SMILES
        
    Stages:
        Stage 1: Simple aliphatic chains + 2 COOH (easiest)
        Stage 2: Single alicyclic ring + 2 COOH
        Stage 3: Ring+chain hybrids + 2-3 COOH
        Stage 4: Complex multi-ring + multiple COOH (hardest)
    """
    analyzer = CRAMComplexityAnalyzer()
    
    # Analyze all molecules
    scored_molecules = []
    for smiles in dataset_smiles:
        score, stage, details = analyzer.classify_molecule(smiles)
        if score > 0: # Valid molecule
            scored_molecules.append({
                'smiles': smiles,
                'score': score,
                'stage': stage,
                'details': details
            })
    
    # Sort by complexity
    scored_molecules.sort(key=lambda x: x['score'])
    
    # Create stage buckets
    stages = {i: [] for i in range(1, num_stages + 1)}
    
    for mol_info in scored_molecules:
        stage = mol_info['stage']
        stages[stage].append(mol_info['smiles'])
    
    # Print statistics
    print(f" Curriculum created with {num_stages} stages:")
    for stage, smiles_list in stages.items():
        print(f" Stage {stage}: {len(smiles_list)} molecules")
    
    return stages


class CurriculumDataLoader:
    """
    Data loader that implements curriculum learning.
    
    Starts with easy molecules and progressively includes harder ones.
    """
    
    def __init__(self, stages, vocab, max_len=128, batch_size=32,
                 epochs_per_stage=10, overlap_ratio=0.3):
        """
        Args:
            stages: Dict from create_curriculum()
            vocab: SELFIES vocabulary
            max_len: Maximum sequence length
            batch_size: Batch size
            epochs_per_stage: Epochs before advancing to next stage
            overlap_ratio: Fraction of previous stage to keep
        """
        self.stages = stages
        self.vocab = vocab
        self.max_len = max_len
        self.batch_size = batch_size
        self.epochs_per_stage = epochs_per_stage
        self.overlap_ratio = overlap_ratio
        
        self.current_stage = 1
        self.current_epoch = 0
        self.num_stages = len(stages)
        
        # Create datasets for each stage
        self.stage_datasets = {}
        for stage, smiles_list in stages.items():
            if smiles_list:
                self.stage_datasets[stage] = IntegratedCRAMDataset(
                    smiles_list=smiles_list,
                    vocab=vocab,
                    max_len=max_len,
                    use_augmentation=True,
                    cache_features=True
                )
        
        print(f"CurriculumDataLoader initialized:")
        print(f" Stages: {self.num_stages}")
        print(f" Epochs per stage: {epochs_per_stage}")
        print(f" Overlap ratio: {overlap_ratio}")
    
    def get_current_dataset(self):
        """Get dataset for current curriculum stage."""
        # Collect molecules from current and previous stages
        all_smiles = []
        
        for stage in range(1, self.current_stage + 1):
            if stage in self.stages:
                if stage < self.current_stage:
                    # Include overlap_ratio of previous stages
                    stage_smiles = self.stages[stage]
                    num_keep = int(len(stage_smiles) * self.overlap_ratio)
                    all_smiles.extend(stage_smiles[:num_keep])
                else:
                    # Include all of current stage
                    all_smiles.extend(self.stages[stage])
        
        return IntegratedCRAMDataset(
            smiles_list=all_smiles,
            vocab=self.vocab,
            max_len=self.max_len,
            use_augmentation=True,
            cache_features=True
        )
    
    def get_loader(self):
        """Get DataLoader for current stage."""
        dataset = self.get_current_dataset()
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_integrated_batch,
            num_workers=0
        )
    
    def step(self):
        """
        Advance curriculum after each epoch.
        
        Returns: True if advanced to new stage
        """
        self.current_epoch += 1
        
        if self.current_epoch >= self.epochs_per_stage:
            if self.current_stage < self.num_stages:
                self.current_stage += 1
                self.current_epoch = 0
                print(f"\n Advanced to Stage {self.current_stage}")
                return True
        
        return False
    
    def get_stage_info(self):
        """Get current stage information."""
        return {
            'stage': self.current_stage,
            'epoch_in_stage': self.current_epoch,
            'total_stages': self.num_stages,
            'molecules_in_stage': len(self.stages.get(self.current_stage, []))
        }


class CurriculumTrainer:
    """
    Trainer with curriculum learning integration.
    """
    
    def __init__(self, model, optimizer, scheduler, device,
                 curriculum_loader, base_trainer):
        """
        Args:
            model: DeepCRAMVAE model
            optimizer: Optimizer
            scheduler: LR scheduler
            device: Device
            curriculum_loader: CurriculumDataLoader
            base_trainer: IntegratedCRAMTrainer instance
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.curriculum = curriculum_loader
        self.trainer = base_trainer
        
        self.history = {
            'stage': [],
            'train_loss': [],
            'val_loss': []
        }
    
    def train(self, val_loader, total_epochs, checkpoint_path='curriculum_model.pt'):
        """
        Train with curriculum learning.
        
        Args:
            val_loader: Validation data loader
            total_epochs: Total training epochs
            checkpoint_path: Path to save best model
        """
        best_val_loss = float('inf')
        
        print(f"\n{'='*70}")
        print(f"CURRICULUM TRAINING - {total_epochs} epochs")
        print(f"{'='*70}")
        
        for epoch in range(total_epochs):
            # Get current stage loader
            train_loader = self.curriculum.get_loader()
            stage_info = self.curriculum.get_stage_info()
            
            # Train one epoch
            train_losses, beta = self.trainer.train_epoch(
                train_loader, epoch, total_epochs
            )
            
            # Validate
            val_losses = self.trainer.validate(val_loader, epoch, total_epochs)
            
            # Log
            self.history['stage'].append(stage_info['stage'])
            self.history['train_loss'].append(train_losses['total'])
            self.history['val_loss'].append(val_losses['total'])
            
            # Print progress
            if (epoch + 1) % 5 == 0:
                print(f"\nEpoch {epoch+1}/{total_epochs} (Stage {stage_info['stage']})")
                print(f" Train: {train_losses['total']:.4f}, Val: {val_losses['total']:.4f}")
            
            # Advance curriculum
            self.curriculum.step()
            
            # Save best model (weights + vocab + configs)
            if val_losses['total'] < best_val_loss:
                best_val_loss = val_losses['total']
                save_full_checkpoint(
                    checkpoint_path,
                    self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    val_loss=best_val_loss,
                    history=self.history,
                    extra={'stage': stage_info['stage']},
                )
        
        print(f"\n{'='*70}")
        print(f"CURRICULUM TRAINING COMPLETE")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"{'='*70}")
        
        return self.history
