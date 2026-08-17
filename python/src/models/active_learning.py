"""Notebook uncertainty / active-learning classes."""
from __future__ import annotations
from typing import List, Optional, Dict, Callable
import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem

# ══════════════════════════════════════════════════════════════════════════════
# ACTIVE LEARNING LOOP
# ══════════════════════════════════════════════════════════════════════════════
#
# Uncertainty-based sampling for efficient training:
# 1. Epistemic uncertainty (model dropout variance)
# 2. Reconstruction uncertainty (decoder confidence)
# 3. CRAM validity uncertainty (classification confidence)
# 4. Acquisition function (select informative samples)
# ══════════════════════════════════════════════════════════════════════════════

class UncertaintyEstimator:
    """
    Estimate model uncertainty for active learning.
    
    Methods:
    1. MC Dropout: Run model multiple times with dropout enabled
    2. Reconstruction variance: Measure output token probabilities
    3. Latent variance: Measure spread in latent space
    """
    
    def __init__(self, model, device, num_mc_samples=10):
        """
        Args:
            model: DeepCRAMVAE model
            device: Computing device
            num_mc_samples: Number of MC dropout samples
        """
        self.model = model
        self.device = device
        self.num_mc_samples = num_mc_samples
    
    def enable_dropout(self):
        """Enable dropout for MC estimation."""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()
    
    def disable_dropout(self):
        """Disable dropout."""
        self.model.eval()
    
    @torch.no_grad()
    def estimate_epistemic_uncertainty(self, batch):
        """
        Estimate epistemic uncertainty using MC Dropout.
        
        Returns: (uncertainty_scores, mean_latent, var_latent)
        - uncertainty_scores: Per-sample uncertainty (higher = more uncertain)
        - mean_latent: Mean latent representation
        - var_latent: Variance of latent representation
        """
        graph = batch['graph'].to(self.device)
        tokens = batch['tokens'].to(self.device)
        
        # Collect MC samples
        latent_samples = []
        logit_samples = []
        
        for _ in range(self.num_mc_samples):
            self.enable_dropout()
            
            # Forward pass
            outputs = self.model(graph, tokens)
            
            if isinstance(outputs, tuple):
                z, mu, logvar = outputs[:3]
                logits = outputs[3] if len(outputs) > 3 else None
            else:
                z = outputs.get('z')
                mu = outputs.get('mu')
                logits = outputs.get('logits')
            
            latent_samples.append(z.cpu())
            if logits is not None:
                logit_samples.append(logits.cpu())
        
        self.disable_dropout()
        
        # Stack samples
        latent_stack = torch.stack(latent_samples) # (num_samples, batch, latent_dim)
        
        # Calculate mean and variance
        mean_latent = latent_stack.mean(dim=0) # (batch, latent_dim)
        var_latent = latent_stack.var(dim=0) # (batch, latent_dim)
        
        # Uncertainty = sum of variance across latent dimensions
        uncertainty_scores = var_latent.sum(dim=-1) # (batch,)
        
        # Also calculate predictive entropy if logits available
        if logit_samples:
            logit_stack = torch.stack(logit_samples) # (num_samples, batch, seq, vocab)
            mean_probs = F.softmax(logit_stack, dim=-1).mean(dim=0)
            predictive_entropy = -torch.sum(
                mean_probs * torch.log(mean_probs + 1e-10), dim=-1
            ).mean(dim=-1) # (batch,)
            
            # Combine uncertainties
            uncertainty_scores = uncertainty_scores + predictive_entropy * 0.1
        
        return uncertainty_scores.numpy(), mean_latent, var_latent
    
    @torch.no_grad()
    def estimate_reconstruction_uncertainty(self, batch):
        """
        Estimate reconstruction uncertainty from decoder probabilities.
        
        Returns: Per-sample uncertainty scores
        """
        graph = batch['graph'].to(self.device)
        tokens = batch['tokens'].to(self.device)
        
        self.model.eval()
        outputs = self.model(graph, tokens)
        
        if isinstance(outputs, tuple) and len(outputs) > 3:
            logits = outputs[3]
        else:
            return np.zeros(tokens.size(0))
        
        # Calculate per-token confidence
        probs = F.softmax(logits, dim=-1)
        max_probs = probs.max(dim=-1).values # (batch, seq)
        
        # Uncertainty = 1 - mean confidence
        uncertainty = 1 - max_probs.mean(dim=-1) # (batch,)
        
        return uncertainty.cpu().numpy()
    
    def estimate_cram_uncertainty(self, smiles_list):
        """
        Estimate CRAM classification uncertainty.
        
        Uses variance in CRAM property predictions.
        """
        uncertainties = []
        
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                uncertainties.append(1.0)
                continue
            
            # Count COOH
            cooh = Chem.MolFromSmarts('[CX3](=O)[OX2H1]')
            num_cooh = len(mol.GetSubstructMatches(cooh)) if cooh else 0
            
            # CRAM requires 2+ COOH
            # Uncertainty is high if borderline (1 COOH)
            if num_cooh == 0:
                uncertainty = 0.3 # Clearly not CRAM
            elif num_cooh == 1:
                uncertainty = 1.0 # Very uncertain
            elif num_cooh == 2:
                uncertainty = 0.5 # Moderate uncertainty
            else:
                uncertainty = 0.2 # Likely CRAM
            
            uncertainties.append(uncertainty)
        
        return np.array(uncertainties)


class AcquisitionFunction:
    """
    Acquisition functions for active learning sample selection.
    
    Functions:
    1. Maximum uncertainty: Select most uncertain samples
    2. Diversity sampling: Select diverse samples in latent space
    3. Expected model change: Select samples that would change model most
    4. Combined: Weighted combination
    """
    
    @staticmethod
    def max_uncertainty(uncertainties, n_select):
        """Select samples with highest uncertainty."""
        indices = np.argsort(uncertainties)[-n_select:]
        return indices
    
    @staticmethod
    def diversity_sampling(latent_vectors, n_select, uncertainties=None):
        """
        Select diverse samples using k-means++ initialization.
        
        Args:
            latent_vectors: (n_samples, latent_dim) array
            n_select: Number of samples to select
            uncertainties: Optional uncertainty weights
        """
        from sklearn.cluster import KMeans
        
        n_samples = len(latent_vectors)
        if n_select >= n_samples:
            return np.arange(n_samples)
        
        # Use k-means to find diverse centers
        kmeans = KMeans(n_clusters=n_select, random_state=42, n_init=10)
        kmeans.fit(latent_vectors)
        
        # For each cluster, select the sample closest to center
        # (or most uncertain if uncertainties provided)
        selected = []
        for i in range(n_select):
            cluster_mask = kmeans.labels_ == i
            cluster_indices = np.where(cluster_mask)[0]
            
            if len(cluster_indices) == 0:
                continue
            
            if uncertainties is not None:
                # Select most uncertain in cluster
                cluster_uncertainties = uncertainties[cluster_indices]
                best_in_cluster = cluster_indices[np.argmax(cluster_uncertainties)]
            else:
                # Select closest to center
                center = kmeans.cluster_centers_[i]
                distances = np.linalg.norm(
                    latent_vectors[cluster_indices] - center, axis=1
                )
                best_in_cluster = cluster_indices[np.argmin(distances)]
            
            selected.append(best_in_cluster)
        
        return np.array(selected)
    
    @staticmethod
    def combined(uncertainties, latent_vectors, n_select, 
                 uncertainty_weight=0.6, diversity_weight=0.4):
        """
        Combined acquisition function.
        
        Balances uncertainty and diversity.
        """
        n_samples = len(uncertainties)
        
        # Normalize uncertainties
        norm_uncertainty = (uncertainties - uncertainties.min()) / (
            uncertainties.max() - uncertainties.min() + 1e-8
        )
        
        # Calculate diversity scores (distance to nearest neighbor)
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=2)
        nn.fit(latent_vectors)
        distances, _ = nn.kneighbors(latent_vectors)
        diversity_scores = distances[:, 1] # Distance to nearest neighbor
        norm_diversity = (diversity_scores - diversity_scores.min()) / (
            diversity_scores.max() - diversity_scores.min() + 1e-8
        )
        
        # Combined score
        combined_scores = (
            uncertainty_weight * norm_uncertainty +
            diversity_weight * norm_diversity
        )
        
        # Select top scoring samples
        indices = np.argsort(combined_scores)[-n_select:]
        return indices


class ActiveLearningLoop:
    """
    Complete active learning loop for CRAM training.
    
    Workflow:
    1. Train on initial labeled set
    2. Predict on unlabeled pool
    3. Estimate uncertainties
    4. Select most informative samples
    5. Add to training set
    6. Repeat
    """
    
    def __init__(self, model, optimizer, trainer, device,
                 initial_train_smiles, pool_smiles, vocab, max_len,
                 batch_size=32, samples_per_round=100):
        """
        Args:
            model: DeepCRAMVAE model
            optimizer: Optimizer
            trainer: IntegratedCRAMTrainer
            device: Device
            initial_train_smiles: Initial training SMILES
            pool_smiles: Pool of unlabeled SMILES
            vocab: SELFIES vocabulary
            max_len: Maximum sequence length
            batch_size: Batch size
            samples_per_round: Samples to add each round
        """
        self.model = model
        self.optimizer = optimizer
        self.trainer = trainer
        self.device = device
        
        self.train_smiles = list(initial_train_smiles)
        self.pool_smiles = list(pool_smiles)
        self.vocab = vocab
        self.max_len = max_len
        self.batch_size = batch_size
        self.samples_per_round = samples_per_round
        
        self.uncertainty_estimator = UncertaintyEstimator(model, device)
        
        self.history = {
            'round': [],
            'train_size': [],
            'pool_size': [],
            'val_loss': [],
            'mean_uncertainty': []
        }
    
    def create_dataloader(self, smiles_list):
        """Create DataLoader from SMILES list."""
        dataset = IntegratedCRAMDataset(
            smiles_list=smiles_list,
            vocab=self.vocab,
            max_len=self.max_len,
            use_augmentation=True,
            cache_features=True
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_integrated_batch,
            num_workers=0
        )
    
    def estimate_pool_uncertainties(self):
        """Estimate uncertainties for all pool samples."""
        pool_loader = self.create_dataloader(self.pool_smiles)
        
        all_uncertainties = []
        all_latents = []
        
        for batch in pool_loader:
            uncertainties, mean_latent, _ = \
                self.uncertainty_estimator.estimate_epistemic_uncertainty(batch)
            all_uncertainties.extend(uncertainties)
            all_latents.append(mean_latent.numpy())
        
        return np.array(all_uncertainties), np.vstack(all_latents)
    
    def select_samples(self, n_select=None):
        """
        Select most informative samples from pool.
        
        Returns: List of selected SMILES
        """
        if n_select is None:
            n_select = self.samples_per_round
        
        n_select = min(n_select, len(self.pool_smiles))
        
        if n_select == 0:
            return []
        
        # Estimate uncertainties
        uncertainties, latent_vectors = self.estimate_pool_uncertainties()
        
        # Use combined acquisition function
        selected_indices = AcquisitionFunction.combined(
            uncertainties, latent_vectors, n_select
        )
        
        selected_smiles = [self.pool_smiles[i] for i in selected_indices]
        
        return selected_smiles, selected_indices, uncertainties
    
    def update_sets(self, selected_smiles, selected_indices):
        """Move selected samples from pool to training set."""
        self.train_smiles.extend(selected_smiles)
        
        # Remove from pool (in reverse order to preserve indices)
        for idx in sorted(selected_indices, reverse=True):
            self.pool_smiles.pop(idx)
    
    def run(self, val_loader, num_rounds=5, epochs_per_round=10):
        """
        Run active learning loop.
        
        Args:
            val_loader: Validation data loader
            num_rounds: Number of active learning rounds
            epochs_per_round: Training epochs per round
        """
        print(f"\n{'='*70}")
        print(f"ACTIVE LEARNING - {num_rounds} rounds")
        print(f"{'='*70}")
        print(f"Initial training set: {len(self.train_smiles)}")
        print(f"Pool size: {len(self.pool_smiles)}")
        print(f"Samples per round: {self.samples_per_round}")
        
        for round_idx in range(num_rounds):
            print(f"\n--- Round {round_idx + 1}/{num_rounds} ---")
            
            # Create training loader
            train_loader = self.create_dataloader(self.train_smiles)
            
            # Train for this round
            for epoch in range(epochs_per_round):
                train_losses, beta = self.trainer.train_epoch(
                    train_loader, epoch, epochs_per_round
                )
            
            # Validate
            val_losses = self.trainer.validate(val_loader, 0, 1)
            
            # Select new samples
            if len(self.pool_smiles) > 0:
                selected_smiles, selected_indices, uncertainties = \
                    self.select_samples()
                
                mean_uncertainty = np.mean(uncertainties)
                
                # Update sets
                self.update_sets(selected_smiles, selected_indices)
            else:
                mean_uncertainty = 0.0
            
            # Log
            self.history['round'].append(round_idx + 1)
            self.history['train_size'].append(len(self.train_smiles))
            self.history['pool_size'].append(len(self.pool_smiles))
            self.history['val_loss'].append(val_losses['total'])
            self.history['mean_uncertainty'].append(mean_uncertainty)
            
            print(f" Val loss: {val_losses['total']:.4f}")
            print(f" Training set: {len(self.train_smiles)}")
            print(f" Pool remaining: {len(self.pool_smiles)}")
            print(f" Mean uncertainty: {mean_uncertainty:.4f}")
        
        print(f"\n{'='*70}")
        print(f"ACTIVE LEARNING COMPLETE")
        print(f"Final training set size: {len(self.train_smiles)}")
        print(f"{'='*70}")
        
        return self.history


# Summary
