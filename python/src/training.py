"""
CRAM Training Module
====================

Training utilities for DeepCRAMVAE model.
"""

from typing import Dict, Optional, List, Callable
import os
import time
from datetime import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast
from tqdm.auto import tqdm

from src.utils import get_device


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross entropy loss with label smoothing.
    
    Improves generalization by softening one-hot targets.
    """
    
    def __init__(self, smoothing: float = 0.1, ignore_index: int = 0):
        super().__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute label-smoothed cross entropy.
        
        Args:
            logits: [batch, seq_len, vocab_size]
            targets: [batch, seq_len]
            
        Returns:
            Loss scalar
        """
        vocab_size = logits.size(-1)
        logits = logits.reshape(-1, vocab_size)
        targets = targets.reshape(-1)
        
        # Create smoothed distribution
        smooth_targets = torch.zeros_like(logits).scatter_(
            -1, targets.unsqueeze(-1), 1.0
        )
        smooth_targets = smooth_targets * (1 - self.smoothing) + self.smoothing / vocab_size
        
        # Mask padding
        mask = (targets != self.ignore_index).float()
        
        # Compute loss
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(smooth_targets * log_probs).sum(dim=-1)
        loss = (loss * mask).sum() / (mask.sum() + 1e-8)
        
        return loss


class CRAMTrainer:
    """
    Trainer for DeepCRAMVAE model.
    
    Features:
    - Multi-task loss (reconstruction + KL + property prediction)
    - Mixed precision training (FP16/BF16)
    - Gradient clipping and accumulation
    - Learning rate scheduling
    - Checkpoint saving/loading
    - Early stopping
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler = None,
        device = None,
        output_dir: str = 'outputs',
        use_amp: bool = True,
        gradient_clip: float = 1.0,
        accumulation_steps: int = 1,
        loss_weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: DeepCRAMVAE model
            train_loader: Training dataloader
            val_loader: Validation dataloader
            optimizer: Optimizer (default: AdamW)
            scheduler: LR scheduler
            device: torch device
            output_dir: Output directory
            use_amp: Use automatic mixed precision
            gradient_clip: Gradient clipping value
            accumulation_steps: Gradient accumulation steps
            loss_weights: Loss component weights
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or get_device()
        self.output_dir = output_dir
        self.use_amp = use_amp and self.device.type == 'cuda'
        self.gradient_clip = gradient_clip
        self.accumulation_steps = accumulation_steps
        
        self.loss_weights = loss_weights or {
            'reconstruction': 1.0,
            'kl': 0.1,
            'is_cram': 0.2,
            'num_cooh': 0.1
        }
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Optimizer
        self.optimizer = optimizer or AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        # Scheduler
        self.scheduler = scheduler or CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2
        )
        
        # Mixed precision scaler
        self.scaler = GradScaler() if self.use_amp else None
        
        # Label smoothing loss
        self.recon_criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'best_val_loss': float('inf'),
            'best_epoch': 0
        }
        
        print(f"CRAMTrainer initialized on {self.device}")
        print(f"  AMP: {self.use_amp}")
        print(f"  Gradient clip: {self.gradient_clip}")
        print(f"  Accumulation steps: {self.accumulation_steps}")
    
    def _compute_loss(
        self,
        output: Dict,
        batch: Dict
    ) -> tuple:
        """Compute multi-task loss."""
        losses = {}
        
        # Reconstruction loss
        if 'logits' in output:
            logits = output['logits'][:, :-1]
            targets = batch['tokens'][:, 1:]
            losses['reconstruction'] = self.recon_criterion(logits, targets)
        
        # KL divergence
        mu, logvar = output['mu'], output['logvar']
        losses['kl'] = -0.5 * torch.mean(
            1 + logvar - mu.pow(2) - logvar.exp()
        )
        
        # Property prediction losses (optional heads)
        preds = output.get('predictions') or {}
        if 'is_cram' in batch and 'is_cram' in preds:
            pred = preds['is_cram'].squeeze()
            target = batch['is_cram']
            losses['is_cram'] = F.binary_cross_entropy(pred, target)
        
        # Total weighted loss
        total = sum(
            self.loss_weights.get(k, 0.1) * v 
            for k, v in losses.items()
        )
        
        return total, losses
    
    def train_epoch(self, epoch: int) -> Dict:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        component_losses = {}
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for step, batch in enumerate(pbar):
            # Move to device
            graph = batch['graph'].to(self.device)
            tokens = batch['tokens'].to(self.device)
            node_feats = graph.ndata['feat']
            batch_size = tokens.size(0)
            properties = {
                'n_cooh': torch.full((batch_size,), 2, dtype=torch.long, device=self.device),
                'ring_class': torch.full((batch_size,), 3, dtype=torch.long, device=self.device),
                'aliphatic_pct': torch.full((batch_size,), 0.7, device=self.device),
                'mw': torch.full((batch_size,), 0.4, device=self.device),
                'has_benzene': torch.zeros(batch_size, device=self.device),
            }
            batch_device = {
                'tokens': tokens,
                'is_cram': batch.get('is_cram', torch.zeros(batch_size)).to(self.device)
            }
            
            # Forward pass with AMP
            if self.use_amp:
                with autocast():
                    logits, mu, logvar = self.model(graph, node_feats, properties, tokens)
                    output = {'logits': logits, 'mu': mu, 'logvar': logvar}
                    loss, losses = self._compute_loss(output, batch_device)
                    loss = loss / self.accumulation_steps
                
                self.scaler.scale(loss).backward()
                
                if (step + 1) % self.accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.gradient_clip
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                logits, mu, logvar = self.model(graph, node_feats, properties, tokens)
                output = {'logits': logits, 'mu': mu, 'logvar': logvar}
                loss, losses = self._compute_loss(output, batch_device)
                loss = loss / self.accumulation_steps
                
                loss.backward()
                
                if (step + 1) % self.accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            # Track losses
            total_loss += loss.item() * self.accumulation_steps
            for k, v in losses.items():
                component_losses[k] = component_losses.get(k, 0) + v.item()
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Average losses
        n_batches = len(self.train_loader)
        avg_loss = total_loss / n_batches
        avg_components = {k: v / n_batches for k, v in component_losses.items()}
        
        return {'loss': avg_loss, **avg_components}
    
    @torch.no_grad()
    def validate(self) -> Dict:
        """Validate model."""
        self.model.eval()
        total_loss = 0
        component_losses = {}
        
        for batch in tqdm(self.val_loader, desc="Validation"):
            graph = batch['graph'].to(self.device)
            tokens = batch['tokens'].to(self.device)
            node_feats = graph.ndata['feat']
            batch_size = tokens.size(0)
            properties = {
                'n_cooh': torch.full((batch_size,), 2, dtype=torch.long, device=self.device),
                'ring_class': torch.full((batch_size,), 3, dtype=torch.long, device=self.device),
                'aliphatic_pct': torch.full((batch_size,), 0.7, device=self.device),
                'mw': torch.full((batch_size,), 0.4, device=self.device),
                'has_benzene': torch.zeros(batch_size, device=self.device),
            }
            batch_device = {
                'tokens': tokens,
                'is_cram': batch.get('is_cram', torch.zeros(batch_size)).to(self.device)
            }
            
            logits, mu, logvar = self.model(graph, node_feats, properties, tokens)
            output = {'logits': logits, 'mu': mu, 'logvar': logvar}
            loss, losses = self._compute_loss(output, batch_device)
            
            total_loss += loss.item()
            for k, v in losses.items():
                component_losses[k] = component_losses.get(k, 0) + v.item()
        
        n_batches = len(self.val_loader)
        avg_loss = total_loss / n_batches
        avg_components = {k: v / n_batches for k, v in component_losses.items()}
        
        return {'loss': avg_loss, **avg_components}
    
    def train(
        self,
        num_epochs: int,
        early_stopping: int = 10,
        save_every: int = 5
    ) -> Dict:
        """
        Full training loop.
        
        Args:
            num_epochs: Number of epochs
            early_stopping: Patience for early stopping
            save_every: Save checkpoint every N epochs
            
        Returns:
            Training history
        """
        print(f"\n{'='*60}")
        print(f"Starting training for {num_epochs} epochs")
        print(f"{'='*60}")
        
        patience_counter = 0
        
        for epoch in range(1, num_epochs + 1):
            start_time = time.time()
            
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate()
            
            # Update scheduler
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_metrics['loss'])
            else:
                self.scheduler.step()
            
            # Record history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['learning_rate'].append(
                self.optimizer.param_groups[0]['lr']
            )
            
            # Check for best model
            if val_metrics['loss'] < self.history['best_val_loss']:
                self.history['best_val_loss'] = val_metrics['loss']
                self.history['best_epoch'] = epoch
                patience_counter = 0
                
                # Save best model
                self.save_checkpoint('best_model.pt', epoch, val_metrics)
            else:
                patience_counter += 1
            
            # Log progress
            elapsed = time.time() - start_time
            print(f"\nEpoch {epoch}/{num_epochs} ({elapsed:.1f}s)")
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Periodic checkpoint
            if epoch % save_every == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pt', epoch, val_metrics)
            
            # Early stopping
            if patience_counter >= early_stopping:
                print(f"\nEarly stopping at epoch {epoch}")
                break
        
        print(f"\n{'='*60}")
        print(f"Training complete!")
        print(f"Best val loss: {self.history['best_val_loss']:.4f} at epoch {self.history['best_epoch']}")
        print(f"{'='*60}")
        
        return self.history
    
    def save_checkpoint(
        self,
        filename: str,
        epoch: int,
        metrics: Dict
    ):
        """Save model checkpoint."""
        path = os.path.join(self.output_dir, filename)
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'history': self.history
        }, path)
        
        print(f"  Saved checkpoint: {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.history = checkpoint.get('history', self.history)
        
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint['epoch']


def train_cram_model(
    model: nn.Module,
    train_loader,
    val_loader,
    num_epochs: int = 50,
    learning_rate: float = 1e-4,
    output_dir: str = 'outputs',
    device = None
) -> Dict:
    """
    Convenience function to train a CRAM model.
    
    Args:
        model: DeepCRAMVAE model
        train_loader: Training dataloader
        val_loader: Validation dataloader
        num_epochs: Number of epochs
        learning_rate: Learning rate
        output_dir: Output directory
        device: Torch device
        
    Returns:
        Training history
    """
    device = device or get_device()
    
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01
    )
    
    trainer = CRAMTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        output_dir=output_dir
    )
    
    return trainer.train(num_epochs=num_epochs)
