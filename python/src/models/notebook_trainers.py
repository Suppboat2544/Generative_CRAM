"""Notebook EnhancedCRAMTrainer and IntegratedCRAMTrainer."""
from __future__ import annotations
from typing import List, Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem

# ============================================================================
# CRAM STRUCTURE LOSS AND TRAINING INTEGRATION
# ============================================================================
#
# This adds multi-task loss for deep CRAM learning:
# - Standard VAE losses (reconstruction, KL, TC)
# - CRAM feature prediction loss (10D chemical features)
# - CRAM structure prediction loss (5D ring-chain structure)
#
# The model learns to:
# 1. Reconstruct molecules accurately
# 2. Understand CRAM chemical properties
# 3. Capture ring-chain hybrid structures explicitly
# ============================================================================

def compute_enhanced_cram_loss(
    logits, targets,
    mu, logvar, z,
    discriminator,
    cram_features_pred, cram_features_target,
    cram_structure_pred, cram_structure_target,
    beta=0.01,
    gamma=1.0,
    alpha_cram=0.2,
    alpha_structure=0.15,
    pad_idx=0
):
    """
    Compute enhanced loss with CRAM structure awareness.
    
    Loss components:
    1. Reconstruction loss (cross-entropy on SELFIES tokens)
    2. KL divergence (regularize latent space)
    3. TC loss (disentangle latent dimensions)
    4. CRAM feature loss (predict chemical properties)
    5. CRAM structure loss (predict ring-chain structure) ← NEW!
    
    Args:
        logits: [batch, seq_len, vocab_size] model predictions
        targets: [batch, seq_len] ground truth tokens
        mu: Latent mean [batch, latent_dim]
        logvar: Latent log variance [batch, latent_dim]
        z: Sampled latent [batch, latent_dim]
        discriminator: TC discriminator network
        cram_features_pred: [batch, 10] predicted CRAM features
        cram_features_target: [batch, 10] target CRAM features
        cram_structure_pred: [batch, 5] predicted structure
        cram_structure_target: [batch, 5] target structure
        beta: Weight for KL term
        gamma: Weight for TC term
        alpha_cram: Weight for CRAM feature loss
        alpha_structure: Weight for CRAM structure loss
        pad_idx: Padding token index
        
    Returns:
        total_loss, loss_dict with all components
    """
    # 1. Reconstruction loss
    recon_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=pad_idx,
        reduction='mean'
    )
    
    # 2. KL divergence
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    # 3. TC loss (total correlation)
    batch_size = z.size(0)
    
    real_scores = discriminator(z)
    
    perm_z = z.clone()
    for dim in range(z.size(1)):
        perm_z[:, dim] = z[torch.randperm(batch_size), dim]
    perm_scores = discriminator(perm_z)
    
    tc_labels_real = torch.ones(batch_size, dtype=torch.long, device=z.device)
    tc_labels_perm = torch.zeros(batch_size, dtype=torch.long, device=z.device)
    
    tc_loss = 0.5 * (
        F.cross_entropy(real_scores, tc_labels_real) +
        F.cross_entropy(perm_scores, tc_labels_perm)
    )
    
    # 4. CRAM feature prediction loss
    cram_feature_loss = F.mse_loss(cram_features_pred, cram_features_target)
    
    # 5. CRAM structure prediction loss (NEW!)
    cram_structure_loss = F.mse_loss(cram_structure_pred, cram_structure_target)
    
    # Additional penalty: boost prediction of carboxyl count
    carboxyl_penalty = F.relu(
        cram_structure_target[:, 3] - cram_structure_pred[:, 3]
    ).mean() * 0.5
    
    # Additional bonus: reward alicyclic prediction when it should be there
    alicyclic_bonus = F.relu(
        cram_structure_pred[:, 0] * cram_structure_target[:, 0]
    ).mean() * 0.1
    
    # Total loss
    total_loss = (
        recon_loss +
        beta * kl_loss +
        gamma * tc_loss +
        alpha_cram * cram_feature_loss +
        alpha_structure * (cram_structure_loss + carboxyl_penalty - alicyclic_bonus)
    )
    
    loss_dict = {
        'total': total_loss.item(),
        'recon': recon_loss.item(),
        'kl': kl_loss.item(),
        'tc': tc_loss.item(),
        'cram_feature': cram_feature_loss.item(),
        'cram_structure': cram_structure_loss.item(),
        'carboxyl_penalty': carboxyl_penalty.item(),
        'alicyclic_bonus': alicyclic_bonus.item()
    }
    
    return total_loss, loss_dict


def extract_substructure_masks_from_graph(g):
    """
    Extract substructure masks from a DGL graph.
    
    Args:
        g: DGL graph with mask_* node data
        
    Returns:
        dict of substructure masks
    """
    masks = {}
    for key in g.ndata:
        if key.startswith('mask_'):
            sub_type = key.replace('mask_', '')
            masks[sub_type] = g.ndata[key]
    return masks


def extract_cram_targets_from_graph(g):
    """
    Extract CRAM structure targets from a DGL graph.
    
    Args:
        g: DGL graph with cram_structure_targets node data
        
    Returns:
        cram_structure_targets: [batch, 5]
    """
    if 'cram_structure_targets' in g.ndata:
        # Take mean across nodes (all nodes have same value for each graph)
        return dgl.mean_nodes(g, 'cram_structure_targets')
    else:
        return torch.zeros(g.batch_size, 5, device=g.device)


class EnhancedCRAMTrainer:
    """
    Training wrapper for the enhanced CRAM model.
    
    Handles:
    - Enhanced encoder with ring-chain awareness
    - Multi-task loss (reconstruction + CRAM features + structure)
    - Progress tracking with CRAM-specific metrics
    """
    
    def __init__(self, model, discriminator, optimizer, disc_optimizer, device='cpu'):
        self.model = model
        self.discriminator = discriminator
        self.optimizer = optimizer
        self.disc_optimizer = disc_optimizer
        self.device = device
        
        # Track CRAM-specific metrics
        self.cram_metrics_history = []
    
    def train_epoch(self, dataloader, beta=0.01, gamma=1.0, 
                    alpha_cram=0.2, alpha_structure=0.15, 
                    clip_grad=5.0):
        """
        Train one epoch with enhanced CRAM loss.
        
        Args:
            dataloader: Training data loader
            beta: KL weight
            gamma: TC weight
            alpha_cram: CRAM feature weight
            alpha_structure: CRAM structure weight
            clip_grad: Gradient clipping threshold
            
        Returns:
            dict with epoch metrics
        """
        self.model.train()
        self.discriminator.train()
        
        epoch_losses = {
            'total': 0, 'recon': 0, 'kl': 0, 'tc': 0,
            'cram_feature': 0, 'cram_structure': 0
        }
        num_batches = 0
        
        for batch in dataloader:
            graph = batch['graph'].to(self.device)
            selfies_tokens = batch['selfies_tokens'].to(self.device)
            properties = batch['properties'].to(self.device)
            
            # Extract substructure masks
            substructure_masks = extract_substructure_masks_from_graph(graph)
            
            # Extract CRAM targets
            cram_structure_targets = extract_cram_targets_from_graph(graph)
            
            # Get CRAM feature targets (if available)
            if 'cram_features_raw' in graph.ndata:
                cram_features_targets = dgl.mean_nodes(graph, 'cram_features_raw')
            else:
                cram_features_targets = torch.zeros(graph.batch_size, 10, device=self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Use enhanced encoder if available
            if hasattr(self.model.encoder, 'ring_chain_encoder'):
                mu, logvar, cram_pred, cram_struct_pred, hier_features = \
                    self.model.encoder(graph, substructure_masks)
                z = self.model.reparameterize(mu, logvar)
                
                # Decode
                prop_embed = self.model.encode_properties(properties)
                z_cond = torch.cat([z, prop_embed], dim=-1)
                z_proj = self.model.latent_to_hidden(z_cond)
                gate = self.model.latent_gate(z_proj)
                memory = (z_proj * gate).unsqueeze(1)
                
                batch_size, seq_len = selfies_tokens.size()
                tgt_embed = self.model.token_embed(selfies_tokens)
                pos_learned = self.model.pos_embed_learned[:, :seq_len, :]
                pos_sinusoidal = self.model.pos_embed_sinusoidal[:, :seq_len, :]
                tgt_embed = tgt_embed + 0.5 * (pos_learned + pos_sinusoidal)
                tgt_embed = self.model.dropout(tgt_embed)
                
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                    seq_len, device=selfies_tokens.device
                )
                decoder_out = self.model.decoder(tgt_embed, memory, tgt_mask=tgt_mask)
                logits = self.model.output_proj(decoder_out)
                
            else:
                # Fallback to standard forward
                z, mu, logvar, logits = self.model(graph, selfies_tokens, properties)
                cram_pred = self.model.encoder.cram_fc(self.model.encoder.last_hg)
                cram_struct_pred = torch.zeros(z.size(0), 5, device=self.device)
            
            # Compute enhanced loss
            loss, loss_dict = compute_enhanced_cram_loss(
                logits, selfies_tokens,
                mu, logvar, z,
                self.discriminator,
                cram_pred, cram_features_targets,
                cram_struct_pred, cram_structure_targets,
                beta=beta, gamma=gamma,
                alpha_cram=alpha_cram, alpha_structure=alpha_structure
            )
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad)
            self.optimizer.step()
            
            # Update discriminator
            self.disc_optimizer.zero_grad()
            tc_loss = loss_dict['tc']
            # Re-compute TC loss for discriminator update
            z_detached = z.detach()
            real_scores = self.discriminator(z_detached)
            perm_z = z_detached.clone()
            for dim in range(z_detached.size(1)):
                perm_z[:, dim] = z_detached[torch.randperm(z_detached.size(0)), dim]
            perm_scores = self.discriminator(perm_z)
            tc_labels_real = torch.ones(z.size(0), dtype=torch.long, device=z.device)
            tc_labels_perm = torch.zeros(z.size(0), dtype=torch.long, device=z.device)
            disc_loss = 0.5 * (
                F.cross_entropy(real_scores, tc_labels_real) +
                F.cross_entropy(perm_scores, tc_labels_perm)
            )
            disc_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), clip_grad)
            self.disc_optimizer.step()
            
            # Accumulate losses
            for key in epoch_losses:
                if key in loss_dict:
                    epoch_losses[key] += loss_dict[key]
            num_batches += 1
        
        # Average
        for key in epoch_losses:
            epoch_losses[key] /= max(num_batches, 1)
        
        return epoch_losses


print("")
print(" Components:")
print(" • compute_enhanced_cram_loss() - Multi-task loss with structure prediction")
print(" • extract_substructure_masks_from_graph() - Get ring/chain/junction masks")
print(" • extract_cram_targets_from_graph() - Get CRAM structure targets")
print(" • EnhancedCRAMTrainer - Training wrapper with CRAM metrics")
print("")
print(" Loss components:")
print(" 1. Reconstruction (cross-entropy)")
print(" 2. KL divergence")
print(" 3. TC loss (total correlation)")
print(" 4. CRAM feature loss (10D chemical properties)")
print(" 5. CRAM structure loss (5D ring-chain structure)")


# ══════════════════════════════════════════════════════════════════════════════
# INTEGRATED CRAM TRAINER
# ══════════════════════════════════════════════════════════════════════════════
# 
# Complete training loop with all deep improvements:
# - Multi-task loss: Reconstruction + KL + Properties + Contrastive
# - Cyclic β-annealing for stable KL
# - Gradient accumulation for effective batch size
# - CRAM validation every N epochs
# - Early stopping with patience
# ══════════════════════════════════════════════════════════════════════════════

class IntegratedCRAMTrainer:
    """
    Complete trainer for DeepCRAMVAE with all integrated losses.
    
    Loss components:
    1. Reconstruction loss (cross-entropy)
    2. KL divergence (with β-annealing)
    3. Property prediction loss (MSE)
    4. Contrastive loss (SupCon + center + margin)
    
    Features:
    - Cyclic β-annealing (4 cycles)
    - Gradient accumulation
    - Mixed precision training (if CUDA available)
    - CRAM validation metrics
    - Early stopping with patience
    """
    
    def __init__(self, model, optimizer, scheduler=None, device='cpu',
                 beta_min=0.00005, beta_max=0.003, n_cycles=4,
                 contrastive_weight=0.1, property_weight=0.5,
                 grad_accumulation=4, use_amp=True):
        """
        Args:
            model: DeepCRAMVAE model
            optimizer: Optimizer
            scheduler: Learning rate scheduler (optional)
            device: Device to train on
            beta_min/beta_max: β-annealing range
            n_cycles: Number of annealing cycles
            contrastive_weight: Weight for contrastive loss
            property_weight: Weight for property prediction loss
            grad_accumulation: Gradient accumulation steps
            use_amp: Use automatic mixed precision
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.n_cycles = n_cycles
        self.contrastive_weight = contrastive_weight
        self.property_weight = property_weight
        self.grad_accumulation = grad_accumulation
        
        # Mixed precision
        self.use_amp = use_amp and torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        # Contrastive learning module
        self.contrastive_module = ContrastiveCRAMLearning(
            latent_dim=64, temperature=0.07
        ).to(device) if 'ContrastiveCRAMLearning' in dir() else None
        
        # Training history
        self.history = {
            'train_loss': [], 'train_recon': [], 'train_kl': [],
            'train_contrastive': [], 'train_property': [],
            'val_loss': [], 'val_recon': [], 'val_kl': [],
            'learning_rates': [], 'betas': [],
            'val_n_active': [], 'val_frac_active': [], 'val_kl_sum_units': [],
        }
        
        print("IntegratedCRAMTrainer initialized:")
        print(f" Device: {device}")
        print(f" β range: [{beta_min:.5f}, {beta_max:.4f}]")
        print(f" Cycles: {n_cycles}")
        print(f" Contrastive weight: {contrastive_weight}")
        print(f" Property weight: {property_weight}")
        print(f" Gradient accumulation: {grad_accumulation}")
        print(f" Mixed precision: {'Enabled' if self.use_amp else 'Disabled'}")
    
    def get_beta(self, epoch, num_epochs):
        """Cyclic β-annealing schedule."""
        cycle_length = num_epochs // self.n_cycles
        position = epoch % cycle_length
        progress = position / cycle_length
        return self.beta_min + (self.beta_max - self.beta_min) * progress
    
    def compute_loss(self, batch, beta):
        """Compute all loss components."""
        graph = batch['graph'].to(self.device)
        tokens = batch['tokens'].to(self.device)
        is_cram = batch['is_cram'].to(self.device)
        mol_features = batch['mol_features'].to(self.device)
        
        # Forward pass
        outputs = self.model(graph, tokens)
        
        # Unpack outputs (handle different return formats)
        if isinstance(outputs, tuple):
            if len(outputs) >= 4:
                z, mu, logvar, logits = outputs[:4]
            else:
                z, mu, logvar = outputs[:3]
                logits = outputs[3] if len(outputs) > 3 else None
        else:
            z = outputs.get('z')
            mu = outputs.get('mu')
            logvar = outputs.get('logvar')
            logits = outputs.get('logits')
        
        losses = {}
        
        # 1. Reconstruction loss
        if logits is not None:
            # Shift for autoregressive loss
            target = tokens[:, 1:].contiguous()
            pred = logits[:, :-1, :].contiguous()
            recon_loss = F.cross_entropy(
                pred.view(-1, pred.size(-1)),
                target.view(-1),
                ignore_index=0 # Ignore padding
            )
            losses['recon'] = recon_loss
        else:
            losses['recon'] = torch.tensor(0.0, device=self.device)
        
        # 2. KL divergence
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        losses['kl'] = kl_loss
        
        # 3. Contrastive loss
        if self.contrastive_module is not None:
            contrastive_loss = self.contrastive_module(z, is_cram)
            losses['contrastive'] = contrastive_loss
        else:
            losses['contrastive'] = torch.tensor(0.0, device=self.device)
        
        # 4. Property prediction loss (if model has property head)
        if hasattr(self.model, 'property_predictor'):
            prop_pred = self.model.property_predictor(z)
            prop_loss = F.mse_loss(prop_pred, mol_features)
            losses['property'] = prop_loss
        else:
            losses['property'] = torch.tensor(0.0, device=self.device)
        
        # Total loss
        total_loss = (
            losses['recon'] +
            beta * losses['kl'] +
            self.contrastive_weight * losses['contrastive'] +
            self.property_weight * losses['property']
        )
        losses['total'] = total_loss
        
        return losses, z, mu, logvar
    
    def train_epoch(self, train_loader, epoch, num_epochs):
        """Train for one epoch."""
        self.model.train()
        
        beta = self.get_beta(epoch, num_epochs)
        
        total_losses = {k: 0.0 for k in ['total', 'recon', 'kl', 'contrastive', 'property']}
        num_batches = 0
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(train_loader):
            # Mixed precision context
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    losses, z, mu, logvar = self.compute_loss(batch, beta)
                    loss = losses['total'] / self.grad_accumulation
                
                self.scaler.scale(loss).backward()
            else:
                losses, z, mu, logvar = self.compute_loss(batch, beta)
                loss = losses['total'] / self.grad_accumulation
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.grad_accumulation == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
            
            # Accumulate losses
            for k, v in losses.items():
                total_losses[k] += v.item()
            num_batches += 1
        
        # Average losses
        avg_losses = {k: v / num_batches for k, v in total_losses.items()}
        
        # Update scheduler
        if self.scheduler is not None:
            self.scheduler.step()
        
        # Update history
        self.history['train_loss'].append(avg_losses['total'])
        self.history['train_recon'].append(avg_losses['recon'])
        self.history['train_kl'].append(avg_losses['kl'])
        self.history['train_contrastive'].append(avg_losses['contrastive'])
        self.history['train_property'].append(avg_losses['property'])
        self.history['betas'].append(beta)
        self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
        
        return avg_losses, beta
    
    @torch.no_grad()
    def validate(self, val_loader, epoch, num_epochs):
        """Validate the model."""
        self.model.eval()
        
        beta = self.get_beta(epoch, num_epochs)
        
        total_losses = {k: 0.0 for k in ['total', 'recon', 'kl', 'contrastive', 'property']}
        num_batches = 0
        unit_kl_chunks = []

        for batch in val_loader:
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    losses, z, mu, logvar = self.compute_loss(batch, beta)
            else:
                losses, z, mu, logvar = self.compute_loss(batch, beta)

            for k, v in losses.items():
                total_losses[k] += v.item()
            num_batches += 1
            try:
                unit_kl_chunks.append(kl_per_unit(mu.detach().float(), logvar.detach().float()).cpu())
            except Exception:
                pass

        avg_losses = {k: v / num_batches for k, v in total_losses.items()}

        self.history['val_loss'].append(avg_losses['total'])
        self.history['val_recon'].append(avg_losses['recon'])
        self.history['val_kl'].append(avg_losses['kl'])
        n_active, frac_active, kl_sum = 0, 0.0, float(avg_losses['kl'])
        if unit_kl_chunks:
            mean_kl_u = torch.cat(unit_kl_chunks, dim=0).mean(dim=0)
            thr = globals().get('ACTIVE_KL_THRESHOLD', 0.01)
            n_active = int((mean_kl_u > thr).sum().item())
            frac_active = float((mean_kl_u > thr).float().mean().item())
            kl_sum = float(mean_kl_u.sum().item())
        self.history['val_n_active'].append(n_active)
        self.history['val_frac_active'].append(frac_active)
        self.history['val_kl_sum_units'].append(kl_sum)
        avg_losses['n_active'] = n_active
        avg_losses['frac_active'] = frac_active

        return avg_losses
    
    def train(self, train_loader, val_loader, num_epochs, 
              patience=20, min_delta=0.005, checkpoint_path='best_model.pt', verbose=True):
        """
        Full training loop with early stopping and detailed progress monitoring.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs
            patience: Early stopping patience
            min_delta: Minimum improvement for early stopping
            checkpoint_path: Path to save best model
            verbose: Show detailed progress with tqdm
        """
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"\n{'='*70}")
        print(f"TRAINING DeepCRAMVAE - {num_epochs} epochs")
        print(f"{'='*70}")
        
        # Use tqdm for epoch progress with detailed description
        epoch_iterator = tqdm(range(num_epochs), desc="Training Progress", 
                             leave=True, position=0) if verbose else range(num_epochs)
        
        for epoch in epoch_iterator:
            # Train
            train_losses, beta = self.train_epoch(train_loader, epoch, num_epochs)
            
            # Validate
            val_losses = self.validate(val_loader, epoch, num_epochs)
            
            # Update tqdm with current losses
            if verbose and hasattr(epoch_iterator, 'set_postfix'):
                epoch_iterator.set_postfix({
                    'TrnLoss': f"{train_losses['total']:.3f}",
                    'ValLoss': f"{val_losses['total']:.3f}",
                    'TrnRecon': f"{train_losses['recon']:.3f}",
                    'ValRecon': f"{val_losses['recon']:.3f}",
                    'KL': f"{train_losses['kl']:.4f}",
                    'β': f"{beta:.5f}",
                    'Best': f"{best_val_loss:.3f}",
                    'Patience': f"{patience_counter}/{patience}"
                })
            
            # Print detailed progress every 5 epochs or first epoch
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"\n{'─'*70}")
                print(f" Epoch {epoch+1}/{num_epochs} Summary")
                print(f"{'─'*70}")
                print(f" β-annealing: {beta:.6f}")
                print(f" Learning rate: {self.optimizer.param_groups[0]['lr']:.6f}")
                print(f"\n Training Losses:")
                print(f" • Total: {train_losses['total']:.4f}")
                print(f" • Recon: {train_losses['recon']:.4f}")
                print(f" • KL: {train_losses['kl']:.4f}")
                print(f" • Contrastive: {train_losses['contrastive']:.4f}")
                print(f" • Property: {train_losses['property']:.4f}")
                print(f"\n Validation Losses:")
                print(f" • Total: {val_losses['total']:.4f}")
                print(f" • Recon: {val_losses['recon']:.4f}")
                print(f" • KL: {val_losses['kl']:.4f}")
                print(f" • Active units: {val_losses.get('n_active', 0)} ({val_losses.get('frac_active', 0):.1%})")
            
            # Early stopping
            if val_losses['total'] < best_val_loss - min_delta:
                best_val_loss = val_losses['total']
                patience_counter = 0
                
                # Save checkpoint: weights, optimizer, vocab, configs
                save_full_checkpoint(
                    checkpoint_path,
                    self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    val_loss=best_val_loss,
                    history=self.history,
                )
                
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n{'='*70}")
                    print(f" Early stopping at epoch {epoch+1}")
                    print(f" No improvement for {patience} epochs")
                    print(f"{'='*70}")
                    break
        
        print(f"\n{'='*70}")
        print(f"{'='*70}")
        print(f" Total epochs: {epoch+1}/{num_epochs}")
        print(f" Best val loss: {best_val_loss:.4f}")
        print(f" Final train loss: {train_losses['total']:.4f}")
        print(f" Checkpoint saved: {checkpoint_path}")
        print(f"{'='*70}")
        
        return self.history
