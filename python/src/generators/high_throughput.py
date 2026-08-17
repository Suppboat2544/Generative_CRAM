"""Notebook HighThroughputCRAMGenerator."""
from __future__ import annotations
from typing import List, Optional, Dict
import numpy as np
from rdkit import Chem

# ══════════════════════════════════════════════════════════════════════════════
# HIGH-THROUGHPUT CRAM GENERATION SYSTEM
# ══════════════════════════════════════════════════════════════════════════════

import pandas as pd
from datetime import datetime
import hashlib
from collections import defaultdict

class HighThroughputCRAMGenerator:
    """
    Generate large-scale CRAM molecule libraries using encoder-based expansion.
    """
    
    def __init__(self, model, train_smiles, device, feature_extractor, 
                 output_dir='/content/drive/MyDrive/CRAM_Generator/results'):
        """
        Initialize the generator.
        
        Args:
            model: Trained DeepCRAMVAE model
            train_smiles: List of training SMILES (candidate pool)
            device: Device for computation
            feature_extractor: DeepCRAMFeatureExtractor instance
            output_dir: Directory to save results (default: Google Drive)
        """
        self.model = model
        self.train_smiles = train_smiles
        self.device = device
        self.feature_extractor = feature_extractor
        self.generated_smiles = set() # Track uniqueness
        self.results = []
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print(" HighThroughputCRAMGenerator initialized")
        print(f" Candidate pool: {len(train_smiles)} molecules")
        print(f" Device: {device}")
        print(f" Output directory: {output_dir}")
    
    def select_diverse_templates(self, num_templates=100):
        """
        Select diverse templates from training data using MaxMin diversity picking.
        
        Args:
            num_templates: Number of templates to select
        
        Returns:
            List of diverse template SMILES
        """
        print(f"\n Selecting {num_templates} diverse templates...")
        
        # Start with CRAM molecules only
        cram_templates = []
        for smi in self.train_smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                num_cooh = len(mol.GetSubstructMatches(Chem.MolFromSmarts('C(=O)O')))
                if num_cooh >= 2:
                    cram_templates.append(smi)
        
        print(f" Found {len(cram_templates)} CRAM molecules in training data")
        
        if len(cram_templates) <= num_templates:
            print(f" Using all {len(cram_templates)} CRAM molecules as templates")
            return cram_templates
        
        # MaxMin diversity picking
        import random
        random.seed(42)
        
        # Start with random molecule
        selected = [random.choice(cram_templates)]
        remaining = set(cram_templates) - set(selected)
        
        # Compute fingerprints
        fps = {}
        for smi in cram_templates:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                fps[smi] = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
        
        # Iteratively pick most diverse
        for _ in tqdm(range(num_templates - 1), desc="MaxMin picking"):
            max_min_sim = -1
            best_candidate = None
            
            for candidate in remaining:
                if candidate not in fps:
                    continue
                
                # Find minimum similarity to selected
                min_sim = min(
                    DataStructs.TanimotoSimilarity(fps[candidate], fps[sel])
                    for sel in selected if sel in fps
                )
                
                if min_sim > max_min_sim:
                    max_min_sim = min_sim
                    best_candidate = candidate
            
            if best_candidate:
                selected.append(best_candidate)
                remaining.remove(best_candidate)
        
        return selected
    
    def expand_from_template(self, template_smiles, num_variants=1000,
                            similarity_threshold=0.4, latent_weight=0.7):
        """
        Generate variants from a single template.
        
        Args:
            template_smiles: Template SMILES
            num_variants: Number of variants to generate
            similarity_threshold: Minimum similarity
            latent_weight: Weight for latent similarity
        
        Returns:
            List of unique CRAM molecules
        """
        # Encode template
        template_mol = Chem.MolFromSmiles(template_smiles)
        if template_mol is None:
            return []
        
        template_graph = smiles_to_graph_deep_cram(template_smiles, self.feature_extractor)
        if template_graph is None:
            return []
        
        # Add self-loops to prevent 0-in-degree node errors
        import dgl
        template_graph = dgl.add_self_loop(template_graph)
        template_graph = template_graph.to(self.device)
        
        with torch.no_grad():
            self.model.eval()
            encoder_output = self.model.encoder(template_graph)
            template_z = encoder_output['z'].cpu().numpy()
        
        if template_z.ndim == 1:
            template_z = template_z.reshape(1, -1)
        
        # Encode candidates in batches
        candidate_encodings = []
        valid_smiles = []
        
        batch_size = 100
        for i in range(0, len(self.train_smiles), batch_size):
            batch = self.train_smiles[i:i+batch_size]
            
            for smi in batch:
                try:
                    g = smiles_to_graph_deep_cram(smi, self.feature_extractor)
                    if g is None:
                        continue
                    
                    # Add self-loops to prevent 0-in-degree node errors
                    g = dgl.add_self_loop(g)
                    g = g.to(self.device)
                    with torch.no_grad():
                        enc_output = self.model.encoder(g)
                        z = enc_output['z'].cpu().numpy()
                    
                    if z.ndim == 2:
                        z = z.squeeze()
                    
                    candidate_encodings.append(z)
                    valid_smiles.append(smi)
                except:
                    continue
        
        if len(candidate_encodings) == 0:
            return []
        
        candidate_encodings = np.vstack(candidate_encodings)
        
        # Compute similarities
        from sklearn.metrics.pairwise import cosine_similarity
        latent_similarities = cosine_similarity(template_z, candidate_encodings)[0]
        
        # Fingerprint similarities
        template_fp = AllChem.GetMorganFingerprintAsBitVect(template_mol, 2, 2048)
        fp_similarities = []
        
        for smi in valid_smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
                sim = DataStructs.TanimotoSimilarity(template_fp, fp)
                fp_similarities.append(sim)
            else:
                fp_similarities.append(0.0)
        
        fp_similarities = np.array(fp_similarities)
        combined_similarities = latent_weight * latent_similarities + (1-latent_weight) * fp_similarities
        
        # Create list of all candidates with their similarities
        candidates = []
        for i, smi in enumerate(valid_smiles):
            if combined_similarities[i] >= similarity_threshold:
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    num_cooh = len(mol.GetSubstructMatches(Chem.MolFromSmarts('C(=O)O')))
                    if num_cooh >= 2:
                        candidates.append({
                            'smiles': smi,
                            'similarity': combined_similarities[i],
                            'num_cooh': num_cooh,
                            'template': template_smiles
                        })
        
        # Sort by similarity (best first)
        candidates.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Filter out duplicates while maintaining order
        results = []
        for candidate in candidates:
            if candidate['smiles'] not in self.generated_smiles:
                results.append(candidate)
                self.generated_smiles.add(candidate['smiles'])
                
                if len(results) >= num_variants:
                    break
        
        return results
    
    def generate_large_library(self, target_size=100000, 
                               variants_per_template=1000,
                               num_templates=100,
                               output_prefix='cram_library',
                               save_interval=10000):
        """
        Generate large CRAM library and save progressively.
        
        Args:
            target_size: Target number of molecules
            variants_per_template: Variants per template
            num_templates: Number of templates to use
            output_prefix: Prefix for output files
            save_interval: Save every N molecules
        
        Returns:
            DataFrame with results
        """
        print("="*80)
        print(f" GENERATING LARGE CRAM LIBRARY")
        print("="*80)
        print(f"Target size: {target_size:,} molecules")
        print(f"Templates: {num_templates}")
        print(f"Per template: {variants_per_template}")
        print(f"Save interval: {save_interval:,}")
        print("="*80)
        
        # Select diverse templates
        templates = self.select_diverse_templates(num_templates)
        
        # Generate from each template
        total_generated = 0
        batch_results = []
        
        start_time = datetime.now()
        
        for i, template in enumerate(tqdm(templates, desc="Processing templates")):
            # Generate variants
            variants = self.expand_from_template(
                template_smiles=template,
                num_variants=variants_per_template,
                similarity_threshold=0.15, # LOWERED: More permissive threshold
                latent_weight=0.7
            )
            
            batch_results.extend(variants)
            total_generated += len(variants)
            
            # Progress update
            if (i + 1) % 10 == 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                rate = total_generated / elapsed if elapsed > 0 else 0
                print(f"\n Progress: {total_generated:,} / {target_size:,} molecules ({rate:.1f} mol/sec)")
            
            # Save intermediate results
            if len(batch_results) >= save_interval:
                self._save_batch(batch_results, output_prefix, total_generated)
                self.results.extend(batch_results)
                batch_results = []
            
            # Check if target reached
            if total_generated >= target_size:
                break
        
        # Save remaining results
        if batch_results:
            self._save_batch(batch_results, output_prefix, total_generated)
            self.results.extend(batch_results)
        
        # Create final DataFrame
        df = self._create_dataframe(self.results)
        
        # Save final file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_file = f"{output_prefix}_FINAL_{len(df)}_molecules_{timestamp}.csv"
        final_path = os.path.join(self.output_dir, final_file)
        df.to_csv(final_path, index=False)
        
        elapsed_time = (datetime.now() - start_time).total_seconds()
        
        print("\n" + "="*80)
        print("="*80)
        print(f"Total generated: {len(df):,} unique molecules")
        print(f"Time elapsed: {elapsed_time/60:.1f} minutes")
        print(f"Generation rate: {len(df)/elapsed_time:.1f} molecules/sec")
        print(f"Output directory: {self.output_dir}")
        print(f"Final file: {final_path}")
        print("="*80)
        
        return df
    
    def _save_batch(self, batch_results, prefix, total_count):
        """Save batch results to CSV."""
        if not batch_results:
            return
        
        df_batch = self._create_dataframe(batch_results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_batch_{total_count}_{timestamp}.csv"
        filepath = os.path.join(self.output_dir, filename)
        df_batch.to_csv(filepath, index=False)
        print(f"\n Saved batch: {filepath} ({len(df_batch):,} molecules)")
    
    def _create_dataframe(self, results):
        """Convert results to DataFrame with computed properties."""
        if not results:
            return pd.DataFrame()
        
        data = []
        for r in results:
            mol = Chem.MolFromSmiles(r['smiles'])
            if mol:
                data.append({
                    'SMILES': r['smiles'],
                    'Similarity': r['similarity'],
                    'COOH_Count': r['num_cooh'],
                    'Template': r['template'],
                    'MolecularWeight': Descriptors.MolWt(mol),
                    'NumAtoms': mol.GetNumAtoms(),
                    'NumHeavyAtoms': mol.GetNumHeavyAtoms(),
                    'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
                    'TPSA': Descriptors.TPSA(mol),
                    'LogP': Descriptors.MolLogP(mol),
                    'HasAlicyclicRing': int(any(
                        not any(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring)
                        for ring in mol.GetRingInfo().AtomRings()
                    ) if mol.GetRingInfo().NumRings() > 0 else False)
                })
        
        return pd.DataFrame(data)
