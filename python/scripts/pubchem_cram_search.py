"""
PubChem CRAM-like Molecule Search and Download
===============================================

This script searches PubChem for molecules matching TRUE CRAM characteristics:
- Carboxyl-rich (multiple -COOH groups, minimum 2)
- Aliphatic (non-aromatic carbons, >60% sp3 hybridization)
- Alicyclic (non-aromatic cyclic structures - REQUIRED)

TRUE CRAM = Carboxyl-Rich Alicyclic Molecules
Must have BOTH alicyclic rings AND aliphatic character.

Author: Generated for CRAM Fragment Analysis
Date: December 2025
"""

import requests
from urllib.parse import quote
import time
import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, AllChem
from tqdm import tqdm
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PubChemCRAMSearch:
    """Search PubChem for CRAM-like molecules"""
    
    def __init__(self, output_dir: str = "./cram_data", delay: float = 0.2):
        """
        Initialize PubChem CRAM search
        
        Args:
            output_dir: Directory to save results
            delay: Delay between API requests (seconds) to respect rate limits
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.delay = delay
        self.base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
        
    def build_cram_queries(self) -> List[Dict[str, str]]:
        """
        Build comprehensive search queries for TRUE CRAM molecules.
        Focus on ALICYCLIC + ALIPHATIC + CARBOXYL-RICH structures.
        
        Returns:
            List of query dictionaries with descriptions
        """
        queries = [
            # ═══════════════════════════════════════════════════════════════
            # ALICYCLIC DICARBOXYLIC ACIDS (Core CRAM structures)
            # ═══════════════════════════════════════════════════════════════
            {
                "name": "cyclohexane_dicarboxylic",
                "smarts": "O=C(O)C1CCCCC1C(=O)O",
                "description": "Cyclohexane-1,2-dicarboxylic acids"
            },
            {
                "name": "cyclohexane_1_4_dicarboxylic",
                "smarts": "O=C(O)C1CCC(C(=O)O)CC1",
                "description": "Cyclohexane-1,4-dicarboxylic acids"
            },
            {
                "name": "cyclopentane_dicarboxylic",
                "smarts": "O=C(O)C1CCCC1C(=O)O",
                "description": "Cyclopentane dicarboxylic acids"
            },
            {
                "name": "cycloheptane_dicarboxylic",
                "smarts": "O=C(O)C1CCCCCC1C(=O)O",
                "description": "Cycloheptane dicarboxylic acids"
            },
            
            # ═══════════════════════════════════════════════════════════════
            # ALICYCLIC TRICARBOXYLIC ACIDS
            # ═══════════════════════════════════════════════════════════════
            {
                "name": "cyclohexane_tricarboxylic",
                "smarts": "O=C(O)C1CC(C(=O)O)CCC1C(=O)O",
                "description": "Cyclohexane tricarboxylic acids"
            },
            {
                "name": "cyclopentane_tricarboxylic",
                "smarts": "O=C(O)C1CC(C(=O)O)CC1C(=O)O",
                "description": "Cyclopentane tricarboxylic acids"
            },
            
            # ═══════════════════════════════════════════════════════════════
            # BICYCLIC AND POLYCYCLIC ALIPHATIC CARBOXYLIC ACIDS
            # ═══════════════════════════════════════════════════════════════
            {
                "name": "decalin_carboxylic",
                "smarts": "O=C(O)C1CCC2CCCCC2C1",
                "description": "Decalin carboxylic acids (bicyclic)"
            },
            {
                "name": "norbornane_carboxylic",
                "smarts": "O=C(O)C1CC2CCC1C2",
                "description": "Norbornane carboxylic acids (bridged bicyclic)"
            },
            {
                "name": "adamantane_carboxylic",
                "smarts": "O=C(O)C12CC3CC(CC(C3)C1)C2",
                "description": "Adamantane carboxylic acids (tricyclic)"
            },
            {
                "name": "bicyclohexyl_carboxylic",
                "smarts": "O=C(O)C1CCCCC1C1CCCCC1",
                "description": "Bicyclohexyl carboxylic acids"
            },
            
            # ═══════════════════════════════════════════════════════════════
            # HYDROXYLATED ALICYCLIC CARBOXYLIC ACIDS
            # ═══════════════════════════════════════════════════════════════
            {
                "name": "hydroxy_cyclohexane_carboxylic",
                "smarts": "OC1CCC(C(=O)O)CC1",
                "description": "Hydroxycyclohexane carboxylic acids"
            },
            {
                "name": "dihydroxy_cyclohexane_dicarboxylic",
                "smarts": "OC1CC(O)C(C(=O)O)CC1C(=O)O",
                "description": "Dihydroxycyclohexane dicarboxylic acids"
            },
            {
                "name": "quinic_acid_like",
                "smarts": "OC1CC(O)(C(=O)O)CC(O)C1O",
                "description": "Quinic acid-like structures"
            },
            {
                "name": "shikimic_acid_like",
                "smarts": "O=C(O)C1CC(O)C(O)C(O)C1",
                "description": "Shikimic acid-like alicyclic structures"
            },
            
            # ═══════════════════════════════════════════════════════════════
            # TERPENOID-DERIVED ALICYCLIC ACIDS (Natural CRAM sources)
            # ═══════════════════════════════════════════════════════════════
            {
                "name": "menthane_carboxylic",
                "smarts": "CC(C)C1CCC(C)CC1C(=O)O",
                "description": "Menthane carboxylic acids (monoterpenoid)"
            },
            {
                "name": "pinane_carboxylic",
                "smarts": "CC1(C)C2CCC1(C)C(C(=O)O)C2",
                "description": "Pinane carboxylic acids"
            },
            {
                "name": "carane_carboxylic",
                "smarts": "CC1(C)CC2C(C(=O)O)CC1C2",
                "description": "Carane carboxylic acids"
            },
            {
                "name": "camphane_carboxylic",
                "smarts": "CC1(C)C2CCC1(C(=O)O)CC2",
                "description": "Camphane carboxylic acids"
            },
            
            # ═══════════════════════════════════════════════════════════════
            # STEROID-DERIVED ALICYCLIC ACIDS
            # ═══════════════════════════════════════════════════════════════
            {
                "name": "cholic_acid_core",
                "smarts": "C1CCC2C(C1)CCC1C2CCC2C1CCC(C(=O)O)C2",
                "description": "Bile acid-like structures"
            },
            
            # ═══════════════════════════════════════════════════════════════
            # BRANCHED ALICYCLIC CARBOXYLIC ACIDS
            # ═══════════════════════════════════════════════════════════════
            {
                "name": "methylcyclohexane_carboxylic",
                "smarts": "CC1CCC(C(=O)O)CC1",
                "description": "Methylcyclohexane carboxylic acids"
            },
            {
                "name": "dimethylcyclohexane_carboxylic",
                "smarts": "CC1CC(C)C(C(=O)O)CC1",
                "description": "Dimethylcyclohexane carboxylic acids"
            },
            {
                "name": "isopropylcyclohexane_carboxylic",
                "smarts": "CC(C)C1CCC(C(=O)O)CC1",
                "description": "Isopropylcyclohexane carboxylic acids"
            },
            
            # ═══════════════════════════════════════════════════════════════
            # SPIRO AND FUSED ALICYCLIC SYSTEMS
            # ═══════════════════════════════════════════════════════════════
            {
                "name": "spiro_dicarboxylic",
                "smarts": "O=C(O)C1CCC2(CCCC2)CC1C(=O)O",
                "description": "Spiro-fused dicarboxylic acids"
            },
            {
                "name": "perhydronaphthalene_carboxylic",
                "smarts": "O=C(O)C1CCC2CCCCC2C1",
                "description": "Perhydronaphthalene (decalin) carboxylic acids"
            },
            
            # ═══════════════════════════════════════════════════════════════
            # CYCLOALKANE CHAINS WITH MULTIPLE COOH
            # ═══════════════════════════════════════════════════════════════
            {
                "name": "cycloalkyl_chain_dicarboxylic",
                "smarts": "O=C(O)CCC1CCCCC1CCC(=O)O",
                "description": "Cycloalkyl-linked dicarboxylic acids"
            },
            {
                "name": "bicyclic_chain_tricarboxylic",
                "smarts": "O=C(O)CC1CCC(CC(=O)O)CC1CC(=O)O",
                "description": "Bicyclic tricarboxylic chain acids"
            },
            
            # ═══════════════════════════════════════════════════════════════
            # ALIPHATIC DICARBOXYLIC ACIDS (for reference/comparison)
            # ═══════════════════════════════════════════════════════════════
            {
                "name": "aliphatic_dicarboxylic_acids",
                "smarts": "OC(=O)CCCCC(=O)O",
                "description": "Aliphatic dicarboxylic acids (adipic-like)"
            },
            {
                "name": "medium_chain_dicarboxylic",
                "smarts": "OC(=O)CCCCCCCC(=O)O",
                "description": "Medium-chain dicarboxylic acids (C8-C12)"
            },
            
            # ═══════════════════════════════════════════════════════════════
            # MARINE DOM / ENVIRONMENTAL CRAM PATTERNS
            # ═══════════════════════════════════════════════════════════════
            {
                "name": "cyclic_terpenoid_acid",
                "smarts": "CC1(C)CCCC(C)(C1)C(=O)O",
                "description": "Cyclic terpenoid acids (marine-like)"
            },
            {
                "name": "fused_ring_polycarboxylic",
                "smarts": "O=C(O)C1CC2C(C(=O)O)CCC2CC1C(=O)O",
                "description": "Fused ring polycarboxylic acids"
            }
        ]
        
        return queries
    
    def search_by_smarts(self, smarts: str, max_results: int = 10000) -> List[int]:
        """
        Search PubChem by SMARTS pattern using substructure search
        
        Args:
            smarts: SMARTS pattern to search
            max_results: Maximum number of results
            
        Returns:
            List of PubChem CIDs
        """
        try:
            # Convert SMARTS to SMILES for simpler patterns or use substructure search
            # Use POST request for better reliability
            url = f"{self.base_url}/compound/fastsubstructure/smarts/cids/JSON"
            
            logger.info(f"Searching PubChem with SMARTS: {smarts}")
            
            # Use POST instead of GET
            headers = {'Content-Type': 'application/x-www-form-urlencoded'}
            data = f"smarts={quote(smarts)}"
            
            response = requests.post(url, data=data, headers=headers, timeout=60)
            time.sleep(self.delay)
            
            if response.status_code == 200:
                result = response.json()
                cids = result.get("IdentifierList", {}).get("CID", [])
                # Limit results
                cids = cids[:max_results] if len(cids) > max_results else cids
                logger.info(f"Found {len(cids)} compounds")
                return cids
            elif response.status_code == 400:
                # Try alternative: convert simple SMARTS to SMILES
                logger.info(f"SMARTS search not supported, trying SMILES conversion...")
                return self._search_by_smiles_alternative(smarts, max_results)
            else:
                logger.warning(f"Search failed with status {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Error in SMARTS search: {e}")
            return []
    
    def _search_by_smiles_alternative(self, pattern: str, max_results: int) -> List[int]:
        """Fallback search using SMILES instead of SMARTS"""
        try:
            # Try to convert SMARTS to SMILES for simple patterns
            mol = Chem.MolFromSmarts(pattern)
            if mol:
                smiles = Chem.MolToSmiles(mol)
                url = f"{self.base_url}/compound/fastsubstructure/smiles/{quote(smiles)}/cids/JSON"
                response = requests.get(url, timeout=60)
                time.sleep(self.delay)
                
                if response.status_code == 200:
                    result = response.json()
                    cids = result.get("IdentifierList", {}).get("CID", [])
                    cids = cids[:max_results] if len(cids) > max_results else cids
                    logger.info(f"Found {len(cids)} compounds using SMILES fallback")
                    return cids
        except Exception as e:
            logger.debug(f"SMILES fallback failed: {e}")
        
        return []
    
    def search_by_name_keywords(self) -> List[int]:
        """
        Search by chemical name keywords related to TRUE CRAM
        Focus on ALICYCLIC + CARBOXYLIC structures
        
        Returns:
            List of PubChem CIDs
        """
        keywords = [
            # ═══════════════════════════════════════════════════════════════
            # ALICYCLIC CARBOXYLIC ACIDS (Core CRAM)
            # ═══════════════════════════════════════════════════════════════
            "cyclohexane carboxylic acid",
            "cyclohexane dicarboxylic acid",
            "cyclohexane tricarboxylic acid",
            "cyclopentane carboxylic acid",
            "cyclopentane dicarboxylic acid",
            "cycloheptane carboxylic acid",
            "cyclobutane carboxylic acid",
            "cyclopropane carboxylic acid",
            
            # ═══════════════════════════════════════════════════════════════
            # BICYCLIC/POLYCYCLIC ALIPHATIC ACIDS
            # ═══════════════════════════════════════════════════════════════
            "decalin carboxylic acid",
            "norbornane carboxylic acid",
            "norbornene carboxylic acid",
            "adamantane carboxylic acid",
            "bicyclohexyl carboxylic acid",
            "pinane carboxylic acid",
            "camphoric acid",
            "camphanic acid",
            
            # ═══════════════════════════════════════════════════════════════
            # HYDROXYLATED ALICYCLIC ACIDS
            # ═══════════════════════════════════════════════════════════════
            "quinic acid",
            "shikimic acid",
            "hydroxycyclohexane carboxylic acid",
            "dihydroxycyclohexane carboxylic acid",
            
            # ═══════════════════════════════════════════════════════════════
            # TERPENOID-DERIVED ACIDS
            # ═══════════════════════════════════════════════════════════════
            "menthane carboxylic acid",
            "limonene carboxylic acid",
            "pinene carboxylic acid",
            "carane carboxylic acid",
            "thujane carboxylic acid",
            "bornane carboxylic acid",
            "fenchane carboxylic acid",
            "terpene carboxylic acid",
            "isoprenoid acid",
            
            # ═══════════════════════════════════════════════════════════════
            # BILE ACID-LIKE STRUCTURES
            # ═══════════════════════════════════════════════════════════════
            "cholic acid",
            "deoxycholic acid",
            "lithocholic acid",
            "chenodeoxycholic acid",
            "ursodeoxycholic acid",
            "bile acid",
            
            # ═══════════════════════════════════════════════════════════════
            # STEROID CARBOXYLIC ACIDS
            # ═══════════════════════════════════════════════════════════════
            "cholanic acid",
            "etianic acid",
            "androstane carboxylic acid",
            
            # ═══════════════════════════════════════════════════════════════
            # ALIPHATIC DICARBOXYLIC ACIDS (reference)
            # ═══════════════════════════════════════════════════════════════
            "dicarboxylic acid",
            "tricarboxylic acid",
            "adipic acid",
            "succinic acid",
            "glutaric acid",
            "pimelic acid",
            "suberic acid",
            "azelaic acid",
            "sebacic acid",
            "dodecanedioic acid",
            
            # ═══════════════════════════════════════════════════════════════
            # MARINE/ENVIRONMENTAL CRAM-LIKE
            # ═══════════════════════════════════════════════════════════════
            "humic acid",
            "fulvic acid",
            "resin acid",
            "abietic acid",
            "pimaric acid",
            "dehydroabietic acid",
            
            # ═══════════════════════════════════════════════════════════════
            # NATURAL PRODUCT CRAM-LIKE
            # ═══════════════════════════════════════════════════════════════
            "gibberellic acid",
            "kaurenic acid",
            "labdane acid",
            "clerodane acid",
            
            # ═══════════════════════════════════════════════════════════════
            # SPECIFIC NAMED CRAM MOLECULES
            # ═══════════════════════════════════════════════════════════════
            "citric acid",
            "isocitric acid",
            "aconitic acid",
            "malic acid",
            "tartaric acid",
            "oxalic acid",
            "malonic acid",
            "fumaric acid",
            "maleic acid",
            "itaconic acid"
        ]
        
        all_cids = set()
        
        for keyword in keywords:
            try:
                logger.info(f"Searching for keyword: {keyword}")
                # URL encode the keyword properly
                encoded_keyword = quote(keyword)
                url = f"{self.base_url}/compound/name/{encoded_keyword}/cids/JSON"
                response = requests.get(url, timeout=30)
                time.sleep(self.delay)
                
                if response.status_code == 200:
                    data = response.json()
                    cids = data.get("IdentifierList", {}).get("CID", [])
                    all_cids.update(cids)
                    logger.info(f"Found {len(cids)} compounds for '{keyword}'")
                    
            except Exception as e:
                logger.error(f"Error searching keyword '{keyword}': {e}")
                continue
        
        return list(all_cids)
    
    def get_compound_properties(self, cids: List[int], batch_size: int = 100) -> pd.DataFrame:
        """
        Retrieve compound properties from PubChem
        
        Args:
            cids: List of PubChem CIDs
            batch_size: Number of CIDs to request at once
            
        Returns:
            DataFrame with compound properties
        """
        all_data = []
        
        # Process in batches
        for i in tqdm(range(0, len(cids), batch_size), desc="Fetching properties"):
            batch_cids = cids[i:i + batch_size]
            cid_string = ",".join(map(str, batch_cids))
            
            try:
                url = f"{self.base_url}/compound/cid/{cid_string}/property/CanonicalSMILES,IsomericSMILES,MolecularFormula,MolecularWeight,HBondDonorCount,HBondAcceptorCount,XLogP/JSON"
                response = requests.get(url, timeout=30)
                time.sleep(self.delay)
                
                if response.status_code == 200:
                    data = response.json()
                    properties = data.get("PropertyTable", {}).get("Properties", [])
                    
                    # Ensure we have SMILES - PubChem may return different SMILES fields
                    for prop in properties:
                        if 'CanonicalSMILES' not in prop and 'IsomericSMILES' in prop:
                            prop['CanonicalSMILES'] = prop['IsomericSMILES']
                        elif 'CanonicalSMILES' not in prop and 'ConnectivitySMILES' in prop:
                            prop['CanonicalSMILES'] = prop['ConnectivitySMILES']
                    
                    all_data.extend(properties)
                else:
                    logger.warning(f"Failed to fetch batch starting at CID {batch_cids[0]}")
                    
            except Exception as e:
                logger.error(f"Error fetching properties for batch: {e}")
                continue
        
        return pd.DataFrame(all_data)
    
    def filter_cram_molecules(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter molecules to keep only TRUE CRAM structures.
        
        TRUE CRAM Requirements:
        1. At least 2 carboxyl groups (COOH or COO-)
        2. At least one alicyclic ring (non-aromatic cyclic)
        3. >60% aliphatic carbons (non-aromatic)
        
        Args:
            df: DataFrame with SMILES and properties
            
        Returns:
            Filtered DataFrame with TRUE CRAM molecules only
        """
        logger.info("Filtering for TRUE CRAM molecules...")
        logger.info("  Requirements: 2+ COOH AND alicyclic ring AND >60% aliphatic")
        
        cram_molecules = []
        filter_stats = {
            'total': 0,
            'invalid': 0,
            'insufficient_cooh': 0,
            'no_alicyclic_ring': 0,
            'low_aliphatic': 0,
            'has_aromatic': 0,
            'passed_true_cram': 0,
            'passed_partial_cram': 0  # Has ring + COOH but may have some aromatic
        }
        
        # Pre-compile SMARTS patterns
        cooh_pattern = Chem.MolFromSmarts('[CX3](=O)[OX2H1,OX1-]')
        alicyclic_5 = Chem.MolFromSmarts('[R1;!a]1~[R1;!a]~[R1;!a]~[R1;!a]~[R1;!a]1')
        alicyclic_6 = Chem.MolFromSmarts('[R1;!a]1~[R1;!a]~[R1;!a]~[R1;!a]~[R1;!a]~[R1;!a]1')
        alicyclic_7 = Chem.MolFromSmarts('[R1;!a]1~[R1;!a]~[R1;!a]~[R1;!a]~[R1;!a]~[R1;!a]~[R1;!a]1')
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Filtering for TRUE CRAM"):
            filter_stats['total'] += 1
            smiles = row.get('CanonicalSMILES')
            if not smiles:
                filter_stats['invalid'] += 1
                continue
            
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    filter_stats['invalid'] += 1
                    continue
                
                # ═══════════════════════════════════════════════════════════
                # 1. Count CARBOXYL groups (must be ≥2 for TRUE CRAM)
                # ═══════════════════════════════════════════════════════════
                carboxyl_count = len(mol.GetSubstructMatches(cooh_pattern)) if cooh_pattern else 0
                
                # ═══════════════════════════════════════════════════════════
                # 2. Check for ALICYCLIC rings (REQUIRED for TRUE CRAM)
                # ═══════════════════════════════════════════════════════════
                has_alicyclic = (
                    (alicyclic_5 and len(mol.GetSubstructMatches(alicyclic_5)) > 0) or
                    (alicyclic_6 and len(mol.GetSubstructMatches(alicyclic_6)) > 0) or
                    (alicyclic_7 and len(mol.GetSubstructMatches(alicyclic_7)) > 0)
                )
                
                # Also check for any non-aromatic ring
                ring_info = mol.GetRingInfo()
                num_rings = ring_info.NumRings()
                has_any_ring = num_rings > 0
                
                # Check if at least one ring is fully non-aromatic
                if has_any_ring and not has_alicyclic:
                    for ring in ring_info.AtomRings():
                        is_alicyclic_ring = all(
                            not mol.GetAtomWithIdx(i).GetIsAromatic() 
                            for i in ring
                        )
                        if is_alicyclic_ring:
                            has_alicyclic = True
                            break
                
                # ═══════════════════════════════════════════════════════════
                # 3. Calculate ALIPHATIC percentage (>60% for TRUE CRAM)
                # ═══════════════════════════════════════════════════════════
                carbon_atoms = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 6]
                c_count = len(carbon_atoms)
                
                if c_count == 0:
                    filter_stats['invalid'] += 1
                    continue
                
                aromatic_c = sum(1 for a in carbon_atoms if a.GetIsAromatic())
                aliphatic_pct = (c_count - aromatic_c) / c_count
                
                # ═══════════════════════════════════════════════════════════
                # 4. Count aromatic rings (should be 0 for pure CRAM)
                # ═══════════════════════════════════════════════════════════
                aromatic_ring_count = Descriptors.NumAromaticRings(mol)
                
                # ═══════════════════════════════════════════════════════════
                # TRUE CRAM VALIDATION
                # ═══════════════════════════════════════════════════════════
                
                # Track failure reasons
                if carboxyl_count < 2:
                    filter_stats['insufficient_cooh'] += 1
                    # Still collect molecules with 1 COOH + alicyclic for dataset diversity
                    if carboxyl_count >= 1 and has_alicyclic and aliphatic_pct >= 0.6:
                        pass  # Will be added as partial CRAM
                    else:
                        continue
                
                if not has_alicyclic:
                    filter_stats['no_alicyclic_ring'] += 1
                    continue
                
                if aliphatic_pct < 0.6:
                    filter_stats['low_aliphatic'] += 1
                    continue
                
                # TRUE CRAM: 2+ COOH + alicyclic + >60% aliphatic
                is_true_cram = (
                    carboxyl_count >= 2 and 
                    has_alicyclic and 
                    aliphatic_pct >= 0.6
                )
                
                if is_true_cram:
                    filter_stats['passed_true_cram'] += 1
                else:
                    filter_stats['passed_partial_cram'] += 1
                
                # Calculate additional properties
                o_count = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 8)
                h_count = sum(a.GetTotalNumHs() for a in mol.GetAtoms())
                
                row_data = row.to_dict()
                row_data.update({
                    'carboxyl_count': carboxyl_count,
                    'has_alicyclic': has_alicyclic,
                    'aliphatic_pct': round(aliphatic_pct, 3),
                    'is_true_cram': is_true_cram,
                    'ring_count': num_rings,
                    'aromatic_ring_count': aromatic_ring_count,
                    'O_C_ratio': round(o_count / c_count, 3) if c_count > 0 else 0,
                    'H_C_ratio': round(h_count / c_count, 3) if c_count > 0 else 0,
                    'num_carbons': c_count,
                    'num_oxygens': o_count
                })
                cram_molecules.append(row_data)
                    
            except Exception as e:
                logger.debug(f"Error processing molecule: {e}")
                filter_stats['invalid'] += 1
                continue
        
        # Log statistics
        logger.info(f"\n{'='*70}")
        logger.info("TRUE CRAM Filter Statistics:")
        logger.info(f"{'='*70}")
        logger.info(f"  Total processed:        {filter_stats['total']:,}")
        logger.info(f"  Invalid/failed:         {filter_stats['invalid']:,}")
        logger.info(f"  Rejected - <2 COOH:     {filter_stats['insufficient_cooh']:,}")
        logger.info(f"  Rejected - no alicyclic:{filter_stats['no_alicyclic_ring']:,}")
        logger.info(f"  Rejected - <60% aliph:  {filter_stats['low_aliphatic']:,}")
        logger.info(f"{'='*70}")
        logger.info(f"  ✅ TRUE CRAM (all criteria): {filter_stats['passed_true_cram']:,}")
        logger.info(f"  ⚠️  Partial CRAM (1 COOH):   {filter_stats['passed_partial_cram']:,}")
        logger.info(f"  📊 Total passed:             {len(cram_molecules):,}")
        logger.info(f"{'='*70}")
        
        return pd.DataFrame(cram_molecules)
    
    def _count_carboxyl_groups(self, mol) -> int:
        """Count carboxyl groups in molecule"""
        carboxyl_pattern = Chem.MolFromSmarts('[CX3](=O)[OX2H1]')
        if carboxyl_pattern:
            matches = mol.GetSubstructMatches(carboxyl_pattern)
            return len(matches)
        return 0
    
    def _calculate_aliphatic_ratio(self, mol) -> float:
        """Calculate ratio of aliphatic (sp3) to total carbons"""
        carbon_atoms = [atom for atom in mol.GetAtoms() if atom.GetSymbol() == 'C']
        if not carbon_atoms:
            # If no carbons, check if it's still a valid organic molecule with heteroatoms
            return 0.0
        
        # Count sp3 hybridized carbons (aliphatic)
        sp3_carbons = sum(1 for atom in carbon_atoms 
                         if atom.GetHybridization() == Chem.HybridizationType.SP3)
        
        # Also count sp2 carbons that are not aromatic (like in C=O)
        sp2_nonaromatic = sum(1 for atom in carbon_atoms 
                             if atom.GetHybridization() == Chem.HybridizationType.SP2 
                             and not atom.GetIsAromatic())
        
        # Calculate ratio - include non-aromatic sp2 as partially aliphatic
        aliphatic_score = sp3_carbons + (0.5 * sp2_nonaromatic)
        
        return aliphatic_score / len(carbon_atoms)
    
    def search_chembl_alicyclic_acids(self) -> List[int]:
        """
        Search ChEMBL for alicyclic carboxylic acids via PubChem integration.
        Uses PubChem's ChEMBL cross-references.
        
        Returns:
            List of PubChem CIDs from ChEMBL sources
        """
        logger.info("Searching ChEMBL-derived compounds via PubChem...")
        
        # ChEMBL compound classes related to CRAM
        chembl_sources = [
            "Carboxylic Acids, Cyclic",
            "Terpenes",
            "Triterpenes", 
            "Steroids",
            "Bile Acids and Salts",
            "Cyclopentanes",
            "Cyclohexanes"
        ]
        
        all_cids = set()
        
        for source in chembl_sources:
            try:
                encoded = quote(source)
                url = f"{self.base_url}/compound/name/{encoded}/cids/JSON"
                response = requests.get(url, timeout=30)
                time.sleep(self.delay)
                
                if response.status_code == 200:
                    data = response.json()
                    cids = data.get("IdentifierList", {}).get("CID", [])
                    all_cids.update(cids[:1000])  # Limit per class
                    logger.info(f"  {source}: {len(cids)} compounds")
                    
            except Exception as e:
                logger.debug(f"Error with {source}: {e}")
                continue
        
        return list(all_cids)
    
    def search_natural_product_databases(self) -> List[int]:
        """
        Search for natural product-derived CRAM molecules.
        Many CRAM-like structures come from terpenoid degradation.
        
        Returns:
            List of PubChem CIDs
        """
        logger.info("Searching for natural product-derived CRAM...")
        
        # Natural product keywords that often yield CRAM-like structures
        np_keywords = [
            # Terpenoid degradation products
            "terpenoid acid",
            "sesquiterpenoid acid",
            "diterpenoid acid",
            "triterpenoid acid",
            "norterpenoid",
            
            # Carotenoid degradation
            "apocarotenoid acid",
            "ionone carboxylic acid",
            
            # Steroid degradation
            "secosteroid acid",
            "androstane acid",
            "pregnane acid",
            
            # Isoprene-derived
            "isoprenoid acid",
            "prenyl acid",
            "geranyl acid",
            
            # Cyclic natural products
            "cycloartenol",
            "lanosterol acid",
            "squalene epoxide",
            
            # Marine natural products (often CRAM-like)
            "marine terpenoid",
            "sponge acid",
            "algal acid",
            
            # Resin-derived (common CRAM sources)
            "diterpenoid resin acid",
            "abietane",
            "pimarane",
            "labdane acid",
            "kaurane acid",
            
            # Fungal metabolites
            "fungal acid",
            "mycotoxin acid",
            "ergosterol acid"
        ]
        
        all_cids = set()
        
        for keyword in np_keywords:
            try:
                encoded = quote(keyword)
                url = f"{self.base_url}/compound/name/{encoded}/cids/JSON"
                response = requests.get(url, timeout=30)
                time.sleep(self.delay)
                
                if response.status_code == 200:
                    data = response.json()
                    cids = data.get("IdentifierList", {}).get("CID", [])
                    all_cids.update(cids)
                    if cids:
                        logger.info(f"  {keyword}: {len(cids)} compounds")
                        
            except Exception as e:
                continue
        
        logger.info(f"  Total from natural products: {len(all_cids)}")
        return list(all_cids)
    
    def search_environmental_dom_molecules(self) -> List[int]:
        """
        Search for environmental DOM (Dissolved Organic Matter) molecules.
        CRAM is a major component of marine DOM.
        
        Returns:
            List of PubChem CIDs
        """
        logger.info("Searching for environmental/DOM-related molecules...")
        
        # Environmental/DOM keywords
        dom_keywords = [
            # Humic substances
            "humic acid",
            "fulvic acid", 
            "humin",
            
            # Lignin degradation products
            "lignin acid",
            "vanillic acid",
            "syringic acid",
            "ferulic acid",
            
            # Soil organic matter
            "soil organic acid",
            "terrestrial organic acid",
            
            # Marine organic matter
            "marine organic acid",
            "phytoplankton acid",
            "zooplankton acid",
            
            # Degradation products
            "biodegradation acid",
            "photodegradation acid",
            
            # Petroleum-related (some CRAM from ancient sources)
            "naphthenic acid",
            "petroleum acid",
            "cycloalkane acid",
            
            # Biogenic acids
            "biogenic acid",
            "microbial acid"
        ]
        
        all_cids = set()
        
        for keyword in dom_keywords:
            try:
                encoded = quote(keyword)
                url = f"{self.base_url}/compound/name/{encoded}/cids/JSON"
                response = requests.get(url, timeout=30)
                time.sleep(self.delay)
                
                if response.status_code == 200:
                    data = response.json()
                    cids = data.get("IdentifierList", {}).get("CID", [])
                    all_cids.update(cids)
                    if cids:
                        logger.info(f"  {keyword}: {len(cids)} compounds")
                        
            except Exception as e:
                continue
        
        logger.info(f"  Total from environmental sources: {len(all_cids)}")
        return list(all_cids)
    
    def run_comprehensive_search(self, max_results_per_query: int = 5000, include_extended_sources: bool = True):
        """
        Run comprehensive TRUE CRAM molecule search from multiple databases.
        
        Args:
            max_results_per_query: Maximum results per individual query
            include_extended_sources: Include ChEMBL, natural products, and environmental sources
        """
        logger.info("=" * 70)
        logger.info("Starting Comprehensive TRUE CRAM Molecule Search")
        logger.info("=" * 70)
        logger.info("Target: Molecules with 2+ COOH + alicyclic ring + >60% aliphatic")
        
        all_cids = set()
        
        # 1. Search by SMARTS patterns (alicyclic carboxylic acids)
        logger.info("\n[1/5] Searching by SMARTS patterns (alicyclic structures)...")
        queries = self.build_cram_queries()
        
        for query in queries:
            logger.info(f"\n  Query: {query['description']}")
            cids = self.search_by_smarts(query['smarts'], max_results_per_query)
            all_cids.update(cids)
            logger.info(f"  Total unique CIDs so far: {len(all_cids)}")
        
        # 2. Search by keywords
        logger.info("\n[2/5] Searching by chemical name keywords...")
        keyword_cids = self.search_by_name_keywords()
        all_cids.update(keyword_cids)
        logger.info(f"Total unique CIDs: {len(all_cids)}")
        
        if include_extended_sources:
            # 3. Search ChEMBL-derived compounds
            logger.info("\n[3/5] Searching ChEMBL-derived compounds...")
            chembl_cids = self.search_chembl_alicyclic_acids()
            all_cids.update(chembl_cids)
            logger.info(f"Total unique CIDs: {len(all_cids)}")
            
            # 4. Search natural product databases
            logger.info("\n[4/5] Searching natural product sources...")
            np_cids = self.search_natural_product_databases()
            all_cids.update(np_cids)
            logger.info(f"Total unique CIDs: {len(all_cids)}")
            
            # 5. Search environmental/DOM molecules
            logger.info("\n[5/5] Searching environmental/DOM molecules...")
            dom_cids = self.search_environmental_dom_molecules()
            all_cids.update(dom_cids)
            logger.info(f"Total unique CIDs: {len(all_cids)}")
        
        # Fetch properties
        logger.info(f"\n📥 Fetching compound properties for {len(all_cids)} CIDs...")
        cid_list = list(all_cids)
        df = self.get_compound_properties(cid_list)
        
        if df.empty:
            logger.error("No compounds retrieved!")
            return None
        
        logger.info(f"Retrieved properties for {len(df)} compounds")
        
        # Save raw data
        raw_file = self.output_dir / "raw_compounds.csv"
        df.to_csv(raw_file, index=False)
        logger.info(f"Raw data saved to {raw_file}")
        
        # Filter for TRUE CRAM molecules
        logger.info("\n🔬 Filtering for TRUE CRAM molecules...")
        cram_df = self.filter_cram_molecules(df)
        
        if len(cram_df) == 0:
            logger.warning("No TRUE CRAM molecules found!")
            return None
        
        # Save results
        output_file = self.output_dir / "cram_molecules.csv"
        cram_df.to_csv(output_file, index=False)
        logger.info(f"\n✅ Saved {len(cram_df)} CRAM molecules to {output_file}")
        
        # Save TRUE CRAM only (2+ COOH)
        if 'is_true_cram' in cram_df.columns:
            true_cram_df = cram_df[cram_df['is_true_cram'] == True]
            true_cram_file = self.output_dir / "true_cram_molecules.csv"
            true_cram_df.to_csv(true_cram_file, index=False)
            logger.info(f"✅ Saved {len(true_cram_df)} TRUE CRAM molecules to {true_cram_file}")
        
        # Generate summary
        self._generate_summary(cram_df)
        
        return cram_df
    
    def _generate_summary(self, df: pd.DataFrame):
        """Generate and display TRUE CRAM summary statistics"""
        logger.info("\n" + "=" * 70)
        logger.info("TRUE CRAM Dataset Summary")
        logger.info("=" * 70)
        
        logger.info(f"\nTotal molecules: {len(df)}")
        
        # TRUE CRAM count
        if 'is_true_cram' in df.columns:
            true_cram_count = df['is_true_cram'].sum()
            logger.info(f"\n🎯 TRUE CRAM (2+ COOH + alicyclic + >60% aliphatic): {true_cram_count} ({100*true_cram_count/len(df):.1f}%)")
        
        if 'carboxyl_count' in df.columns:
            logger.info(f"\n📊 Carboxyl group distribution:")
            logger.info(f"  Mean: {df['carboxyl_count'].mean():.2f}")
            logger.info(f"  Min: {df['carboxyl_count'].min()}")
            logger.info(f"  Max: {df['carboxyl_count'].max()}")
            logger.info(f"  2+ COOH: {(df['carboxyl_count'] >= 2).sum()} ({100*(df['carboxyl_count'] >= 2).mean():.1f}%)")
            
        if 'has_alicyclic' in df.columns:
            alicyclic_count = df['has_alicyclic'].sum()
            logger.info(f"\n🔵 Alicyclic rings: {alicyclic_count} ({100*alicyclic_count/len(df):.1f}%)")
            
        if 'aliphatic_pct' in df.columns:
            logger.info(f"\n🟢 Aliphatic character:")
            logger.info(f"  Mean: {df['aliphatic_pct'].mean():.1%}")
            logger.info(f"  Min: {df['aliphatic_pct'].min():.1%}")
            logger.info(f"  Max: {df['aliphatic_pct'].max():.1%}")
            logger.info(f"  >60% aliphatic: {(df['aliphatic_pct'] >= 0.6).sum()} ({100*(df['aliphatic_pct'] >= 0.6).mean():.1f}%)")
        
        if 'O_C_ratio' in df.columns and 'H_C_ratio' in df.columns:
            logger.info(f"\n⚗️ Elemental ratios (Van Krevelen):")
            logger.info(f"  O/C: {df['O_C_ratio'].mean():.3f} (range {df['O_C_ratio'].min():.3f}-{df['O_C_ratio'].max():.3f})")
            logger.info(f"  H/C: {df['H_C_ratio'].mean():.3f} (range {df['H_C_ratio'].min():.3f}-{df['H_C_ratio'].max():.3f})")
        
        if 'MolecularWeight' in df.columns:
            logger.info(f"\n⚖️ Molecular weight:")
            try:
                mw = pd.to_numeric(df['MolecularWeight'], errors='coerce')
                logger.info(f"  Mean: {mw.mean():.1f}")
                logger.info(f"  Range: {mw.min():.1f} - {mw.max():.1f}")
            except Exception as e:
                logger.warning(f"  Could not calculate: {e}")
        
        if 'ring_count' in df.columns:
            logger.info(f"\n💍 Ring statistics:")
            logger.info(f"  Mean rings: {df['ring_count'].mean():.2f}")
            logger.info(f"  Max rings: {df['ring_count'].max()}")
        
        # Save summary to file
        summary_file = self.output_dir / "cram_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("TRUE CRAM Dataset Summary\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Total molecules: {len(df)}\n")
            if 'is_true_cram' in df.columns:
                f.write(f"TRUE CRAM (all criteria): {df['is_true_cram'].sum()} ({100*df['is_true_cram'].mean():.1f}%)\n\n")
            f.write("TRUE CRAM Requirements:\n")
            f.write("  ✓ 2+ carboxyl groups (COOH or COO-)\n")
            f.write("  ✓ Alicyclic ring (non-aromatic cyclic structure)\n")
            f.write("  ✓ >60% aliphatic carbons\n\n")
            if 'carboxyl_count' in df.columns:
                f.write(f"COOH groups: {df['carboxyl_count'].mean():.2f} average\n")
            if 'has_alicyclic' in df.columns:
                f.write(f"Has alicyclic: {100*df['has_alicyclic'].mean():.1f}%\n")
            if 'aliphatic_pct' in df.columns:
                f.write(f"Aliphatic %: {100*df['aliphatic_pct'].mean():.1f}% average\n")
            if 'O_C_ratio' in df.columns:
                f.write(f"O/C ratio: {df['O_C_ratio'].mean():.3f} average\n")
                f.write(f"H/C ratio: {df['H_C_ratio'].mean():.3f} average\n")
        
        logger.info(f"\n📄 Summary saved to {summary_file}")


def main():
    """Main execution function"""
    print("""
    ╔════════════════════════════════════════════════════════════════════╗
    ║   PubChem TRUE CRAM Molecule Search                                ║
    ║   Carboxyl-Rich Alicyclic Molecules                                ║
    ╠════════════════════════════════════════════════════════════════════╣
    ║   TRUE CRAM Requirements:                                          ║
    ║   ✓ 2+ carboxyl groups (COOH or COO-)                             ║
    ║   ✓ Alicyclic ring (non-aromatic cyclic structure)                ║
    ║   ✓ >60% aliphatic carbons (non-aromatic)                         ║
    ╚════════════════════════════════════════════════════════════════════╝
    """)
    
    # Initialize searcher
    searcher = PubChemCRAMSearch(
        output_dir="./cram_data",
        delay=0.2  # Respect PubChem rate limits
    )
    
    # Run comprehensive search with extended sources
    cram_df = searcher.run_comprehensive_search(
        max_results_per_query=5000,
        include_extended_sources=True  # Include ChEMBL, natural products, environmental
    )
    
    print("\n" + "=" * 70)
    print("Search completed!")
    print("=" * 70)
    print(f"\nResults saved to: {searcher.output_dir}")
    print("\nFiles generated:")
    print("  - cram_molecules.csv:      All CRAM-like molecules (1+ COOH + alicyclic)")
    print("  - true_cram_molecules.csv: TRUE CRAM only (2+ COOH + alicyclic + >60% aliphatic)")
    print("  - raw_compounds.csv:       All downloaded compounds (for debugging)")
    print("  - cram_summary.txt:        Summary statistics")
    print("\nYou can now use this dataset for training your model!")
    print("\n💡 Tip: Use true_cram_molecules.csv for stricter CRAM definition")


if __name__ == "__main__":
    main()
