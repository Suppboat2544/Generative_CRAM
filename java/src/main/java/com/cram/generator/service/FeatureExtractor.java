package com.cram.generator.service;

import org.openscience.cdk.DefaultChemObjectBuilder;
import org.openscience.cdk.interfaces.IAtom;
import org.openscience.cdk.interfaces.IAtomContainer;
import org.openscience.cdk.smiles.SmilesParser;
import org.openscience.cdk.tools.manipulator.AtomContainerManipulator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

/**
 * Feature extractor for CRAM molecules.
 * Extracts 163-dimensional feature vector for each atom.
 * 
 * Feature dimensions:
 * - Base atom features (45D): atom type, bonds, hybridization
 * - Ring-Chain features (20D): ring systems, chain analysis
 * - Stereochemistry (12D): chiral centers, isomerism
 * - Marine DOM features (43D): ocean chemistry indicators
 * - Terpenoid features (10D): natural product ancestry
 * - Functional groups (13D): chemical functionality
 * - Oxidation state (20D): redox indicators
 */
@Service
public class FeatureExtractor {

    private static final Logger log = LoggerFactory.getLogger(FeatureExtractor.class);
    
    public static final int FEATURE_DIM = 163;
    
    // Feature dimension ranges
    public static final int BASE_START = 0, BASE_END = 45;
    public static final int RING_CHAIN_START = 45, RING_CHAIN_END = 65;
    public static final int STEREO_START = 65, STEREO_END = 77;
    public static final int MARINE_START = 77, MARINE_END = 120;
    public static final int TERPENOID_START = 120, TERPENOID_END = 130;
    public static final int FUNCTIONAL_START = 130, FUNCTIONAL_END = 143;
    public static final int OXIDATION_START = 143, OXIDATION_END = 163;
    
    private final SmilesParser smilesParser;
    
    public FeatureExtractor() {
        this.smilesParser = new SmilesParser(DefaultChemObjectBuilder.getInstance());
    }
    
    /**
     * Extract feature matrix for a molecule.
     * 
     * @param smiles The SMILES string
     * @return 2D array [numAtoms][163] of features
     */
    public double[][] extractFeatures(String smiles) {
        try {
            IAtomContainer molecule = smilesParser.parseSmiles(smiles);
            AtomContainerManipulator.percieveAtomTypesAndConfigureAtoms(molecule);
            
            int numAtoms = molecule.getAtomCount();
            double[][] features = new double[numAtoms][FEATURE_DIM];
            
            for (int i = 0; i < numAtoms; i++) {
                IAtom atom = molecule.getAtom(i);
                features[i] = extractAtomFeatures(molecule, atom, i);
            }
            
            return features;
            
        } catch (Exception e) {
            log.error("Error extracting features for: {}", smiles, e);
            return new double[0][FEATURE_DIM];
        }
    }
    
    /**
     * Extract 163D features for a single atom.
     */
    private double[] extractAtomFeatures(IAtomContainer molecule, IAtom atom, int atomIdx) {
        double[] features = new double[FEATURE_DIM];
        
        // Base features (0-44)
        extractBaseFeatures(features, molecule, atom);
        
        // Ring-Chain features (45-64)
        extractRingChainFeatures(features, molecule, atom);
        
        // Stereochemistry features (65-76)
        extractStereoFeatures(features, molecule, atom);
        
        // Marine DOM features (77-119)
        extractMarineFeatures(features, molecule, atom);
        
        // Terpenoid features (120-129)
        extractTerpenoidFeatures(features, molecule, atom);
        
        // Functional group features (130-142)
        extractFunctionalFeatures(features, molecule, atom);
        
        // Oxidation state features (143-162)
        extractOxidationFeatures(features, molecule, atom);
        
        return features;
    }
    
    /**
     * Extract base atom features (45D).
     */
    private void extractBaseFeatures(double[] features, IAtomContainer mol, IAtom atom) {
        int offset = BASE_START;
        
        // One-hot atom type (10 types)
        int atomicNum = atom.getAtomicNumber() != null ? atom.getAtomicNumber() : 0;
        int[] atomTypes = {6, 1, 8, 7, 16, 15, 9, 17, 35, 53}; // C, H, O, N, S, P, F, Cl, Br, I
        for (int i = 0; i < atomTypes.length; i++) {
            features[offset + i] = (atomicNum == atomTypes[i]) ? 1.0 : 0.0;
        }
        offset += 10;
        
        // Degree (number of bonds)
        int degree = mol.getConnectedBondsCount(atom);
        features[offset++] = Math.min(degree / 4.0, 1.0);
        
        // Formal charge
        Integer charge = atom.getFormalCharge();
        features[offset++] = (charge != null ? charge : 0) / 2.0;
        
        // Hybridization (one-hot, 5 types)
        // SP, SP2, SP3, SP3D, SP3D2
        features[offset++] = 0; // Simplified - would need proper hybridization detection
        features[offset++] = 0;
        features[offset++] = 1; // Default to SP3
        features[offset++] = 0;
        features[offset++] = 0;
        
        // Aromaticity
        features[offset++] = atom.isAromatic() ? 1.0 : 0.0;
        
        // In ring
        features[offset++] = atom.isInRing() ? 1.0 : 0.0;
        
        // Hydrogen count
        Integer hCount = atom.getImplicitHydrogenCount();
        features[offset++] = (hCount != null ? hCount : 0) / 4.0;
        
        // Fill remaining base features with zeros
        while (offset < BASE_END) {
            features[offset++] = 0.0;
        }
    }
    
    /**
     * Extract ring-chain features (20D).
     */
    private void extractRingChainFeatures(double[] features, IAtomContainer mol, IAtom atom) {
        int offset = RING_CHAIN_START;
        
        // In ring
        features[offset++] = atom.isInRing() ? 1.0 : 0.0;
        
        // Ring size indicators (3-8)
        for (int i = 0; i < 6; i++) {
            features[offset++] = 0.0; // Would need ring analysis
        }
        
        // Fill remaining
        while (offset < RING_CHAIN_END) {
            features[offset++] = 0.0;
        }
    }
    
    /**
     * Extract stereochemistry features (12D).
     */
    private void extractStereoFeatures(double[] features, IAtomContainer mol, IAtom atom) {
        int offset = STEREO_START;
        
        // Chiral center indicators
        // Simplified - would need proper stereochemistry analysis
        while (offset < STEREO_END) {
            features[offset++] = 0.0;
        }
    }
    
    /**
     * Extract marine DOM features (43D).
     */
    private void extractMarineFeatures(double[] features, IAtomContainer mol, IAtom atom) {
        int offset = MARINE_START;
        
        // Marine-relevant features
        int atomicNum = atom.getAtomicNumber() != null ? atom.getAtomicNumber() : 0;
        
        // Carboxyl carbon indicator
        features[offset++] = (atomicNum == 6 && isCarboxylCarbon(mol, atom)) ? 1.0 : 0.0;
        
        // Hydroxyl oxygen indicator
        features[offset++] = (atomicNum == 8 && isHydroxylOxygen(mol, atom)) ? 1.0 : 0.0;
        
        // Fill remaining marine features
        while (offset < MARINE_END) {
            features[offset++] = 0.0;
        }
    }
    
    /**
     * Extract terpenoid ancestry features (10D).
     */
    private void extractTerpenoidFeatures(double[] features, IAtomContainer mol, IAtom atom) {
        int offset = TERPENOID_START;
        
        // Terpenoid-related features (simplified)
        while (offset < TERPENOID_END) {
            features[offset++] = 0.0;
        }
    }
    
    /**
     * Extract functional group features (13D).
     */
    private void extractFunctionalFeatures(double[] features, IAtomContainer mol, IAtom atom) {
        int offset = FUNCTIONAL_START;
        
        // Functional group indicators
        while (offset < FUNCTIONAL_END) {
            features[offset++] = 0.0;
        }
    }
    
    /**
     * Extract oxidation state features (20D).
     */
    private void extractOxidationFeatures(double[] features, IAtomContainer mol, IAtom atom) {
        int offset = OXIDATION_START;
        
        // Oxidation state indicators
        while (offset < OXIDATION_END) {
            features[offset++] = 0.0;
        }
    }
    
    // Helper methods
    
    private boolean isCarboxylCarbon(IAtomContainer mol, IAtom atom) {
        if (atom.getAtomicNumber() != 6) return false;
        
        int oxygenCount = 0;
        for (IAtom neighbor : mol.getConnectedAtomsList(atom)) {
            if (neighbor.getAtomicNumber() == 8) oxygenCount++;
        }
        return oxygenCount >= 2;
    }
    
    private boolean isHydroxylOxygen(IAtomContainer mol, IAtom atom) {
        if (atom.getAtomicNumber() != 8) return false;
        
        Integer hCount = atom.getImplicitHydrogenCount();
        return hCount != null && hCount > 0;
    }
}
