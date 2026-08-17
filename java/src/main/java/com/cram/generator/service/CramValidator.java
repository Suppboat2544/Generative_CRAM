package com.cram.generator.service;

import com.cram.generator.model.CramCriteria;
import com.cram.generator.model.ValidationResult;
import org.openscience.cdk.DefaultChemObjectBuilder;
import org.openscience.cdk.exception.InvalidSmilesException;
import org.openscience.cdk.interfaces.IAtom;
import org.openscience.cdk.interfaces.IAtomContainer;
import org.openscience.cdk.interfaces.IRingSet;
import org.openscience.cdk.isomorphism.Pattern;
import org.openscience.cdk.ringsearch.AllRingsFinder;
import org.openscience.cdk.smiles.SmilesParser;
import org.openscience.cdk.smiles.SmilesGenerator;
import org.openscience.cdk.smiles.SmiFlavor;
import org.openscience.cdk.smarts.SmartsPattern;
import org.openscience.cdk.tools.manipulator.AtomContainerManipulator;
import org.openscience.cdk.qsar.descriptors.molecular.WeightDescriptor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import java.util.HashMap;
import java.util.Map;

/**
 * Service for validating molecules against CRAM criteria.
 * 
 * CRAM (Carboxyl-Rich Alicyclic Molecules) must satisfy:
 * - ≥2 Carboxyl groups (COOH or COO⁻)
 * - Alicyclic ring (non-aromatic)
 * - >60% aliphatic carbon
 * - Molecular weight 100-750 Da
 */
@Service
public class CramValidator {

    private static final Logger log = LoggerFactory.getLogger(CramValidator.class);
    
    private final SmilesParser smilesParser;
    private final SmilesGenerator smilesGenerator;
    private final Pattern carboxylPattern;
    private final Pattern carboxylatePattern;
    
    public CramValidator() {
        this.smilesParser = new SmilesParser(DefaultChemObjectBuilder.getInstance());
        this.smilesGenerator = new SmilesGenerator(SmiFlavor.Canonical);
        
        try {
            this.carboxylPattern = SmartsPattern.create(CramCriteria.CARBOXYL_SMARTS);
            this.carboxylatePattern = SmartsPattern.create(CramCriteria.CARBOXYLATE_SMARTS);
        } catch (Exception e) {
            throw new RuntimeException("Failed to initialize SMARTS patterns", e);
        }
    }
    
    /**
     * Validate a SMILES string against CRAM criteria.
     * 
     * @param smiles The SMILES string to validate
     * @return ValidationResult with all criteria checks
     */
    public ValidationResult validate(String smiles) {
        if (smiles == null || smiles.trim().isEmpty()) {
            return ValidationResult.failed(smiles, "Empty SMILES string");
        }
        
        try {
            // Parse SMILES
            IAtomContainer molecule = smilesParser.parseSmiles(smiles.trim());
            AtomContainerManipulator.percieveAtomTypesAndConfigureAtoms(molecule);
            org.openscience.cdk.tools.CDKHydrogenAdder
                    .getInstance(DefaultChemObjectBuilder.getInstance())
                    .addImplicitHydrogens(molecule);
            
            // Generate canonical SMILES
            String canonical = smilesGenerator.create(molecule);
            
            // Count carboxyl groups
            int carboxylCount = countCarboxylGroups(molecule);
            
            // Check for alicyclic ring
            boolean hasAlicyclicRing = hasAlicyclicRing(molecule);
            
            // Calculate aliphatic carbon percentage
            double aliphaticPercent = calculateAliphaticPercent(molecule);
            
            // Calculate molecular weight
            double molecularWeight = calculateMolecularWeight(molecule);
            
            // Check all criteria
            boolean valid = CramCriteria.isValidCarboxylCount(carboxylCount)
                && hasAlicyclicRing
                && CramCriteria.isValidAliphaticPercent(aliphaticPercent)
                && CramCriteria.isValidMolecularWeight(molecularWeight);
            
            // Calculate CRAM score
            double cramScore = calculateCramScore(carboxylCount, hasAlicyclicRing, 
                aliphaticPercent, molecularWeight);
            
            // Additional properties
            Map<String, Object> properties = calculateProperties(molecule);
            
            return ValidationResult.builder()
                .smiles(smiles)
                .canonicalSmiles(canonical)
                .valid(valid)
                .carboxylCount(carboxylCount)
                .hasAlicyclicRing(hasAlicyclicRing)
                .aliphaticPercent(aliphaticPercent)
                .molecularWeight(molecularWeight)
                .cramScore(cramScore)
                .properties(properties)
                .build();
                
        } catch (InvalidSmilesException e) {
            log.warn("Invalid SMILES: {}", smiles);
            return ValidationResult.failed(smiles, "Invalid SMILES: " + e.getMessage());
        } catch (Exception e) {
            log.error("Error validating SMILES: {}", smiles, e);
            return ValidationResult.failed(smiles, "Validation error: " + e.getMessage());
        }
    }
    
    /**
     * Count carboxyl groups (COOH and COO-).
     */
    private int countCarboxylGroups(IAtomContainer molecule) {
        int count = 0;
        
        try {
            // Count COOH groups
            int coohMatches = carboxylPattern.matchAll(molecule).count();
            count += coohMatches;
            
            // Count COO- groups
            int cooMatches = carboxylatePattern.matchAll(molecule).count();
            count += cooMatches;
        } catch (Exception e) {
            log.warn("Error counting carboxyl groups", e);
        }
        
        return count;
    }
    
    /**
     * Check if molecule contains an alicyclic (non-aromatic) ring.
     */
    private boolean hasAlicyclicRing(IAtomContainer molecule) {
        try {
            AllRingsFinder ringFinder = new AllRingsFinder();
            IRingSet rings = ringFinder.findAllRings(molecule);
            
            for (IAtomContainer ring : rings.atomContainers()) {
                boolean isAromatic = false;
                for (IAtom atom : ring.atoms()) {
                    if (atom.isAromatic()) {
                        isAromatic = true;
                        break;
                    }
                }
                if (!isAromatic) {
                    return true; // Found a non-aromatic ring
                }
            }
        } catch (Exception e) {
            log.warn("Error checking alicyclic rings", e);
        }
        
        return false;
    }
    
    /**
     * Calculate percentage of aliphatic carbons.
     */
    private double calculateAliphaticPercent(IAtomContainer molecule) {
        int totalCarbons = 0;
        int aliphaticCarbons = 0;
        
        for (IAtom atom : molecule.atoms()) {
            if (atom.getAtomicNumber() == 6) { // Carbon
                totalCarbons++;
                if (!atom.isAromatic()) {
                    aliphaticCarbons++;
                }
            }
        }
        
        if (totalCarbons == 0) return 0.0;
        return 100.0 * aliphaticCarbons / totalCarbons;
    }
    
    /**
     * Calculate molecular weight.
     */
    private double calculateMolecularWeight(IAtomContainer molecule) {
        try {
            WeightDescriptor descriptor = new WeightDescriptor();
            return Double.parseDouble(
                descriptor.calculate(molecule).getValue().toString());
        } catch (Exception e) {
            // Fallback: sum atomic weights
            double weight = 0;
            for (IAtom atom : molecule.atoms()) {
                weight += atom.getExactMass() != null ? atom.getExactMass() : 0;
            }
            return weight;
        }
    }
    
    /**
     * Calculate CRAM score (0-1).
     */
    private double calculateCramScore(int carboxylCount, boolean hasAlicyclicRing,
                                      double aliphaticPercent, double molecularWeight) {
        double score = 0.0;
        
        // Carboxyl contribution (0-0.3)
        score += Math.min(carboxylCount / 3.0, 1.0) * 0.3;
        
        // Alicyclic ring contribution (0-0.25)
        if (hasAlicyclicRing) score += 0.25;
        
        // Aliphatic contribution (0-0.25)
        score += Math.min(aliphaticPercent / 100.0, 1.0) * 0.25;
        
        // Molecular weight contribution (0-0.2)
        if (CramCriteria.isValidMolecularWeight(molecularWeight)) {
            // Prefer mid-range weights
            double midWeight = (CramCriteria.MIN_MOLECULAR_WEIGHT + 
                               CramCriteria.MAX_MOLECULAR_WEIGHT) / 2;
            double deviation = Math.abs(molecularWeight - midWeight) / midWeight;
            score += (1.0 - deviation) * 0.2;
        }
        
        return Math.min(score, 1.0);
    }
    
    /**
     * Calculate additional molecular properties.
     */
    private Map<String, Object> calculateProperties(IAtomContainer molecule) {
        Map<String, Object> props = new HashMap<>();
        
        int hCount = 0, cCount = 0, oCount = 0, nCount = 0;
        
        for (IAtom atom : molecule.atoms()) {
            Integer atomicNum = atom.getAtomicNumber();
            if (atomicNum == null) {
                continue;
            }
            switch (atomicNum) {
                case 1 -> hCount++;
                case 6 -> cCount++;
                case 8 -> oCount++;
                case 7 -> nCount++;
            }
            Integer implicitH = atom.getImplicitHydrogenCount();
            if (implicitH != null) {
                hCount += implicitH;
            }
        }
        
        props.put("atomCount", molecule.getAtomCount());
        props.put("bondCount", molecule.getBondCount());
        props.put("carbonCount", cCount);
        props.put("hydrogenCount", hCount);
        props.put("oxygenCount", oCount);
        props.put("nitrogenCount", nCount);
        
        if (cCount > 0) {
            props.put("hcRatio", (double) hCount / cCount);
            props.put("ocRatio", (double) oCount / cCount);
        }
        
        return props;
    }
}
