package com.cram.generator.model;

import lombok.Data;
import lombok.Builder;
import lombok.AllArgsConstructor;
import lombok.NoArgsConstructor;

import java.util.Map;
import java.util.HashMap;

/**
 * Result of CRAM validation containing all criteria checks.
 */
@Data
@Builder
@AllArgsConstructor
@NoArgsConstructor
public class ValidationResult {

    /** Original input SMILES */
    private String smiles;
    
    /** Canonical SMILES (if parseable) */
    private String canonicalSmiles;
    
    /** Whether molecule passes all CRAM criteria */
    private boolean valid;
    
    /** Number of carboxyl groups (COOH or COO-) */
    private int carboxylCount;
    
    /** Whether molecule contains alicyclic (non-aromatic) ring */
    private boolean hasAlicyclicRing;
    
    /** Percentage of aliphatic carbons */
    private double aliphaticPercent;
    
    /** Molecular weight in Daltons */
    private double molecularWeight;
    
    /** CRAM score (0-1) */
    private double cramScore;
    
    /** Error message if validation failed */
    private String errorMessage;
    
    /** Additional computed properties */
    @Builder.Default
    private Map<String, Object> properties = new HashMap<>();
    
    /**
     * Create a failed validation result.
     */
    public static ValidationResult failed(String smiles, String error) {
        return ValidationResult.builder()
            .smiles(smiles)
            .valid(false)
            .errorMessage(error)
            .build();
    }
    
    /**
     * Check if all criteria are met.
     */
    public boolean meetsAllCriteria() {
        return carboxylCount >= 2 
            && hasAlicyclicRing 
            && aliphaticPercent >= 60.0 
            && molecularWeight >= 100 
            && molecularWeight <= 750;
    }
}
