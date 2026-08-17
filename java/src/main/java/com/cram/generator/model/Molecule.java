package com.cram.generator.model;

import lombok.Data;
import lombok.Builder;
import lombok.AllArgsConstructor;
import lombok.NoArgsConstructor;

/**
 * Represents a molecule with its SMILES and properties.
 */
@Data
@Builder
@AllArgsConstructor
@NoArgsConstructor
public class Molecule {

    /** SMILES representation */
    private String smiles;
    
    /** Canonical SMILES */
    private String canonicalSmiles;
    
    /** Molecular weight */
    private double molecularWeight;
    
    /** Molecular formula */
    private String formula;
    
    /** Number of atoms */
    private int atomCount;
    
    /** Number of bonds */
    private int bondCount;
    
    /** Number of rings */
    private int ringCount;
    
    /** Number of carboxyl groups */
    private int carboxylCount;
    
    /** Percentage of aliphatic carbons */
    private double aliphaticPercent;
    
    /** Has alicyclic ring */
    private boolean hasAlicyclicRing;
    
    /** Is valid CRAM */
    private boolean validCram;
    
    /** CRAM score */
    private double cramScore;
    
    /** H/C ratio */
    private double hcRatio;
    
    /** O/C ratio */
    private double ocRatio;
}
