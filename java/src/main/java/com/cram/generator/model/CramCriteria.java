package com.cram.generator.model;

import lombok.Data;
import lombok.Builder;
import java.util.Map;

/**
 * CRAM validation criteria constants.
 * 
 * Ring class encoding (aligned with notebook):
 * - FUSED_ARO_ALICYCLIC (0): Fused aromatic-alicyclic rings
 * - MIXED_ARO_ALICYCLIC (1): Mixed aromatic-alicyclic rings
 * - BENZENE_ALIPHATIC (2): Benzene with aliphatic chains
 * - PURE_ALICYCLIC (3): Only alicyclic rings
 * - PURE_AROMATIC (4): Only aromatic rings
 * - ACYCLIC (5): No rings
 */
@Data
@Builder
public class CramCriteria {

    // Ring class indices (matching Python PropertyConditionEncoder)
    public static final int RING_CLASS_FUSED_ARO_ALICYCLIC = 0;
    public static final int RING_CLASS_MIXED_ARO_ALICYCLIC = 1;
    public static final int RING_CLASS_BENZENE_ALIPHATIC = 2;
    public static final int RING_CLASS_PURE_ALICYCLIC = 3;
    public static final int RING_CLASS_PURE_AROMATIC = 4;
    public static final int RING_CLASS_ACYCLIC = 5;
    public static final int NUM_RING_CLASSES = 6;

    // Ring size class indices
    public static final int RING_SIZE_STRAINED = 0;      // 3-4 membered
    public static final int RING_SIZE_CRAM_PREFERRED = 1; // 5-7 membered
    public static final int RING_SIZE_LARGE = 2;          // 8+ membered
    public static final int NUM_RING_SIZE_CLASSES = 3;

    /** Minimum number of carboxyl groups required */
    public static final int MIN_CARBOXYL_GROUPS = 2;
    
    /** Maximum number of carboxyl groups */
    public static final int MAX_CARBOXYL_GROUPS = 6;
    
    /** Minimum percentage of aliphatic carbons */
    public static final double MIN_ALIPHATIC_PERCENT = 40.0;
    
    /** Maximum percentage of aliphatic carbons */
    public static final double MAX_ALIPHATIC_PERCENT = 90.0;
    
    /** Minimum molecular weight (Da) */
    public static final double MIN_MOLECULAR_WEIGHT = 100.0;
    
    /** Maximum molecular weight (Da) */
    public static final double MAX_MOLECULAR_WEIGHT = 800.0;
    
    /** SMARTS pattern for carboxyl group (COOH) */
    public static final String CARBOXYL_SMARTS = "[CX3](=O)[OX2H1]";
    
    /** SMARTS pattern for carboxylate (COO-) */
    public static final String CARBOXYLATE_SMARTS = "[CX3](=O)[OX1-]";
    
    /** Allowed atoms for CRAM molecules */
    public static final int[] ALLOWED_ATOMS = {6, 1, 8, 7, 16, 9, 17, 35, 53}; // C, H, O, N, S, F, Cl, Br, I

    /**
     * Ring class name to index mapping.
     */
    public static final Map<String, Integer> RING_CLASS_TO_IDX = Map.of(
        "fused", RING_CLASS_FUSED_ARO_ALICYCLIC,
        "fused_aro_alicyclic", RING_CLASS_FUSED_ARO_ALICYCLIC,
        "mixed", RING_CLASS_MIXED_ARO_ALICYCLIC,
        "mixed_aro_alicyclic", RING_CLASS_MIXED_ARO_ALICYCLIC,
        "benzene_aliphatic", RING_CLASS_BENZENE_ALIPHATIC,
        "pure_alicyclic", RING_CLASS_PURE_ALICYCLIC,
        "pure_aromatic", RING_CLASS_PURE_AROMATIC,
        "acyclic", RING_CLASS_ACYCLIC
    );

    /**
     * Get ring class index from name.
     */
    public static int getRingClassIndex(String name) {
        return RING_CLASS_TO_IDX.getOrDefault(name.toLowerCase().replace(" ", "_"), 0);
    }

    /**
     * Normalize molecular weight to [0, 1] range.
     */
    public static double normalizeMw(double mw) {
        return (mw - MIN_MOLECULAR_WEIGHT) / (MAX_MOLECULAR_WEIGHT - MIN_MOLECULAR_WEIGHT);
    }

    /**
     * Normalize DBE/C ratio to [0, 1] range (typical range 0.3-0.7).
     */
    public static double normalizeDbec(double dbeC) {
        return (dbeC - 0.3) / (0.7 - 0.3);
    }

    /**
     * Normalize DBE/O ratio to [0, 1] range (typical range 0.5-2.5).
     */
    public static double normalizeDbeo(double dbeO) {
        return (dbeO - 0.5) / (2.5 - 0.5);
    }

    /**
     * Normalize DBE/H ratio to [0, 1] range (typical range 0.1-0.4).
     */
    public static double normalizeDbeh(double dbeH) {
        return (dbeH - 0.1) / (0.4 - 0.1);
    }
    
    /**
     * Check if molecular weight is in valid range.
     */
    public static boolean isValidMolecularWeight(double mw) {
        return mw >= MIN_MOLECULAR_WEIGHT && mw <= MAX_MOLECULAR_WEIGHT;
    }
    
    /**
     * Check if aliphatic percentage is sufficient.
     */
    public static boolean isValidAliphaticPercent(double percent) {
        return percent >= MIN_ALIPHATIC_PERCENT;
    }
    
    /**
     * Check if carboxyl count is sufficient.
     */
    public static boolean isValidCarboxylCount(int count) {
        return count >= MIN_CARBOXYL_GROUPS;
    }
}
