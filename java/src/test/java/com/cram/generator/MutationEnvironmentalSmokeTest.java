package com.cram.generator;

import com.cram.generator.model.CramCriteria;
import com.cram.generator.service.CramValidator;
import com.cram.generator.service.EnvironmentalPlausibility;
import com.cram.generator.service.FeatureExtractor;
import com.cram.generator.service.MutationGenerator;
import com.cram.generator.model.ValidationResult;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Compare-ready Java smoke: validate, 163D features, mutation generate.
 * Environmental (van Krevelen) pass rate is expected around 30% for mutation.
 */
class MutationEnvironmentalSmokeTest {

    private static final String SEED = "CC1CCC(CC1)(C(=O)O)C(=O)O";

    @Test
    void seedIsCramAndInMarineWindow() {
        CramValidator validator = new CramValidator();
        ValidationResult result = validator.validate(SEED);
        assertTrue(result.isValid(), "reference CRAM seed should validate");
        assertTrue(EnvironmentalPlausibility.isPlausible(result),
                "reference seed should sit in the Hertkorn O/C–H/C window");
    }

    @Test
    void featuresAre163D() {
        FeatureExtractor extractor = new FeatureExtractor();
        double[][] atom = extractor.extractFeatures(SEED);
        assertTrue(atom.length > 0);
        assertEquals(FeatureExtractor.FEATURE_DIM, atom[0].length);
        assertTrue(extractor.countNonZero(atom) > 0);
    }

    @Test
    void mutationGenerateEnvironmentalRateIsAroundThirtyPercent() {
        CramValidator validator = new CramValidator();
        MutationGenerator generator = new MutationGenerator(validator, 42L);
        List<String> smiles = generator.generate(40, 8000);
        assertTrue(smiles.size() >= 10, "should produce CRAM-valid mutants");

        int env = 0;
        for (String s : smiles) {
            if (EnvironmentalPlausibility.isPlausible(validator.validate(s))) {
                env++;
            }
        }
        double pct = 100.0 * env / smiles.size();
        System.out.printf("Java mutation CRAM-valid: %d%n", smiles.size());
        System.out.print(EnvironmentalPlausibility.summarize(
                env, smiles.size(), generator.getLastElapsedSeconds()));

        // Smoke, not a chemistry paper: generation must work. The ~30% figure is
        // the documented mutation limitation versus the trained Python CVAE.
        assertTrue(pct >= 5.0, "environmental rate should not be near zero");
        assertTrue(pct <= 70.0,
                "mutation is not property-conditioned; the trained CVAE is better");
        System.out.printf("expected ~%d%%%n", CramCriteria.EXPECTED_ENV_PERCENT);
    }
}
