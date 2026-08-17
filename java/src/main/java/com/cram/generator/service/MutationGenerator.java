package com.cram.generator.service;

import com.cram.generator.model.ValidationResult;
import org.openscience.cdk.DefaultChemObjectBuilder;
import org.openscience.cdk.interfaces.IAtomContainer;
import org.openscience.cdk.smiles.SmiFlavor;
import org.openscience.cdk.smiles.SmilesGenerator;
import org.openscience.cdk.smiles.SmilesParser;
import org.openscience.cdk.tools.manipulator.AtomContainerManipulator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

/**
 * Mutation-based CRAM generation (CDK).
 * This is not the Python CVAE. It mutates seed SMILES, parses with CDK,
 * and keeps molecules that pass {@link CramValidator}.
 */
@Service
public class MutationGenerator {

    private static final Logger log = LoggerFactory.getLogger(MutationGenerator.class);

    static final String[] DEFAULT_SEEDS = {
            "CC1CCC(CC1)(C(=O)O)C(=O)O",
            "OC(=O)C1CCCCC1C(=O)O",
            "CC(C)C1CCC(CC1)(C(=O)O)C(=O)O",
            "OC(=O)C1CCC(C(=O)O)CC1",
            "CC1CC(C(=O)O)CC(C(=O)O)C1",
            "OC(=O)CC1CCCCC1C(=O)O"
    };

    static final String[] FRAGMENTS = {
            "C(=O)O", "CC", "CCC", "C", "C(C)C", "O"
    };

    private final CramValidator validator;
    private final SmilesParser smilesParser;
    private final SmilesGenerator smilesGenerator;
    private final Random random = new Random();

    public MutationGenerator(CramValidator validator) {
        this.validator = validator;
        this.smilesParser = new SmilesParser(DefaultChemObjectBuilder.getInstance());
        this.smilesGenerator = new SmilesGenerator(SmiFlavor.Canonical);
    }

    public List<String> generate(int n, int maxAttempts) {
        Set<String> unique = new LinkedHashSet<>();
        int attempts = 0;
        while (unique.size() < n && attempts < maxAttempts) {
            attempts++;
            String seed = DEFAULT_SEEDS[random.nextInt(DEFAULT_SEEDS.length)];
            String candidate = mutate(seed);
            String canonical = canonicalize(candidate);
            if (canonical == null) {
                continue;
            }
            ValidationResult result = validator.validate(canonical);
            if (result.isValid()) {
                unique.add(result.getCanonicalSmiles() != null
                        ? result.getCanonicalSmiles()
                        : canonical);
            }
        }
        log.info("Generated {}/{} CRAM molecules in {} attempts", unique.size(), n, attempts);
        return new ArrayList<>(unique);
    }

    private String mutate(String smiles) {
        int mode = random.nextInt(3);
        return switch (mode) {
            case 0 -> insertFragment(smiles);
            case 1 -> extendCarbon(smiles);
            default -> swapCarboxylPlacement(smiles);
        };
    }

    private String insertFragment(String smiles) {
        String frag = FRAGMENTS[random.nextInt(FRAGMENTS.length)];
        int at = Math.max(1, random.nextInt(Math.max(smiles.length(), 1)));
        return smiles.substring(0, at) + frag + smiles.substring(at);
    }

    private String extendCarbon(String smiles) {
        int idx = smiles.indexOf('C');
        if (idx < 0) {
            return smiles + "C";
        }
        return smiles.substring(0, idx) + "CC" + smiles.substring(idx + 1);
    }

    private String swapCarboxylPlacement(String smiles) {
        if (smiles.contains("C(=O)O")) {
            return smiles.replaceFirst("C\\(=O\\)O", "CC(=O)O");
        }
        return smiles + "C(=O)O";
    }

    private String canonicalize(String smiles) {
        try {
            IAtomContainer mol = smilesParser.parseSmiles(smiles);
            AtomContainerManipulator.percieveAtomTypesAndConfigureAtoms(mol);
            return smilesGenerator.create(mol);
        } catch (Exception e) {
            return null;
        }
    }
}
