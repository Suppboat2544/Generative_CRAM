package com.cram.generator;

import com.cram.generator.service.CramValidator;
import com.cram.generator.service.EnvironmentalPlausibility;
import com.cram.generator.service.FeatureExtractor;
import com.cram.generator.service.MutationGenerator;
import com.cram.generator.model.ValidationResult;
import picocli.CommandLine;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;
import picocli.CommandLine.Parameters;

import java.io.*;
import java.util.concurrent.Callable;

/**
 * Command Line Interface for CRAM Generator.
 */
@Command(
    name = "cram",
    mixinStandardHelpOptions = true,
    version = "CRAM Generator 1.0.0",
    description = "CRAM validation, 163D features, and mutation-based generation (CVAE is Python-only)",
    subcommands = {
        CramCli.ValidateCommand.class,
        CramCli.BatchCommand.class,
        CramCli.FeaturesCommand.class,
        CramCli.GenerateCommand.class,
        CramCli.SmokeCommand.class,
        CramCli.InfoCommand.class
    }
)
public class CramCli implements Callable<Integer> {

    @Override
    public Integer call() {
        CommandLine.usage(this, System.out);
        return 0;
    }

    /**
     * Validate a single SMILES string.
     */
    @Command(name = "validate", description = "Validate a SMILES string as CRAM")
    static class ValidateCommand implements Callable<Integer> {

        @Parameters(index = "0", description = "SMILES string to validate")
        private String smiles;

        @Option(names = {"-v", "--verbose"}, description = "Show detailed output")
        private boolean verbose;

        @Override
        public Integer call() {
            CramValidator validator = new CramValidator();
            ValidationResult result = validator.validate(smiles);

            printValidationResult(result, verbose);
            return result.isValid() ? 0 : 1;
        }

        private void printValidationResult(ValidationResult result, boolean verbose) {
            System.out.println();
            System.out.println("═══════════════════════════════════════════════════════════");
            System.out.println("🧬 CRAM Validation Report");
            System.out.println("═══════════════════════════════════════════════════════════");
            System.out.println();
            System.out.println("📋 SMILES: " + result.getSmiles());
            
            if (result.getCanonicalSmiles() != null) {
                System.out.println("📋 Canonical: " + result.getCanonicalSmiles());
            }
            
            System.out.println();
            System.out.println("──────────────────────────────────────────────────────────");
            System.out.println("📊 CRAM Criteria:");
            System.out.println("──────────────────────────────────────────────────────────");
            
            printCriterion("Carboxyl groups", 
                result.getCarboxylCount() >= 2,
                String.format("%d (≥2 required)", result.getCarboxylCount()));
            
            printCriterion("Alicyclic ring", 
                result.isHasAlicyclicRing(),
                result.isHasAlicyclicRing() ? "Present" : "Absent");
            
            printCriterion("Aliphatic carbon", 
                result.getAliphaticPercent() >= 60,
                String.format("%.1f%% (≥60%% required)", result.getAliphaticPercent()));
            
            printCriterion("Molecular weight", 
                result.getMolecularWeight() >= 100 && result.getMolecularWeight() <= 750,
                String.format("%.1f Da (100-750 range)", result.getMolecularWeight()));
            
            System.out.println();
            System.out.println("──────────────────────────────────────────────────────────");
            System.out.println("🎯 Result: " + (result.isValid() ? 
                "✅ VALID CRAM MOLECULE" : "❌ NOT A CRAM MOLECULE"));
            System.out.println("═══════════════════════════════════════════════════════════");
            System.out.println();

            if (verbose && result.getProperties() != null) {
                System.out.println("📊 Additional Properties:");
                result.getProperties().forEach((key, value) -> 
                    System.out.printf("   %s: %s%n", key, value));
            }
        }

        private void printCriterion(String name, boolean passed, String value) {
            String icon = passed ? "✅" : "❌";
            System.out.printf("  %s %s: %s%n", icon, name, value);
        }
    }

    /**
     * Batch validate multiple SMILES from file.
     */
    @Command(name = "batch", description = "Validate multiple SMILES from file")
    static class BatchCommand implements Callable<Integer> {

        @Parameters(index = "0", description = "Input file with SMILES (one per line)")
        private File inputFile;

        @Option(names = {"-o", "--output"}, description = "Output CSV file")
        private File outputFile;

        @Override
        public Integer call() {
            CramValidator validator = new CramValidator();
            
            try (BufferedReader reader = new BufferedReader(new FileReader(inputFile))) {
                PrintWriter writer = outputFile != null ? 
                    new PrintWriter(new FileWriter(outputFile)) : null;
                
                if (writer != null) {
                    writer.println("smiles,valid,carboxyl_count,aliphatic_percent,molecular_weight,has_alicyclic_ring");
                }

                int total = 0, valid = 0;
                String line;
                
                while ((line = reader.readLine()) != null) {
                    String smiles = line.trim();
                    if (smiles.isEmpty() || smiles.startsWith("#")) continue;
                    
                    ValidationResult result = validator.validate(smiles);
                    total++;
                    if (result.isValid()) valid++;
                    
                    if (writer != null) {
                        writer.printf("%s,%b,%d,%.2f,%.2f,%b%n",
                            smiles,
                            result.isValid(),
                            result.getCarboxylCount(),
                            result.getAliphaticPercent(),
                            result.getMolecularWeight(),
                            result.isHasAlicyclicRing());
                    }
                }

                System.out.printf("%n📊 Batch Validation Complete%n");
                System.out.printf("   Total: %d%n", total);
                System.out.printf("   Valid CRAM: %d (%.1f%%)%n", valid, 100.0 * valid / total);
                System.out.printf("   Invalid: %d%n", total - valid);
                
                if (writer != null) {
                    writer.close();
                    System.out.printf("   Results saved to: %s%n", outputFile.getAbsolutePath());
                }

                return 0;
            } catch (IOException e) {
                System.err.println("Error: " + e.getMessage());
                return 1;
            }
        }
    }

    @Command(name = "features", description = "Extract 163D atom features from a SMILES string")
    static class FeaturesCommand implements Callable<Integer> {

        @Parameters(index = "0", description = "SMILES string")
        private String smiles;

        @Option(names = {"-v", "--verbose"}, description = "Print molecule-level mean vector")
        private boolean verbose;

        @Override
        public Integer call() {
            FeatureExtractor extractor = new FeatureExtractor();
            double[][] atom = extractor.extractFeatures(smiles);
            if (atom.length == 0) {
                System.err.println("Could not extract features (invalid SMILES).");
                return 1;
            }
            System.out.println("SMILES: " + smiles);
            System.out.printf("Atoms: %d%n", atom.length);
            System.out.printf("Atom features: %d x %d%n", atom.length, FeatureExtractor.FEATURE_DIM);
            System.out.printf("Non-zero values: %d%n", extractor.countNonZero(atom));
            if (verbose) {
                double[] mean = extractor.extractMoleculeFeatures(smiles);
                System.out.print("Molecule mean (first 16 dims):");
                for (int i = 0; i < 16 && i < mean.length; i++) {
                    System.out.printf(" %.3f", mean[i]);
                }
                System.out.println();
            }
            return 0;
        }
    }

    @Command(name = "generate", description = "Mutation-based CRAM generation (not the Python CVAE)")
    static class GenerateCommand implements Callable<Integer> {

        @Option(names = {"-n", "--count"}, description = "Number of molecules", defaultValue = "20")
        private int count;

        @Option(names = {"--max-attempts"}, description = "Maximum mutation attempts", defaultValue = "5000")
        private int maxAttempts;

        @Option(names = {"--seed"}, description = "Random seed")
        private Long seed;

        @Option(names = {"-o", "--output"}, description = "Output CSV file")
        private File outputFile;

        @Override
        public Integer call() {
            CramValidator validator = new CramValidator();
            MutationGenerator generator = seed == null
                    ? new MutationGenerator(validator)
                    : new MutationGenerator(validator, seed);
            var smiles = generator.generate(count, maxAttempts);
            System.out.printf("Generated %d CRAM molecules (mutation / CDK)%n", smiles.size());
            if (outputFile != null) {
                try (PrintWriter writer = new PrintWriter(new FileWriter(outputFile))) {
                    writer.println("smiles");
                    for (String s : smiles) {
                        writer.println(s);
                    }
                    System.out.printf("Saved %s%n", outputFile.getAbsolutePath());
                } catch (IOException e) {
                    System.err.println("Error: " + e.getMessage());
                    return 1;
                }
            } else {
                for (String s : smiles) {
                    System.out.println(s);
                }
            }
            return 0;
        }
    }

    @Command(name = "smoke", description = "Validate, features, generate; report ~30% environmental rate")
    static class SmokeCommand implements Callable<Integer> {

        @Option(names = {"-n", "--count"}, description = "Molecules to generate", defaultValue = "40")
        private int count;

        @Option(names = {"--seed"}, description = "Random seed", defaultValue = "42")
        private long seed;

        @Override
        public Integer call() {
            final String seedSmiles = "CC1CCC(CC1)(C(=O)O)C(=O)O";
            CramValidator validator = new CramValidator();
            ValidationResult seedResult = validator.validate(seedSmiles);
            if (!seedResult.isValid() || !EnvironmentalPlausibility.isPlausible(seedResult)) {
                System.err.println("smoke FAIL: reference seed is not CRAM / not in Hertkorn window");
                return 1;
            }
            System.out.println("validate: PASS");

            FeatureExtractor extractor = new FeatureExtractor();
            double[][] atom = extractor.extractFeatures(seedSmiles);
            if (atom.length == 0 || atom[0].length != FeatureExtractor.FEATURE_DIM) {
                System.err.println("smoke FAIL: 163D features");
                return 1;
            }
            System.out.printf("features: %d x %d PASS%n", atom.length, FeatureExtractor.FEATURE_DIM);

            MutationGenerator generator = new MutationGenerator(validator, seed);
            var smiles = generator.generate(count, 8000);
            int env = 0;
            for (String s : smiles) {
                if (EnvironmentalPlausibility.isPlausible(validator.validate(s))) {
                    env++;
                }
            }
            System.out.printf("generate: %d CRAM-valid%n", smiles.size());
            System.out.print(EnvironmentalPlausibility.summarize(env, smiles.size()));
            if (smiles.size() < 10) {
                System.err.println("smoke FAIL: too few CRAM-valid mutants");
                return 1;
            }
            System.out.println("smoke: PASS");
            return 0;
        }
    }

    /**
     * Show CRAM criteria information.
     */
    @Command(name = "info", description = "Show CRAM criteria information")
    static class InfoCommand implements Callable<Integer> {

        @Override
        public Integer call() {
            System.out.println("""
                
                ╔═══════════════════════════════════════════════════════════╗
                ║              🧬 CRAM Molecule Criteria                    ║
                ╚═══════════════════════════════════════════════════════════╝
                
                CRAM (Carboxyl-Rich Alicyclic Molecules) are key components
                of marine dissolved organic matter (DOM).
                
                ┌───────────────────────────────────────────────────────────┐
                │ Criterion            │ Requirement                       │
                ├───────────────────────────────────────────────────────────┤
                │ Carboxyl groups      │ ≥2 (COOH or COO⁻)                 │
                │ Ring structure       │ Alicyclic (non-aromatic)          │
                │ Aliphatic carbon     │ >60%                              │
                │ Molecular weight     │ 100-750 Da                        │
                └───────────────────────────────────────────────────────────┘
                
                Example valid CRAM: CC1CCC(CC1)(C(=O)O)C(=O)O

                Java edition: validate, 163D features, mutation generate.
                Mutation typically passes only ~30% environmentally plausible
                (Hertkorn O/C 0.40–0.60, H/C 1.20–1.70). CVAE stays in python/.
                
                """);
            return 0;
        }
    }
}
