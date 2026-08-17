package com.cram.generator;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.CommandLineRunner;
import picocli.CommandLine;

/**
 * CRAM Generator Application
 * 
 * Deep Learning Toolkit for Carboxyl-Rich Alicyclic Molecule Generation.
 * Supports both REST API and CLI modes.
 * 
 * @author KLABKLAYDEE SUPAPORN
 * @version 1.0.0
 */
@SpringBootApplication
public class CramGeneratorApplication implements CommandLineRunner {

    public static void main(String[] args) {
        // Check if running in CLI mode or server mode
        if (args.length > 0 && !args[0].startsWith("--server")) {
            // CLI mode - don't start Spring context fully
            System.exit(new CommandLine(new CramCli()).execute(args));
        } else {
            // Server mode - start Spring Boot
            SpringApplication.run(CramGeneratorApplication.class, args);
        }
    }

    @Override
    public void run(String... args) throws Exception {
        if (args.length == 0 || args[0].equals("--server")) {
            System.out.println("""
                
                ╔═══════════════════════════════════════════════════════════╗
                ║           🧬 CRAM Generator - Java Edition                ║
                ║     Deep Learning Toolkit for Marine DOM Molecules        ║
                ╠═══════════════════════════════════════════════════════════╣
                ║  REST API: http://localhost:8080                          ║
                ║  validate / features / mutation generate                  ║
                ║  CVAE training and sampling: python/ edition              ║
                ╚═══════════════════════════════════════════════════════════╝
                
                """);
        }
    }
}
