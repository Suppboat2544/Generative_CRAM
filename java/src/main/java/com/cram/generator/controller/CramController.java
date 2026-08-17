package com.cram.generator.controller;

import com.cram.generator.model.ValidationResult;
import com.cram.generator.service.CramValidator;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * REST API controller for CRAM validation.
 */
@RestController
@RequestMapping("/api/v1")
@RequiredArgsConstructor
@CrossOrigin(origins = "*")
public class CramController {

    private final CramValidator cramValidator;

    /**
     * Health check endpoint.
     */
    @GetMapping("/health")
    public ResponseEntity<Map<String, String>> health() {
        return ResponseEntity.ok(Map.of(
            "status", "healthy",
            "service", "CRAM Generator",
            "version", "1.0.0"
        ));
    }

    /**
     * Validate a single SMILES string.
     * 
     * @param smiles The SMILES string to validate
     * @return ValidationResult
     */
    @GetMapping("/validate")
    public ResponseEntity<ValidationResult> validate(@RequestParam String smiles) {
        ValidationResult result = cramValidator.validate(smiles);
        return ResponseEntity.ok(result);
    }

    /**
     * Validate a single SMILES string (POST).
     */
    @PostMapping("/validate")
    public ResponseEntity<ValidationResult> validatePost(@RequestBody ValidateRequest request) {
        ValidationResult result = cramValidator.validate(request.smiles());
        return ResponseEntity.ok(result);
    }

    /**
     * Batch validate multiple SMILES.
     */
    @PostMapping("/validate/batch")
    public ResponseEntity<BatchValidationResponse> validateBatch(
            @RequestBody BatchValidateRequest request) {
        
        List<ValidationResult> results = request.smilesList().stream()
            .map(cramValidator::validate)
            .collect(Collectors.toList());
        
        long validCount = results.stream().filter(ValidationResult::isValid).count();
        
        return ResponseEntity.ok(new BatchValidationResponse(
            results.size(),
            (int) validCount,
            results.size() - (int) validCount,
            results
        ));
    }

    /**
     * Get CRAM criteria information.
     */
    @GetMapping("/criteria")
    public ResponseEntity<Map<String, Object>> getCriteria() {
        return ResponseEntity.ok(Map.of(
            "minCarboxylGroups", 2,
            "minAliphaticPercent", 60.0,
            "minMolecularWeight", 100.0,
            "maxMolecularWeight", 750.0,
            "allowedAtoms", List.of("C", "H", "O", "N", "S", "F", "Cl", "Br", "I"),
            "description", "Carboxyl-Rich Alicyclic Molecules found in marine DOM"
        ));
    }

    // Request/Response DTOs
    
    public record ValidateRequest(String smiles) {}
    
    public record BatchValidateRequest(List<String> smilesList) {}
    
    public record BatchValidationResponse(
        int total,
        int valid,
        int invalid,
        List<ValidationResult> results
    ) {}
}
