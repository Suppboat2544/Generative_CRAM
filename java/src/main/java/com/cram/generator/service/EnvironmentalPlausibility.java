package com.cram.generator.service;

import com.cram.generator.model.CramCriteria;
import com.cram.generator.model.ValidationResult;

/**
 * Marine CRAM van Krevelen window (Hertkorn et al. 2006).
 *
 * Structural CRAM validation (COOH, alicyclic, aliphatic %, MW) is broader.
 * Mutation generators typically place only about 30% of CRAM-valid outputs
 * in this O/C–H/C window. Property-conditioned CVAE stays in python/.
 */
public final class EnvironmentalPlausibility {

    public static final String NOTE =
            "Mutation generators typically pass only ~"
                    + CramCriteria.EXPECTED_ENV_PERCENT
                    + "% environmentally plausible "
                    + "(Hertkorn O/C "
                    + CramCriteria.ENV_OC_MIN + "–" + CramCriteria.ENV_OC_MAX
                    + ", H/C "
                    + CramCriteria.ENV_HC_MIN + "–" + CramCriteria.ENV_HC_MAX
                    + "). CVAE sampling is Python-only.";

    private EnvironmentalPlausibility() {}

    public static boolean inWindow(double oc, double hc) {
        return oc >= CramCriteria.ENV_OC_MIN && oc <= CramCriteria.ENV_OC_MAX
                && hc >= CramCriteria.ENV_HC_MIN && hc <= CramCriteria.ENV_HC_MAX;
    }

    public static boolean isPlausible(ValidationResult result) {
        if (result == null || !result.isValid() || result.getProperties() == null) {
            return false;
        }
        Object oc = result.getProperties().get("ocRatio");
        Object hc = result.getProperties().get("hcRatio");
        if (!(oc instanceof Number) || !(hc instanceof Number)) {
            return false;
        }
        return inWindow(((Number) oc).doubleValue(), ((Number) hc).doubleValue());
    }

    public static String summarize(int envCount, int cramCount) {
        double pct = cramCount == 0 ? 0.0 : 100.0 * envCount / cramCount;
        return String.format(
                "environmentally plausible: %d/%d = %.0f%% (expected ~%d%% for mutation; this smoke often lands 30–50%%)%n%s%n",
                envCount,
                cramCount,
                pct,
                CramCriteria.EXPECTED_ENV_PERCENT,
                NOTE
        );
    }
}
