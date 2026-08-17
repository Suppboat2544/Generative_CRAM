"""
Marine CRAM van Krevelen window (Hertkorn et al. 2006).

Structural CRAM validation is broader. Mutation generators typically place
only about 30% of CRAM-valid outputs in this O/C–H/C window.
Property-conditioned CVAE stays in python/.
"""

const ENV_OC_MIN = 0.40
const ENV_OC_MAX = 0.60
const ENV_HC_MIN = 1.20
const ENV_HC_MAX = 1.70
const EXPECTED_ENV_PERCENT = 30

const ENV_NOTE = "Mutation generators typically pass only ~$(EXPECTED_ENV_PERCENT)% environmentally plausible (Hertkorn O/C $(ENV_OC_MIN)–$(ENV_OC_MAX), H/C $(ENV_HC_MIN)–$(ENV_HC_MAX)). CVAE sampling is Python-only."

"""
    elemental_ratios(mol) -> NamedTuple{(:oc, :hc, :nC, :nO, :nH), ...}
"""
function elemental_ratios(mol)
    nC = 0
    nO = 0
    nH = 0
    for i in 1:nodecount(mol)
        sym = string(atomsymbol(getatom(mol, i)))
        if sym == "C"
            nC += 1
        elseif sym == "O"
            nO += 1
        elseif sym == "H"
            nH += 1
        end
        nH += implicithcount(mol, i)
    end
    oc = nC == 0 ? 0.0 : nO / nC
    hc = nC == 0 ? 0.0 : nH / nC
    return (oc = oc, hc = hc, nC = nC, nO = nO, nH = nH)
end

function in_env_window(oc::Float64, hc::Float64)::Bool
    return ENV_OC_MIN <= oc <= ENV_OC_MAX && ENV_HC_MIN <= hc <= ENV_HC_MAX
end

"""
    is_environmentally_plausible(smiles::String) -> Bool

True when the molecule is structural CRAM and sits in the Hertkorn window.
"""
function is_environmentally_plausible(smiles::String)::Bool
    result = validate_cram(smiles)
    if !result.is_cram
        return false
    end
    mol = smiles_to_mol(smiles)
    if isnothing(mol)
        return false
    end
    ratios = elemental_ratios(mol)
    return in_env_window(ratios.oc, ratios.hc)
end

function summarize_env(env_count::Int, cram_count::Int)::String
    pct = cram_count == 0 ? 0.0 : 100.0 * env_count / cram_count
    return "environmentally plausible: $(env_count)/$(cram_count) = $(round(Int, pct))% (expected ~$(EXPECTED_ENV_PERCENT)% for mutation; this smoke often lands 30–50%)\n$(ENV_NOTE)"
end
