package com.cram.generator.service;

import org.openscience.cdk.DefaultChemObjectBuilder;
import org.openscience.cdk.interfaces.IAtom;
import org.openscience.cdk.interfaces.IAtomContainer;
import org.openscience.cdk.interfaces.IAtomType;
import org.openscience.cdk.interfaces.IBond;
import org.openscience.cdk.interfaces.IRingSet;
import org.openscience.cdk.ringsearch.AllRingsFinder;
import org.openscience.cdk.smiles.SmilesParser;
import org.openscience.cdk.tools.manipulator.AtomContainerManipulator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

/**
 * 163-dimensional per-atom features (CDK), aligned with the Python extractor groups.
 */
@Service
public class FeatureExtractor {

    private static final Logger log = LoggerFactory.getLogger(FeatureExtractor.class);

    public static final int FEATURE_DIM = 163;
    public static final int BASE_START = 0, BASE_END = 45;
    public static final int RING_CHAIN_START = 45, RING_CHAIN_END = 65;
    public static final int STEREO_START = 65, STEREO_END = 77;
    public static final int MARINE_START = 77, MARINE_END = 120;
    public static final int TERPENOID_START = 120, TERPENOID_END = 130;
    public static final int FUNCTIONAL_START = 130, FUNCTIONAL_END = 143;
    public static final int OXIDATION_START = 143, OXIDATION_END = 163;

    private final SmilesParser smilesParser;

    public FeatureExtractor() {
        this.smilesParser = new SmilesParser(DefaultChemObjectBuilder.getInstance());
    }

    public double[][] extractFeatures(String smiles) {
        try {
            IAtomContainer molecule = smilesParser.parseSmiles(smiles);
            AtomContainerManipulator.percieveAtomTypesAndConfigureAtoms(molecule);
            org.openscience.cdk.tools.CDKHydrogenAdder
                    .getInstance(DefaultChemObjectBuilder.getInstance())
                    .addImplicitHydrogens(molecule);
            IRingSet rings = new AllRingsFinder().findAllRings(molecule);
            for (IAtom a : molecule.atoms()) {
                a.setIsInRing(false);
            }
            for (IAtomContainer ring : rings.atomContainers()) {
                for (IAtom a : ring.atoms()) {
                    a.setIsInRing(true);
                }
            }

            int numAtoms = molecule.getAtomCount();
            double[][] features = new double[numAtoms][FEATURE_DIM];
            for (int i = 0; i < numAtoms; i++) {
                features[i] = extractAtomFeatures(molecule, molecule.getAtom(i), rings);
            }
            return features;
        } catch (Exception e) {
            log.error("Error extracting features for: {}", smiles, e);
            return new double[0][FEATURE_DIM];
        }
    }

    public double[] extractMoleculeFeatures(String smiles) {
        double[][] atom = extractFeatures(smiles);
        double[] mean = new double[FEATURE_DIM];
        if (atom.length == 0) {
            return mean;
        }
        for (double[] row : atom) {
            for (int i = 0; i < FEATURE_DIM; i++) {
                mean[i] += row[i];
            }
        }
        for (int i = 0; i < FEATURE_DIM; i++) {
            mean[i] /= atom.length;
        }
        return mean;
    }

    public int countNonZero(double[][] atomFeatures) {
        int n = 0;
        for (double[] row : atomFeatures) {
            for (double v : row) {
                if (v != 0.0) {
                    n++;
                }
            }
        }
        return n;
    }

    private double[] extractAtomFeatures(IAtomContainer mol, IAtom atom, IRingSet rings) {
        double[] features = new double[FEATURE_DIM];
        extractBaseFeatures(features, mol, atom);
        extractRingChainFeatures(features, mol, atom, rings);
        extractStereoFeatures(features, atom);
        extractMarineFeatures(features, mol, atom, rings);
        extractTerpenoidFeatures(features, mol, atom, rings);
        extractFunctionalFeatures(features, mol, atom);
        extractOxidationFeatures(features, mol, atom);
        return features;
    }

    private void extractBaseFeatures(double[] features, IAtomContainer mol, IAtom atom) {
        int offset = BASE_START;
        int atomicNum = atom.getAtomicNumber() != null ? atom.getAtomicNumber() : 0;
        int[] atomTypes = {6, 1, 8, 7, 16, 15, 9, 17, 35, 53};
        for (int i = 0; i < atomTypes.length; i++) {
            features[offset + i] = (atomicNum == atomTypes[i]) ? 1.0 : 0.0;
        }
        offset += 10;

        features[offset++] = Math.min(mol.getConnectedBondsCount(atom) / 4.0, 1.0);
        Integer charge = atom.getFormalCharge();
        features[offset++] = (charge != null ? charge : 0) / 2.0;

        IAtomType.Hybridization hyb = atom.getHybridization();
        String hybName = hyb == null ? "" : hyb.name();
        features[offset++] = "SP1".equals(hybName) ? 1.0 : 0.0;
        features[offset++] = "SP2".equals(hybName) ? 1.0 : 0.0;
        features[offset++] = (hyb == null || "SP3".equals(hybName) || "PLANAR3".equals(hybName)) ? 1.0 : 0.0;
        features[offset++] = hybName.startsWith("SP3D") && !"SP3D2".equals(hybName) ? 1.0 : 0.0;
        features[offset++] = "SP3D2".equals(hybName) ? 1.0 : 0.0;

        features[offset++] = atom.isAromatic() ? 1.0 : 0.0;
        features[offset++] = atom.isInRing() ? 1.0 : 0.0;
        Integer hCount = atom.getImplicitHydrogenCount();
        features[offset++] = (hCount != null ? hCount : 0) / 4.0;

        while (offset < BASE_END) {
            features[offset++] = 0.0;
        }
    }

    private void extractRingChainFeatures(double[] features, IAtomContainer mol, IAtom atom, IRingSet rings) {
        int offset = RING_CHAIN_START;
        features[offset++] = atom.isInRing() ? 1.0 : 0.0;

        boolean[] sizeHit = new boolean[6];
        int ringCount = 0;
        int maxSize = 0;
        for (IAtomContainer ring : rings.atomContainers()) {
            if (!ringContains(ring, atom)) {
                continue;
            }
            ringCount++;
            int size = ring.getAtomCount();
            maxSize = Math.max(maxSize, size);
            if (size >= 3 && size <= 8) {
                sizeHit[size - 3] = true;
            }
        }
        for (int i = 0; i < 6; i++) {
            features[offset++] = sizeHit[i] ? 1.0 : 0.0;
        }
        features[offset++] = Math.min(ringCount / 3.0, 1.0);
        features[offset++] = ringCount > 1 ? 1.0 : 0.0;
        features[offset++] = maxSize / 10.0;
        features[offset++] = (!atom.isInRing() && mol.getConnectedBondsCount(atom) <= 2) ? 1.0 : 0.0;

        while (offset < RING_CHAIN_END) {
            features[offset++] = 0.0;
        }
    }

    private void extractStereoFeatures(double[] features, IAtom atom) {
        int offset = STEREO_START;
        Integer parity = atom.getStereoParity();
        features[offset++] = (parity != null && parity != 0) ? 1.0 : 0.0;
        features[offset++] = (parity != null && parity == 1) ? 1.0 : 0.0;
        features[offset++] = (parity != null && parity == 2) ? 1.0 : 0.0;
        while (offset < STEREO_END) {
            features[offset++] = 0.0;
        }
    }

    private void extractMarineFeatures(double[] features, IAtomContainer mol, IAtom atom, IRingSet rings) {
        int offset = MARINE_START;
        int atomicNum = atom.getAtomicNumber() != null ? atom.getAtomicNumber() : 0;
        features[offset++] = (atomicNum == 6 && isCarboxylCarbon(mol, atom)) ? 1.0 : 0.0;
        features[offset++] = (atomicNum == 8 && isHydroxylOxygen(mol, atom)) ? 1.0 : 0.0;
        features[offset++] = (atomicNum == 6 && atom.isInRing() && !atom.isAromatic()) ? 1.0 : 0.0;
        features[offset++] = (atomicNum == 8) ? 1.0 : 0.0;
        features[offset++] = oxygenNeighbors(mol, atom) / 3.0;
        while (offset < MARINE_END) {
            features[offset++] = 0.0;
        }
    }

    private void extractTerpenoidFeatures(double[] features, IAtomContainer mol, IAtom atom, IRingSet rings) {
        int offset = TERPENOID_START;
        int degree = mol.getConnectedBondsCount(atom);
        features[offset++] = (atom.getAtomicNumber() != null && atom.getAtomicNumber() == 6 && degree == 4) ? 1.0 : 0.0;
        features[offset++] = (atom.isInRing() && degree >= 3) ? 1.0 : 0.0;
        int methyl = 0;
        for (IAtom n : mol.getConnectedAtomsList(atom)) {
            if (n.getAtomicNumber() != null && n.getAtomicNumber() == 6 && mol.getConnectedBondsCount(n) == 1) {
                methyl++;
            }
        }
        features[offset++] = Math.min(methyl / 3.0, 1.0);
        while (offset < TERPENOID_END) {
            features[offset++] = 0.0;
        }
    }

    private void extractFunctionalFeatures(double[] features, IAtomContainer mol, IAtom atom) {
        int offset = FUNCTIONAL_START;
        features[offset++] = isCarboxylCarbon(mol, atom) ? 1.0 : 0.0;
        features[offset++] = isHydroxylOxygen(mol, atom) ? 1.0 : 0.0;
        features[offset++] = isCarbonylCarbon(mol, atom) ? 1.0 : 0.0;
        features[offset++] = isEtherOxygen(mol, atom) ? 1.0 : 0.0;
        while (offset < FUNCTIONAL_END) {
            features[offset++] = 0.0;
        }
    }

    private void extractOxidationFeatures(double[] features, IAtomContainer mol, IAtom atom) {
        int offset = OXIDATION_START;
        int oN = oxygenNeighbors(mol, atom);
        features[offset++] = oN / 4.0;
        features[offset++] = oN >= 2 ? 1.0 : 0.0;
        features[offset++] = oN == 1 ? 1.0 : 0.0;
        features[offset++] = (atom.getAtomicNumber() != null && atom.getAtomicNumber() == 6 && oN == 0) ? 1.0 : 0.0;
        while (offset < OXIDATION_END) {
            features[offset++] = 0.0;
        }
    }

    private boolean ringContains(IAtomContainer ring, IAtom atom) {
        for (IAtom a : ring.atoms()) {
            if (a == atom) {
                return true;
            }
        }
        return false;
    }

    private int oxygenNeighbors(IAtomContainer mol, IAtom atom) {
        int n = 0;
        for (IAtom neighbor : mol.getConnectedAtomsList(atom)) {
            if (neighbor.getAtomicNumber() != null && neighbor.getAtomicNumber() == 8) {
                n++;
            }
        }
        return n;
    }

    private boolean isCarboxylCarbon(IAtomContainer mol, IAtom atom) {
        if (atom.getAtomicNumber() == null || atom.getAtomicNumber() != 6) {
            return false;
        }
        return oxygenNeighbors(mol, atom) >= 2;
    }

    private boolean isHydroxylOxygen(IAtomContainer mol, IAtom atom) {
        if (atom.getAtomicNumber() == null || atom.getAtomicNumber() != 8) {
            return false;
        }
        Integer hCount = atom.getImplicitHydrogenCount();
        return hCount != null && hCount > 0 && mol.getConnectedBondsCount(atom) == 1;
    }

    private boolean isCarbonylCarbon(IAtomContainer mol, IAtom atom) {
        if (atom.getAtomicNumber() == null || atom.getAtomicNumber() != 6) {
            return false;
        }
        for (IBond bond : mol.getConnectedBondsList(atom)) {
            IAtom other = bond.getOther(atom);
            if (other.getAtomicNumber() != null && other.getAtomicNumber() == 8
                    && bond.getOrder() == IBond.Order.DOUBLE) {
                return true;
            }
        }
        return false;
    }

    private boolean isEtherOxygen(IAtomContainer mol, IAtom atom) {
        if (atom.getAtomicNumber() == null || atom.getAtomicNumber() != 8) {
            return false;
        }
        if (mol.getConnectedBondsCount(atom) != 2) {
            return false;
        }
        Integer hCount = atom.getImplicitHydrogenCount();
        return hCount == null || hCount == 0;
    }
}
