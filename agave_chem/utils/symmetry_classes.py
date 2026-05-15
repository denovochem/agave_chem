from functools import reduce
from typing import Any, List

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

_TAUTOMER_ENUMERATOR = rdMolStandardize.TautomerEnumerator()

# Define transformations focusing on making equivalent atoms within
# functional groups appear identical, using neutral forms with explicit charges
TRANSFORMS = [
    # ============================================================================
    # NITROGEN-OXYGEN FUNCTIONAL GROUPS
    # ============================================================================
    # Nitro group: Two oxygens are equivalent
    ("Nitro_charged", "[N+:1](=[O:2])[O-:3]>>[N+0:1](=[O+0:2])=[O+0:3]"),
    # Nitrate ester: Two oxygens on nitrogen are equivalent (RO-NO2)
    ("Nitrate_ester", "[O:1][N+:2]([O-:3])=[O:4]>>[O:1][N+0:2](=[O+0:3])=[O+0:4]"),
    # Nitrate anion: All three oxygens equivalent (standardize to one form)
    (
        "Nitrate_anion",
        "[O-:1][N+:2](=[O:3])[O;H1:4]>>[O;H0;+0:1]=[N+0:2](=[O;H0;+0:3])=[O;H0;+0:4]",
    ),
    (
        "Nitrate_anion_deprotonated",
        "[O-:1][N+:2](=[O:3])-[O-:4]>>[O;H0;+0:1]=[N+0:2](=[O;H0;+0:3])=[O;H0;+0:4]",
    ),
    # Nitrite: Two oxygens are equivalent
    ("Nitrite_anion", "[O-:1][N+:2]=[O:3]>>[O+0:1]=[N-1:2]=[O+0:3]"),
    # ============================================================================
    # SULFUR-OXYGEN FUNCTIONAL GROUPS
    # ============================================================================
    # Sulfone: Two oxygens are equivalent
    ("Sulfone_2", "[S+:1]([*:2])(=[O:3])[O-:4]>>[S+0:1]([*:2])(=[O+0:3])=[O+0:4]"),
    # Sulfonic acid/Sulfonate: Two =O oxygens are equivalent, third is -OH or -O-
    ("Sulfonate_1", "[S+:1]([*:2])([O-:3])=[O:4]>>[S+0:1]([*:2])(=[O+0:3])=[O+0:4]"),
    (
        "Sulfonate_2",
        "[S+:1]([*:2])(=[O:3])(=[O:4])[O-:5]>>[S+0:1]([*:2])(=[O+0:3])(=[O+0:4])[O-:5]",
    ),
    ("Sulfonate_3", "[S+:1]([*:2])([O-:3])[O-:4]>>[S+0:1]([*:2])(=[O-:3])[O-:4]"),
    # Sulfate ester: Multiple equivalent oxygens (RO-SO3- or RO-SO2-OR)
    (
        "Sulfate_ester_1",
        "[S+:1](=[O:2])(=[O:3])([O:4])[O-:5]>>[S+0:1](=[O+0:2])(=[O+0:3])([O:4])[O-:5]",
    ),
    (
        "Sulfate_ester_2",
        "[S+:1]([O:2])([O-:3])=[O:4]>>[S+0:1]([O:2])(=[O+0:3])=[O+0:4]",
    ),
    # Sulfate anion: Four oxygens (all equivalent in free anion)
    (
        "Sulfate_anion",
        "[S+:1]([O-:2])([O-:3])(=[O:4])=[O:5]>>[S+0:1]([O-:2])([O-:3])(=[O+0:4])=[O+0:5]",
    ),
    # Sulfonamide: Two oxygens on sulfur are equivalent
    ("Sulfonamide", "[S+:1]([O-:2])(=[O:3])[N:4]>>[S+0:1](=[O+0:2])(=[O+0:3])[N:4]"),
    # Sulfamate:
    ("Sulfamate", "[S+:1]([O-:2])([O:3])[N:4]>>[S+0:1](=[O+0:2])([O:3])[N:4]"),
    # ============================================================================
    # PHOSPHORUS-OXYGEN FUNCTIONAL GROUPS
    # ============================================================================
    # Phosphate: Multiple oxygens (in esters, the =O vs -O- should be standardized)
    (
        "Phosphate_1",
        "[P+:1](=[O:2])([O:3])([O:4])[O-:5]>>[P+0:1](=[O+0:2])([O:3])([O:4])[O-:5]",
    ),
    (
        "Phosphate_2",
        "[P+:1]([O-:2])([O:3])([O:4])=[O:5]>>[P+0:1]([O-:2])([O:3])([O:4])=[O+0:5]",
    ),
    (
        "Phosphate_3",
        "[P+:1]([O-:2])([O-:3])([O:4])=[O:5]>>[P+0:1]([O-:2])([O-:3])([O:4])=[O+0:5]",
    ),
    # Phosphonate
    ("Phosphonate_1", "[P+:1]([*:2])([O-:3])=[O:4]>>[P+0:1]([*:2])([O-:3])=[O+0:4]"),
    ("Phosphonate_2", "[P+:1]([*:2])(=[O:3])[O-:4]>>[P+0:1]([*:2])(=[O+0:3])[O-:4]"),
    # Phosphinate
    ("Phosphinate", "[P+:1]([*:2])([*:3])[O-:4]>>[P+0:1]([*:2])([*:3])[O-:4]"),
    # Phosphonium oxide (different from phosphine oxide)
    (
        "Phosphonium_oxide",
        "[P+:1]([*:2])([*:3])([*:4])[O-:5]>>[P+0:1]([*:2])([*:3])([*:4])=[O+0:5]",
    ),
    # ============================================================================
    # CARBON-OXYGEN FUNCTIONAL GROUPS
    # ============================================================================
    # Carboxylic acid (neutral form - oxygens are actually different: C=O vs C-OH)
    # Only neutralize if you want COOH instead of COO-
    ("Carboxylate_to_acid", "[C:1](=[O:2])[O-:3]>>[C:1](=[O+0:2])[O+0:3]"),
    # Carbonate ester: The two ester oxygens (R-O-) vs carbonyl =O
    # Standardize to neutral form
    ("Carbonate_charged", "[O:1][C+:2]([O:3])[O-:4]>>[O:1][C+0:2]([O:3])=[O+0:4]"),
    # ============================================================================
    # SELENIUM (analogous to sulfur)
    # ============================================================================
    ("Selenone", "[Se+:1]([*:2])(=[O:3])[O-:4]>>[Se+0:1]([*:2])(=[O+0:3])=[O+0:4]"),
    ("Selenonic", "[Se+:1]([*:2])(=[O:3])=[O:4]>>[Se+0:1]([*:2])(=[O+0:3])=[O+0:4]"),
    # ============================================================================
    # CHLORINE/HALOGEN-OXYGEN (common in inorganic reagents)
    # ============================================================================
    # Chlorate: Three oxygens (two =O, one -O-)
    ("Chlorate", "[Cl+:1]([O-:2])(=[O:3])=[O:4]>>[Cl+0:1]([O-:2])(=[O+0:3])=[O+0:4]"),
    # Perchlorate: Four oxygens (three =O, one -O-)
    (
        "Perchlorate",
        "[Cl+:1]([O-:2])(=[O:3])(=[O:4])=[O:5]>>[Cl+0:1]([O-:2])(=[O+0:3])(=[O+0:4])=[O+0:5]",
    ),
    # Chlorite: Two oxygens
    ("Chlorite", "[Cl+:1]([O-:2])=[O:3]>>[Cl+0:1]([O-:2])=[O+0:3]"),
    # Bromate/Iodate (analogous to chlorate)
    ("Bromate", "[Br+:1]([O-:2])(=[O:3])=[O:4]>>[Br+0:1]([O-:2])(=[O+0:3])=[O+0:4]"),
    ("Iodate", "[I+:1]([O-:2])(=[O:3])=[O:4]>>[I+0:1]([O-:2])(=[O+0:3])=[O+0:4]"),
    (
        "Periodate",
        "[I+:1]([O-:2])(=[O:3])(=[O:4])=[O:5]>>[I+0:1]([O-:2])(=[O+0:3])(=[O+0:4])=[O+0:5]",
    ),
    # ============================================================================
    # BORON-OXYGEN (boronic acids, borate esters)
    # ============================================================================
    # Boronate: Three oxygens (in borate anion, all equivalent)
    ("Boronate", "[B+:1]([O-:2])([O:3])[O:4]>>[B+0:1]([O-:2])([O:3])[O:4]"),
    # ============================================================================
    # CHROMIUM/MANGANESE (common oxidizing agents)
    # ============================================================================
    # Chromate: Four oxygens (all equivalent in CrO4 2-)
    (
        "Chromate",
        "[Cr+:1]([O-:2])([O-:3])(=[O:4])=[O:5]>>[Cr+0:1]([O-:2])([O-:3])(=[O+0:4])=[O+0:5]",
    ),
    # Dichromate: Bridge structure, but standardize terminal oxygens
    (
        "Dichromate_terminal",
        "[Cr+:1]([O-:2])(=[O:3])=[O:4]>>[Cr+0:1]([O-:2])(=[O+0:3])=[O+0:4]",
    ),
    # Permanganate: Four oxygens (all equivalent in MnO4-)
    (
        "Permanganate",
        "[Mn+:1]([O-:2])(=[O:3])(=[O:4])=[O:5]>>[Mn+0:1]([O-:2])(=[O+0:3])(=[O+0:4])=[O+0:5]",
    ),
    # ============================================================================
    # ARSENIC (arsenate, arsenite - similar to phosphate)
    # ============================================================================
    (
        "Arsenate",
        "[As+:1](=[O:2])([O:3])([O:4])[O-:5]>>[As+0:1](=[O+0:2])([O:3])([O:4])[O-:5]",
    ),
    ("Arsenite", "[As+:1]([O-:2])([O:3])[O:4]>>[As+0:1]([O-:2])([O:3])[O:4]"),
]


def build_normalizer() -> Any:
    """
    Build an RDKit Normalizer from the module-level TRANSFORMS SMIRKS rules.

    Iterates over the ``TRANSFORMS`` list of (name, smirks) tuples and formats
    them into the custom-normalization string expected by
    ``rdMolStandardize.NormalizerFromData``. The resulting normalizer can be
    used to neutralize charged resonance forms and make equivalent atoms
    within functional groups (e.g. nitro, sulfone, sulfonate, phosphate,
    halogen oxyanions) appear identical.

    Returns:
        Any: An RDKit ``Normalizer`` instance configured with the custom
        functional-group standardization rules.
    """
    custom_normalizations = "//\tName\tSMIRKS\n"
    for name, smirks in TRANSFORMS:
        custom_normalizations += f"{name}\t{smirks}\n"

    params = rdMolStandardize.CleanupParameters()
    normalizer = rdMolStandardize.NormalizerFromData(custom_normalizations, params)

    return normalizer


NORMALIZER = build_normalizer()


def normalize_mol(mol: Chem.Mol) -> Chem.Mol:
    """
    Normalize an RDKit molecule by applying functional group standardization transforms.

    The molecule is passed through the module-level ``NORMALIZER`` (built from
    custom SMIRKS rules) to neutralize charged resonance forms and make
    equivalent atoms within functional groups (e.g. nitro, sulfone, sulfonate)
    appear identical. The result is then round-tripped through SMILES with
    ``sanitize=False`` and the property cache is updated non-strictly to
    ensure a clean, consistent molecular representation.

    Args:
        mol (Chem.Mol): The molecule to normalize.

    Returns:
        Chem.Mol: The normalized molecule.
    """
    normalized_mol = NORMALIZER.normalize(mol)
    normalized_mol = Chem.MolFromSmiles(
        Chem.MolToSmiles(normalized_mol), sanitize=False
    )
    normalized_mol.UpdatePropertyCache(strict=False)
    return normalized_mol


def normalize_smiles(smiles: str) -> str:
    """
    Normalize a SMILES string by applying functional group standardization transforms.

    The input SMILES is parsed into an RDKit molecule, normalized via
    `normalize_mol` to neutralize charged resonance forms and make
    equivalent atoms within functional groups identical, then converted
    back to a SMILES string without canonicalization.

    Args:
        smiles (str): The SMILES string to normalize.

    Returns:
        str: The normalized SMILES string.

    Raises:
        Exception: If the SMILES string is invalid and cannot be parsed by RDKit.
    """
    mol = Chem.MolFromSmiles(smiles)
    mol = normalize_mol(mol)
    return Chem.MolToSmiles(mol, canonical=False)


def get_symmetry_class_from_mol(
    mol: Chem.Mol, consider_tautomers: bool = True, consider_transforms: bool = True
) -> List[int]:
    """
    Compute symmetry class labels for each atom in a molecule.

    Optionally applies functional group normalization transforms and/or tautomeric
    invariance before computing canonical symmetry classes via RDKit's
    `CanonicalRankAtoms`. Atoms sharing the same integer label are symmetrically
    equivalent under the requested invariances.

    Args:
        mol (Chem.Mol): The input RDKit molecule.
        consider_tautomers (bool): If True, atoms that interconvert via tautomerism
            are assigned the same symmetry class.
        consider_transforms (bool): If True, apply functional group normalization
            transforms (e.g., standardizing charged resonance forms) before
            computing symmetry classes.

    Returns:
        List[int]: A list of integer symmetry class labels, one per atom in the
            molecule, where atoms with the same value are symmetrically equivalent.
    """

    mol_copy = Chem.Mol(mol)

    if consider_transforms:
        mol_copy = normalize_mol(mol_copy)

    if consider_tautomers:
        return resolve_symmtery_class_for_tautomers(mol_copy)

    canonical_ranks = list(Chem.CanonicalRankAtoms(mol_copy, breakTies=False))

    return canonical_ranks


def get_symmetry_class_from_smiles(
    smiles: str, consider_tautomers: bool = True, consider_transforms: bool = True
) -> List[int]:
    """
    Compute symmetry class labels for each atom in a molecule from its SMILES string.

    This is a convenience wrapper around get_symmetry_class_from_mol that first parses
    the SMILES string into an RDKit molecule, then computes canonical symmetry classes
    with optional tautomeric invariance and functional group normalization.

    Args:
        smiles (str): The SMILES representation of the molecule.
        consider_tautomers (bool): If True, atoms that interconvert via tautomerism are
            assigned the same symmetry class.
        consider_transforms (bool): If True, apply functional group normalization
            transforms before computing symmetry classes.

    Returns:
        List[int]: A list of integer symmetry class labels, one per atom in the molecule,
            where atoms with the same value are symmetrically equivalent.

    Raises:
        Exception: If the SMILES string is invalid and cannot be parsed by RDKit.
    """
    mol = Chem.MolFromSmiles(smiles)
    return get_symmetry_class_from_mol(mol, consider_tautomers, consider_transforms)


def resolve_symmtery_class_for_tautomers(mol: Chem.Mol) -> List[int]:
    """
    Compute symmetry classes for a molecule, merging classes of atoms that become
    equivalent through tautomeric rearrangements.

    This function enumerates all tautomers of the input molecule and uses the
    canonical ranking of a combined molecule (reference + tautomer) to identify
    atoms in the reference that are symmetric with atoms in the tautomer. When
    such a symmetry is found and the two atoms originally had different symmetry
    classes, their classes are merged (the tautomer atom's class is updated to
    match the reference atom's class). This produces symmetry classes that are
    invariant to tautomerism.

    Args:
        mol (Chem.Mol): The input molecule for which to compute tautomer-resolved
            symmetry classes.

    Returns:
        List[int]: A list of integer symmetry class labels, one per atom in the
            input molecule, where atoms that can interconvert via tautomerism
            share the same class label.
    """
    reference_mol = Chem.Mol(mol)
    reference_mol_canonical_ranks = list(
        Chem.CanonicalRankAtoms(reference_mol, breakTies=False)
    )
    mol_with_atom_maps = Chem.Mol(mol)
    [atom.SetAtomMapNum(atom.GetIdx() + 1) for atom in mol_with_atom_maps.GetAtoms()]
    tautomers_with_atom_maps = list(_TAUTOMER_ENUMERATOR.Enumerate(mol_with_atom_maps))

    for tautomer_with_atom_map in tautomers_with_atom_maps:
        tautomer = Chem.Mol(tautomer_with_atom_map)
        [atom.SetAtomMapNum(0) for atom in tautomer.GetAtoms()]
        combo = reduce(Chem.CombineMols, [reference_mol, tautomer])
        canonical_ranks = list(Chem.CanonicalRankAtoms(combo, breakTies=False))

        middle_index = len(canonical_ranks) // 2

        first_half = canonical_ranks[:middle_index]
        second_half = canonical_ranks[middle_index:]

        for i, reference_rank in enumerate(first_half):
            for j, tautomer_rank in enumerate(second_half):
                if (
                    reference_rank == tautomer_rank
                    and reference_mol_canonical_ranks[i]
                    != reference_mol_canonical_ranks[j]
                ):
                    reference_mol_canonical_ranks[j] = reference_mol_canonical_ranks[i]

    return reference_mol_canonical_ranks
