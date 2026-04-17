from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

from rdkit import Chem

from agave_chem.mappers.mcs.mcs_mapper import MCSReactionMapper

DEFAULT_MCS_MAPPER_LOWER_BOUND_RATIO_MAPPED_PRODUCT_ATOMS = 0.7


def _iter_rxn_smiles_from_file(path: Path) -> List[str]:
    rxns: List[str] = []
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s:
            continue
        rxns.append(s)
    return rxns


def _atom_signature(atom: Chem.Atom) -> Tuple[int, int, int, int, int, int]:
    return (
        atom.GetAtomicNum(),
        atom.GetFormalCharge(),
        1 if atom.GetIsAromatic() else 0,
        1 if atom.IsInRing() else 0,
        atom.GetTotalNumHs(),
        atom.GetDegree(),
    )


def _mapping_dict_from_side(
    side_smiles: str,
) -> Dict[int, tuple[int, int, int, int, int, int]]:
    mapping: Dict[int, tuple[int, int, int, int, int, int]] = {}
    for frag in side_smiles.split("."):
        mol = Chem.MolFromSmiles(frag)
        assert mol is not None
        for atom in mol.GetAtoms():
            amap = atom.GetAtomMapNum()
            if amap == 0:
                continue
            assert amap not in mapping
            mapping[amap] = _atom_signature(atom)
    return mapping


def test_mcs_mapper_uspto_1k_reactions_sanity() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    rxn_path = repo_root / "tests" / "integration" / "uspto_1k_reactions.txt"

    rxns = _iter_rxn_smiles_from_file(rxn_path)
    assert len(rxns) > 0

    mapper = MCSReactionMapper(mapper_name="integration_test")

    num_mapped = 0
    product_num_atoms = 0
    product_mapped_atoms = 0
    for i, rxn in enumerate(rxns):
        res = mapper.map_reaction(rxn)
        mapped = res["selected_mapping"]

        product_str = mapped.split(">>")[-1]
        product_mol = Chem.MolFromSmiles(product_str)
        for atom in product_mol.GetAtoms():
            product_num_atoms += 1
            if atom.GetAtomMapNum() != 0:
                product_mapped_atoms += 1

        num_mapped += 1
        assert mapped.count(">>") == 1

        r_smiles, p_smiles = mapped.split(">>")
        r_map = _mapping_dict_from_side(r_smiles)
        p_map = _mapping_dict_from_side(p_smiles)

        # Sanity: any map number that appears is unique within each side.
        assert len(r_map) == len(set(r_map.keys()))
        assert len(p_map) == len(set(p_map.keys()))

        # Sanity: corresponding mapped atoms have identical atom properties.
        shared = set(r_map.keys()) & set(p_map.keys())
        for amap in shared:
            assert r_map[amap] == p_map[amap]

        # Helpful failure context.
        if i == -1:  # pragma: no cover
            raise AssertionError(mapped)

    ratio = product_mapped_atoms / product_num_atoms
    print(f"Product atom mapping ratio: {ratio}")
    assert ratio > DEFAULT_MCS_MAPPER_LOWER_BOUND_RATIO_MAPPED_PRODUCT_ATOMS

    assert num_mapped > 0


## TODO: potentially add rdkit MCS check as well
if __name__ == "__main__":
    test_mcs_mapper_uspto_1k_reactions_sanity()
