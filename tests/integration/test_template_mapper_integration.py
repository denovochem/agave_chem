from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from rdkit import Chem

from agave_chem.mappers.template.template_mapper import TemplateReactionMapper

DEFAULT_TEMPLATE_MAPPER_LOWER_BOUND_RATIO_FULLY_MAPPED_PRODUCTS = 0.4


def _iter_rxn_smiles_from_file(path: Path) -> List[str]:
    rxns: List[str] = []
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s:
            continue
        rxns.append(s)
    return rxns


def _mapping_atomic_nums_from_side(side_smiles: str) -> Dict[int, int]:
    mapping: Dict[int, int] = {}
    for frag in side_smiles.split("."):
        mol = Chem.MolFromSmiles(frag)
        assert mol is not None
        for atom in mol.GetAtoms():
            amap = atom.GetAtomMapNum()
            if amap == 0:
                continue
            assert amap not in mapping
            mapping[amap] = atom.GetAtomicNum()
    return mapping


def _count_atoms_and_mapped_atoms(side_smiles: str) -> tuple[int, int]:
    num_atoms = 0
    num_mapped_atoms = 0
    for frag in side_smiles.split("."):
        mol = Chem.MolFromSmiles(frag)
        assert mol is not None
        for atom in mol.GetAtoms():
            num_atoms += 1
            if atom.GetAtomMapNum() != 0:
                num_mapped_atoms += 1
    return num_atoms, num_mapped_atoms


def test_template_mapper_uspto_1k_reactions_sanity() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    rxn_path = repo_root / "tests" / "integration" / "uspto_1k_reactions.txt"

    rxns = _iter_rxn_smiles_from_file(rxn_path)
    assert len(rxns) > 0

    mapper = TemplateReactionMapper(mapper_name="integration_test")

    num_mapped = 0
    num_total = 0

    for i, rxn in enumerate(rxns):
        num_total += 1
        res = mapper.map_reaction(rxn)
        mapped = res["selected_mapping"]
        if not mapped:
            continue

        num_mapped += 1
        assert mapped.count(">>") == 1

        r_smiles, p_smiles = mapped.split(">>")

        r_atomic_nums = _mapping_atomic_nums_from_side(r_smiles)
        p_atomic_nums = _mapping_atomic_nums_from_side(p_smiles)

        # Sanity: any map number that appears is unique within each side.
        assert len(r_atomic_nums) == len(set(r_atomic_nums.keys()))
        assert len(p_atomic_nums) == len(set(p_atomic_nums.keys()))

        # Sanity: corresponding mapped atoms have the same element.
        shared = set(r_atomic_nums.keys()) & set(p_atomic_nums.keys())
        for amap in shared:
            assert r_atomic_nums[amap] == p_atomic_nums[amap]

        # Helpful failure context.
        if i == -1:  # pragma: no cover
            raise AssertionError(mapped)

    assert num_mapped > 0

    ratio = num_mapped / num_total
    print(f"Fully-mapped product reaction ratio: {ratio}")
    assert ratio > DEFAULT_TEMPLATE_MAPPER_LOWER_BOUND_RATIO_FULLY_MAPPED_PRODUCTS


if __name__ == "__main__":
    test_template_mapper_uspto_1k_reactions_sanity()
