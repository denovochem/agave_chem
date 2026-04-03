import json
from importlib.resources import files

from rdchiral import main as rdc
from rdkit import Chem

from agave_chem.mappers.template.template_initialization import (
    expand_all_brackets,
    verify_validity_of_template,
)


def class_str_key(d: dict) -> tuple:
    # Sort numerically by each dotted component, e.g. "4.2.10" > "4.2.2"
    s = d.get("class_str")
    if not s:
        return (float("inf"),)  # push missing/empty class_str to the end
    return tuple(int(part) for part in s.split("."))


if __name__ == "__main__":
    SMIRKS_PATTERNS_FILE = files("agave_chem.datafiles.smirks_patterns").joinpath(
        "smirks_patterns.json"
    )
    default_smirks_patterns = []
    with SMIRKS_PATTERNS_FILE.open("r") as f:
        default_smirks_patterns = json.load(f)

    default_smirks_with_children = []
    for default_smirk_pattern in default_smirks_patterns:
        smirks_list = []
        smirks = (
            default_smirk_pattern["smirks"].split(">>")[1]
            + ">>"
            + default_smirk_pattern["smirks"].split(">>")[0]
        )

        if len(expand_all_brackets(smirks)) < 100:
            smirks_list.extend(expand_all_brackets(smirks))

        filtered_smirks_list = []
        for smirk in smirks_list:
            products_smarts = [
                Chem.MolFromSmarts(smarts) for smarts in smirk.split(">>")[0].split(".")
            ]

            if None in products_smarts:
                continue

            reactants_smarts = [
                Chem.MolFromSmarts(smarts) for smarts in smirk.split(">>")[1].split(".")
            ]

            if None in reactants_smarts:
                continue

            try:
                rdc_rxn = rdc.rdchiralReaction(smirk)
            except Exception as e:
                print(f"Error converting smirks to rdchiral reaction: {e}")
                continue

            if not verify_validity_of_template(smirk):
                continue

            filtered_smirks_list.append(smirk)

        default_smirk_pattern["child_smirks"] = filtered_smirks_list
        default_smirk_pattern["class_str"] = (
            f"{default_smirk_pattern['superclass_id']}.{default_smirk_pattern['class_id']}.{default_smirk_pattern['subclass_id']}"
        )

        default_smirks_with_children.append(default_smirk_pattern)

    records_sorted = sorted(default_smirks_with_children, key=class_str_key)

    with open(
        "/home/csnbritt/projects/denovochem_projects/agave_chem/agave_chem/datafiles/smirks_patterns/smirks_patterns_with_children.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(records_sorted, f, indent=2, ensure_ascii=False)
