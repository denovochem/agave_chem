"""
generate_nuc_sub_smirks.py
--------------------------
Enumerate plausible nucleophilic-substitution SMIRKS patterns by combining
electrophilic centers, leaving groups, and nucleophiles with chemical
compatibility filtering.
"""

import itertools
import json
import uuid
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from rdchiral import main as rdc
from rdkit import Chem

# ---------------------------------------------------------------------------
# 1.  Building-block definitions
# ---------------------------------------------------------------------------


@dataclass
class ElectrophilicCenter:
    """
    Describes the carbon (or atom) being attacked.

    Args:
        name (str): Human-readable label.
        reactant_smarts (str): SMARTS for the electrophilic atom *with*
            a placeholder ``{LG}`` where the leaving-group bond attaches.
            Must contain exactly one atom-mapped atom (the electrophilic C)
            using map number :1.
        product_smarts (str): SMARTS for the same atom after substitution,
            with ``{NU}`` placeholder for the new bond to the nucleophile.
            Same map :1.
        compatible_lg_tags (Set[str]): Which leaving-group categories work here.
        compatible_nu_tags (Set[str]): Which nucleophile categories work here.
    """

    name: str
    reactant_smarts: str
    product_smarts: str
    compatible_lg_tags: Set[str] = field(default_factory=set)
    compatible_nu_tags: Set[str] = field(default_factory=set)


@dataclass
class LeavingGroup:
    """
    Describes the departing fragment.

    Args:
        name (str): Human-readable label.
        smarts (str): SMARTS fragment that bonds to the electrophilic center.
            No atom map numbers (these atoms leave).
        tags (Set[str]): Category tags for compatibility filtering.
    """

    name: str
    smarts: str
    tags: Set[str] = field(default_factory=set)


@dataclass
class Nucleophile:
    """
    Describes the incoming nucleophile.

    Args:
        name (str): Human-readable label.
        reactant_smarts (str): Full SMARTS of the nucleophile as a *separate*
            reactant fragment.  The attacking atom carries map :10.
            Other mapped atoms use :11, :12, … as needed.
        product_smarts (str): SMARTS of the nucleophile fragment after bond
            formation (map :10 is the atom that bonds to :1).
        tags (Set[str]): Category tags for compatibility filtering.
    """

    name: str
    reactant_smarts: str  # separate reactant molecule
    product_smarts: str  # fragment bonded to electrophilic C in product
    tags: Set[str] = field(default_factory=set)


# ---------------------------------------------------------------------------
# 2.  Define the building blocks
# ---------------------------------------------------------------------------

# --- Electrophilic centers ------------------------------------------------

ELECTROPHILIC_CENTERS: List[ElectrophilicCenter] = [
    # SN2 at sp3 carbon (generic alkyl)
    ElectrophilicCenter(
        name="sp3 alkyl",
        reactant_smarts="[C;H1;D3;+0:1]-{LG}",
        product_smarts="[C;+0:1]-{NU}",
        compatible_lg_tags={"halide", "sulfonate", "epoxide", "phosphonate"},
        compatible_nu_tags={
            "N_nuc",
            "O_nuc",
            "S_nuc",
            "azide",
            "cyanide",
            "carboxylate",
            "heterocyclic_N",
            "enolate",
            "halide_nuc",
            "C_nuc",
            "P_nuc",
        },
    ),
    ElectrophilicCenter(
        name="sp3 alkyl",
        reactant_smarts="[C;H2;D2;+0:1]-{LG}",
        product_smarts="[C;+0:1]-{NU}",
        compatible_lg_tags={"halide", "sulfonate", "epoxide", "phosphonate"},
        compatible_nu_tags={
            "N_nuc",
            "O_nuc",
            "S_nuc",
            "azide",
            "cyanide",
            "carboxylate",
            "heterocyclic_N",
            "enolate",
            "halide_nuc",
            "C_nuc",
            "P_nuc",
        },
    ),
    # SN2 at sp3 benzylic
    ElectrophilicCenter(
        name="sp3 benzylic",
        reactant_smarts="[C;H1;D3;+0:1](-[c:2])-{LG}",
        product_smarts="[C;+0:1](-[c:2])-{NU}",
        compatible_lg_tags={"halide", "sulfonate"},
        compatible_nu_tags={
            "N_nuc",
            "O_nuc",
            "S_nuc",
            "azide",
            "cyanide",
            "carboxylate",
            "heterocyclic_N",
            "C_nuc",
            "P_nuc",
        },
    ),
    ElectrophilicCenter(
        name="sp3 benzylic",
        reactant_smarts="[C;H2;D2;+0:1](-[c:2])-{LG}",
        product_smarts="[C;+0:1](-[c:2])-{NU}",
        compatible_lg_tags={"halide", "sulfonate"},
        compatible_nu_tags={
            "N_nuc",
            "O_nuc",
            "S_nuc",
            "azide",
            "cyanide",
            "carboxylate",
            "heterocyclic_N",
            "C_nuc",
            "P_nuc",
        },
    ),
    # Acyl (carbonyl) — nucleophilic acyl substitution
    ElectrophilicCenter(
        name="acyl (carbonyl)",
        reactant_smarts="[C;H0;D3;+0:1](=[O;H0;D1;+0:2])-{LG}",
        product_smarts="[C;H0;D3;+0:1](=[O;H0;D1;+0:2])-{NU}",
        compatible_lg_tags={"halide", "acyloxy", "activated_ester"},
        compatible_nu_tags={
            "N_nuc",
            "O_nuc",
            "S_nuc",
            "heterocyclic_N",
        },
    ),
    # SNAr — electron-deficient aromatic ring
    ElectrophilicCenter(
        name="SNAr aromatic",
        reactant_smarts="[c;H0;D3;+0:1]-{LG}",
        product_smarts="[c;H0;D3;+0:1]-{NU}",
        compatible_lg_tags={"halide", "nitro_lg"},
        compatible_nu_tags={
            "N_nuc",
            "O_nuc",
            "S_nuc",
            "heterocyclic_N",
        },
    ),
    # Epoxide opening (special — LG is part of the ring)
    ElectrophilicCenter(
        name="epoxide",
        reactant_smarts="[C;+0:1]1-[O;H0;D2;+0:2]-[C;+0:3]1",
        product_smarts="[C;+0:1](-[O;+0:2])-[C;+0:3]-{NU}",
        compatible_lg_tags={"epoxide"},
        compatible_nu_tags={
            "N_nuc",
            "O_nuc",
            "S_nuc",
            "azide",
            "cyanide",
            "heterocyclic_N",
        },
    ),
    # SN2 at sp3 allylic carbon
    ElectrophilicCenter(
        name="sp3 allylic",
        reactant_smarts="[C;H1;D3;+0:1](-[C:2]=[C:3])-{LG}",
        product_smarts="[C;+0:1](-[C:2]=[C:3])-{NU}",
        compatible_lg_tags={"halide", "sulfonate"},
        compatible_nu_tags={
            "N_nuc",
            "O_nuc",
            "S_nuc",
            "azide",
            "cyanide",
            "carboxylate",
            "heterocyclic_N",
            "enolate",
            "halide_nuc",
            "C_nuc",
            "P_nuc",
        },
    ),
    ElectrophilicCenter(
        name="sp3 allylic",
        reactant_smarts="[C;H2;D2;+0:1](-[C:2]=[C:3])-{LG}",
        product_smarts="[C;+0:1](-[C:2]=[C:3])-{NU}",
        compatible_lg_tags={"halide", "sulfonate"},
        compatible_nu_tags={
            "N_nuc",
            "O_nuc",
            "S_nuc",
            "azide",
            "cyanide",
            "carboxylate",
            "heterocyclic_N",
            "enolate",
            "halide_nuc",
            "C_nuc",
            "P_nuc",
        },
    ),
    # SN2 at sp3 propargylic carbon
    ElectrophilicCenter(
        name="sp3 propargylic",
        reactant_smarts="[C;H2;D2;+0:1](-[C:2]#[C:3])-{LG}",
        product_smarts="[C;+0:1](-[C:2]#[C:3])-{NU}",
        compatible_lg_tags={"halide", "sulfonate"},
        compatible_nu_tags={
            "N_nuc",
            "O_nuc",
            "S_nuc",
            "azide",
            "cyanide",
            "carboxylate",
            "heterocyclic_N",
            "halide_nuc",
            "C_nuc",
        },
    ),
    ElectrophilicCenter(
        name="sp3 propargylic",
        reactant_smarts="[C;H1;D3;+0:1](-[C:2]#[C:3])-{LG}",
        product_smarts="[C;+0:1](-[C:2]#[C:3])-{NU}",
        compatible_lg_tags={"halide", "sulfonate"},
        compatible_nu_tags={
            "N_nuc",
            "O_nuc",
            "S_nuc",
            "azide",
            "cyanide",
            "carboxylate",
            "heterocyclic_N",
            "halide_nuc",
            "C_nuc",
        },
    ),
    # SN2 at sp3 carbon alpha to EWG (carbonyl, nitrile, sulfonyl)
    ElectrophilicCenter(
        name="sp3 alpha to EWG",
        reactant_smarts="[C;H1;D3;+0:1](-[C:2]=[O,S,N:3])-{LG}",
        product_smarts="[C;+0:1](-[C:2]=[O,S,N:3])-{NU}",
        compatible_lg_tags={"halide", "sulfonate"},
        compatible_nu_tags={
            "N_nuc",
            "O_nuc",
            "S_nuc",
            "azide",
            "cyanide",
            "carboxylate",
            "heterocyclic_N",
            "enolate",
            "halide_nuc",
            "C_nuc",
        },
    ),
    ElectrophilicCenter(
        name="sp3 alpha to EWG",
        reactant_smarts="[C;H1;D3;+0:1](-[S:2]=[O,S,N:3])-{LG}",
        product_smarts="[C;+0:1](-[S:2]=[O,S,N:3])-{NU}",
        compatible_lg_tags={"halide", "sulfonate"},
        compatible_nu_tags={
            "N_nuc",
            "O_nuc",
            "S_nuc",
            "azide",
            "cyanide",
            "carboxylate",
            "heterocyclic_N",
            "enolate",
            "halide_nuc",
            "C_nuc",
        },
    ),
    ElectrophilicCenter(
        name="sp3 alpha to EWG",
        reactant_smarts="[C;H1;D3;+0:1](-[N:2]=[O,S,N:3])-{LG}",
        product_smarts="[C;+0:1](-[N:2]=[O,S,N:3])-{NU}",
        compatible_lg_tags={"halide", "sulfonate"},
        compatible_nu_tags={
            "N_nuc",
            "O_nuc",
            "S_nuc",
            "azide",
            "cyanide",
            "carboxylate",
            "heterocyclic_N",
            "enolate",
            "halide_nuc",
            "C_nuc",
        },
    ),
    ElectrophilicCenter(
        name="sp3 alpha to EWG",
        reactant_smarts="[C;H2;D2;+0:1](-[C:2]=[O,S,N:3])-{LG}",
        product_smarts="[C;+0:1](-[C:2]=[O,S,N:3])-{NU}",
        compatible_lg_tags={"halide", "sulfonate"},
        compatible_nu_tags={
            "N_nuc",
            "O_nuc",
            "S_nuc",
            "azide",
            "cyanide",
            "carboxylate",
            "heterocyclic_N",
            "enolate",
            "halide_nuc",
            "C_nuc",
        },
    ),
    ElectrophilicCenter(
        name="sp3 alpha to EWG",
        reactant_smarts="[C;H2;D2;+0:1](-[S:2]=[O,S,N:3])-{LG}",
        product_smarts="[C;+0:1](-[S:2]=[O,S,N:3])-{NU}",
        compatible_lg_tags={"halide", "sulfonate"},
        compatible_nu_tags={
            "N_nuc",
            "O_nuc",
            "S_nuc",
            "azide",
            "cyanide",
            "carboxylate",
            "heterocyclic_N",
            "enolate",
            "halide_nuc",
            "C_nuc",
        },
    ),
    ElectrophilicCenter(
        name="sp3 alpha to EWG",
        reactant_smarts="[C;H2;D2;+0:1](-[N:2]=[O,S,N:3])-{LG}",
        product_smarts="[C;+0:1](-[N:2]=[O,S,N:3])-{NU}",
        compatible_lg_tags={"halide", "sulfonate"},
        compatible_nu_tags={
            "N_nuc",
            "O_nuc",
            "S_nuc",
            "azide",
            "cyanide",
            "carboxylate",
            "heterocyclic_N",
            "enolate",
            "halide_nuc",
            "C_nuc",
        },
    ),
    # Sulfonyl center (R-SO2-X) — sulfonamide / sulfonate ester synthesis
    ElectrophilicCenter(
        name="sulfonyl",
        reactant_smarts="[S;H0;D3;+0:1](=[O;H0;D1;+0:2])(=[O;H0;D1;+0:3])-{LG}",
        product_smarts="[S;H0;D4;+0:1](=[O;H0;D1;+0:2])(=[O;H0;D1;+0:3])-{NU}",
        compatible_lg_tags={"sulfonyl_halide"},
        compatible_nu_tags={
            "N_nuc",
            "O_nuc",
            "S_nuc",
            "heterocyclic_N",
        },
    ),
    # Chloroformate / carbonate — carbamate and carbonate synthesis
    ElectrophilicCenter(
        name="chloroformate/carbonate",
        reactant_smarts="[C;H0;D3;+0:1](=[O;H0;D1;+0:2])(-[O;H0;D2;+0:3]-[#6:4])-{LG}",
        product_smarts="[C;H0;D3;+0:1](=[O;H0;D1;+0:2])(-[O;H0;D2;+0:3]-[#6:4])-{NU}",
        compatible_lg_tags={"halide"},
        compatible_nu_tags={
            "N_nuc",
            "O_nuc",
            "heterocyclic_N",
        },
    ),
    # Aziridine ring-opening (special — LG is part of the ring)
    ElectrophilicCenter(
        name="aziridine",
        reactant_smarts="[C;+0:1]1-[N;H1;D2;+0:2]-[C;+0:3]1",
        product_smarts="[C;+0:1](-[N;+0:2])-[C;+0:3]-{NU}",
        compatible_lg_tags={"aziridine"},
        compatible_nu_tags={
            "N_nuc",
            "O_nuc",
            "S_nuc",
            "azide",
            "cyanide",
            "heterocyclic_N",
        },
    ),
]

# --- Leaving groups -------------------------------------------------------

LEAVING_GROUPS: List[LeavingGroup] = [
    LeavingGroup("fluoride", "[F]", {"halide"}),
    LeavingGroup("chloride", "[Cl]", {"halide"}),
    LeavingGroup("bromide", "[Br]", {"halide"}),
    LeavingGroup("iodide", "[I]", {"halide"}),
    LeavingGroup("mesylate", "[O]S(=O)(=O)[CH3]", {"sulfonate"}),
    LeavingGroup("tosylate", "[O]S(=O)(=O)c1ccc(C)cc1", {"sulfonate"}),
    LeavingGroup("triflate", "[O]S(=O)(=O)C(F)(F)F", {"sulfonate"}),
    LeavingGroup("nosylate", "[O]S(=O)(=O)c1ccc([N+](=O)[O-])cc1", {"sulfonate"}),
    LeavingGroup("brosylate", "[O]S(=O)(=O)c1ccc(Br)cc1", {"sulfonate"}),
    LeavingGroup("generic sulfonate", "[O]S(=O)(=O)-[*]", {"sulfonate"}),
    LeavingGroup("acetate (acyloxy)", "[O]C(=O)C", {"acyloxy"}),
    LeavingGroup("carboxylate (acyloxy)", "[O]C(=O)-[*]", {"acyloxy"}),
    LeavingGroup("NHS ester", "[O]N1C(=O)CCC1=O", {"activated_ester"}),
    LeavingGroup("PFP ester", "[O]c1c(F)c(F)c(F)c(F)c1F", {"activated_ester"}),
    LeavingGroup("HOBt ester", "[O]n1nnc2ccccc21", {"activated_ester"}),
    LeavingGroup("phosphonate", "[O]P(=O)([O,#6])[O,#6]", {"phosphonate"}),
    LeavingGroup("nitro (SNAr)", "[N+](=O)[O-]", {"nitro_lg"}),
    LeavingGroup("epoxide", "", {"epoxide"}),
    LeavingGroup("aziridine", "", {"aziridine"}),
    LeavingGroup("sulfonyl chloride", "[Cl]", {"sulfonyl_halide", "halide"}),
    LeavingGroup("sulfonyl fluoride", "[F]", {"sulfonyl_halide", "halide"}),
    LeavingGroup("anhydride (symmetric)", "[O]C(=O)-[*]", {"anhydride", "acyloxy"}),
    LeavingGroup("mixed anhydride", "[O]C(=O)[O,N]-[*]", {"anhydride", "acyloxy"}),
    LeavingGroup("CDI imidazolide", "[n]1ccnc1", {"activated_ester", "cdi"}),
    LeavingGroup("alkyl carbonate", "[O]C(=O)O-[*]", {"acyloxy"}),
    LeavingGroup("diazonium (SNAr)", "[N+]#[N]", {"diazonium", "nitro_lg"}),
]

# --- Nucleophiles ---------------------------------------------------------

NUCLEOPHILES: List[Nucleophile] = [
    # ---- N nucleophiles ----
    Nucleophile(
        "primary amine (R-NH2)",
        "[*:11]-[N;H2;D1;+0:10]",
        "[N;H0,H1;+0:10]-[*:11]",
        {"N_nuc"},
    ),
    Nucleophile(
        "secondary amine (R2-NH)",
        "[*:11]-[N;H1;D2;+0:10]-[*:12]",
        "[N;H0;+0:10](-[*:11])-[*:12]",
        {"N_nuc"},
    ),
    Nucleophile(
        "aniline (Ar-NH2)",
        "[c:11]-[N;H2;D1;+0:10]",
        "[N;H0,H1;+0:10]-[c:11]",
        {"N_nuc"},
    ),
    Nucleophile(
        "NH3 / ammonia",
        "[N;H3;D0;+0:10]",
        "[N;H2;D1;+0:10]",
        {"N_nuc"},
    ),
    Nucleophile(
        "amide N (R-C(=O)-NH)",
        "[*:12]-[C;H0;D3;+0:11](=[O;H0;D1;+0:13])-[N;H1;D2;+0:10]",
        "[N;H0;D3;+0:10]-[C;H0;D3;+0:11](=[O;H0;D1;+0:13])-[*:12]",
        {"N_nuc"},
    ),
    Nucleophile(
        "sulfonamide NH",
        "[*:11]-[S:12](=[O:13])(=[O:14])-[N;H1;D2;+0:10]",
        "[N;H0;D3;+0:10]-[S:12](=[O:13])(=[O:14])-[*:11]",
        {"N_nuc"},
    ),
    # ---- Heterocyclic N nucleophiles ----
    Nucleophile(
        "imidazole N",
        "[n;H1;+0:10]1:[c:11]:[n:12]:[c:13]:[c:14]:1",
        "[n;H0;+0:10]1:[c:11]:[n:12]:[c:13]:[c:14]:1",
        {"heterocyclic_N", "N_nuc"},
    ),
    Nucleophile(
        "1,2,3-triazole NH",
        "[n;H1;+0:10]1:[n:11]:[n:12]:[c:13]:[c:14]:1",
        "[n;H0;+0:10]1:[n:11]:[n:12]:[c:13]:[c:14]:1",
        {"heterocyclic_N", "N_nuc"},
    ),
    Nucleophile(
        "1,2,4-triazole NH",
        "[n;H1;+0:10]1:[c:11]:[n:12]:[n:13]:[c:14]:1",
        "[n;H0;+0:10]1:[c:11]:[n:12]:[n:13]:[c:14]:1",
        {"heterocyclic_N", "N_nuc"},
    ),
    Nucleophile(
        "tetrazole NH",
        "[n;H1;+0:10]1:[n:11]:[n:12]:[n:13]:[c:14]:1",
        "[n;H0;+0:10]1:[n:11]:[n:12]:[n:13]:[c:14]:1",
        {"heterocyclic_N", "N_nuc"},
    ),
    Nucleophile(
        "pyrazole NH",
        "[n;H1;+0:10]1:[c:11]:[c:12]:[c:13]:[n:14]:1",
        "[n;H0;+0:10]1:[c:11]:[c:12]:[c:13]:[n:14]:1",
        {"heterocyclic_N", "N_nuc"},
    ),
    Nucleophile(
        "indole NH",
        "[n;H1;+0:10]1:[c:11]:[c:12]:[c:13]2:[c:14]:[c:15]:[c:16]:[c:17]:[c:18]:1:2",
        "[n;H0;+0:10]1:[c:11]:[c:12]:[c:13]2:[c:14]:[c:15]:[c:16]:[c:17]:[c:18]:1:2",
        {"heterocyclic_N", "N_nuc"},
    ),
    Nucleophile(
        "benzimidazole NH",
        "[n;H1;+0:10]1:[c:11]:[n:12]:[c:13]2:[c:14]:[c:15]:[c:16]:[c:17]:[c:18]:1:2",
        "[n;H0;+0:10]1:[c:11]:[n:12]:[c:13]2:[c:14]:[c:15]:[c:16]:[c:17]:[c:18]:1:2",
        {"heterocyclic_N", "N_nuc"},
    ),
    # ---- O nucleophiles ----
    Nucleophile(
        "alcohol (R-OH)",
        "[#6;+0:11]-[O;H1;D1;+0:10]",
        "[O;H0;D2;+0:10]-[#6;+0:11]",
        {"O_nuc"},
    ),
    Nucleophile(
        "phenol (Ar-OH)",
        "[c;+0:11]-[O;H1;D1;+0:10]",
        "[O;H0;D2;+0:10]-[c;+0:11]",
        {"O_nuc"},
    ),
    Nucleophile(
        "carboxylate (R-COO⁻)",
        "[#6;+0:12]-[C;H0;D3;+0:11](=[O;H0;D1;+0:13])-[O;H0;D1;-1:10]",
        "[O;H0;D2;+0:10]-[C;H0;D3;+0:11](=[O;H0;D1;+0:13])-[#6;+0:12]",
        {"carboxylate", "O_nuc"},
    ),
    Nucleophile(
        "carboxylic acid (R-COOH)",
        "[#6;+0:12]-[C;H0;D3;+0:11](=[O;H0;D1;+0:13])-[O;H1;D1;+0:10]",
        "[O;H0;D2;+0:10]-[C;H0;D3;+0:11](=[O;H0;D1;+0:13])-[#6;+0:12]",
        {"carboxylate", "O_nuc"},
    ),
    # ---- S nucleophiles ----
    Nucleophile(
        "thiol (R-SH)",
        "[#6;+0:11]-[S;H1;D1;+0:10]",
        "[S;H0;D2;+0:10]-[#6;+0:11]",
        {"S_nuc"},
    ),
    Nucleophile(
        "thiolate (R-S⁻)",
        "[#6;+0:11]-[S;H0;D1;-1:10]",
        "[S;H0;D2;+0:10]-[#6;+0:11]",
        {"S_nuc"},
    ),
    Nucleophile(
        "thiophenol (Ar-SH)",
        "[c;+0:11]-[S;H1;D1;+0:10]",
        "[S;H0;D2;+0:10]-[c;+0:11]",
        {"S_nuc"},
    ),
    # ---- Other nucleophiles ----
    Nucleophile(
        "azide (NaN3)",
        "[Na,Li]-[N;H0;D2;+0:10]=[N;H0;D2;+1:11]=[N;H0;D1;-1:12]",
        "[N;H0;D2;+0:10]=[N;H0;D2;+1:11]=[N;H0;D1;-1:12]",
        {"azide"},
    ),
    Nucleophile(
        "azide anion (N3⁻)",
        "[N;H0;D1;-1:10]=[N;H0;D2;+1:11]=[N;H0;D1;-1:12]",
        "[N;H0;D2;+0:10]=[N;H0;D2;+1:11]=[N;H0;D1;-1:12]",
        {"azide"},
    ),
    Nucleophile(
        "cyanide (NaCN)",
        "[Na,Li]-[C;H0;D2;-1:10]#[N;H0;D1;+0:11]",
        "[C;H0;D2;+0:10]#[N;H0;D1;+0:11]",
        {"cyanide"},
    ),
    Nucleophile(
        "cyanide anion (CN⁻)",
        "[C;H0;D1;-1:10]#[N;H0;D1;+0:11]",
        "[C;H0;D2;+0:10]#[N;H0;D1;+0:11]",
        {"cyanide"},
    ),
    Nucleophile(
        "fluoride (for Finkelstein / halide exchange)",
        "[F;H0;D0;-1:10]",
        "[F;H0;D1;+0:10]",
        {"halide_nuc"},
    ),
    Nucleophile(
        "chloride (halide exchange)",
        "[Cl;H0;D0;-1:10]",
        "[Cl;H0;D1;+0:10]",
        {"halide_nuc"},
    ),
    Nucleophile(
        "bromide (halide exchange)",
        "[Br;H0;D0;-1:10]",
        "[Br;H0;D1;+0:10]",
        {"halide_nuc"},
    ),
    Nucleophile(
        "iodide (halide exchange)",
        "[I;H0;D0;-1:10]",
        "[I;H0;D1;+0:10]",
        {"halide_nuc"},
    ),
    # ---- O nucleophiles (additional) ----
    Nucleophile(
        "alkoxide (R-O⁻)",
        "[#6;+0:11]-[O;H0;D1;-1:10]",
        "[O;H0;D2;+0:10]-[#6;+0:11]",
        {"O_nuc"},
    ),
    Nucleophile(
        "hydroxylamine (H2N-OH)",
        "[N;H2;D1;+0:11]-[O;H1;D1;+0:10]",
        "[O;H0;D2;+0:10]-[N;H2;D1;+0:11]",
        {"O_nuc"},
    ),
    Nucleophile(
        "hydroxylamine N-attack (H2N-OH)",
        "[O;H1;D1;+0:11]-[N;H2;D1;+0:10]",
        "[N;H0,H1;+0:10](-[O;H0,H1;+0:11])",
        {"N_nuc"},
    ),
    # ---- N nucleophiles (additional) ----
    Nucleophile(
        "hydrazine (H2N-NH2)",
        "[N;H2;D1;+0:11]-[N;H2;D1;+0:10]",
        "[N;H0,H1;+0:10]-[N;H2;D1;+0:11]",
        {"N_nuc"},
    ),
    Nucleophile(
        "hydrazide (R-C(=O)-NH-NH2)",
        "[*:12]-[C;H0;D3;+0:11](=[O;H0;D1;+0:13])-[N;H1;D2;+0:14]-[N;H2;D1;+0:10]",
        "[N;H0,H1;+0:10]-[N;H1;D2;+0:14]-[C;H0;D3;+0:11](=[O;H0;D1;+0:13])-[*:12]",
        {"N_nuc"},
    ),
    Nucleophile(
        "phthalimide anion (Gabriel)",
        "[n;H0;-1:10]1[c:11](=[O:19])[c:12][c:13][c:14]2[c:15][c:16][c:17][c:18]12",
        "[n;H0;+0:10]1[c:11](=[O:19])[c:12][c:13][c:14]2[c:15][c:16][c:17][c:18]12",
        {"N_nuc"},
    ),
    # ---- C nucleophiles ----
    Nucleophile(
        "active methylene / malonate enolate",
        "[#6:12]-[C;H1,H2;+0:10](-[C:13]=[O:14])-[C:16]=[O,S,N:15]",
        "[C;H0,H1;+0:10](-[#6:12])(-[C:13]=[O:14])-[C:16]=[O,S,N:15]",
        {"enolate", "C_nuc"},
    ),
    Nucleophile(
        "active methylene / malonate enolate",
        "[#6:12]-[C;H1,H2;+0:10](-[C:13]=[S:14])-[C:16]=[O,S,N:15]",
        "[C;H0,H1;+0:10](-[#6:12])(-[C:13]=[S:14])-[C:16]=[O,S,N:15]",
        {"enolate", "C_nuc"},
    ),
    Nucleophile(
        "active methylene / malonate enolate",
        "[#6:12]-[C;H1,H2;+0:10](-[C:13]=[N:14])-[C:16]=[O,S,N:15]",
        "[C;H0,H1;+0:10](-[#6:12])(-[C:13]=[N:14])-[C:16]=[O,S,N:15]",
        {"enolate", "C_nuc"},
    ),
    Nucleophile(
        "acetylide anion (R-C≡C⁻)",
        "[#6:11]-[C;H0;D2;+0:12]#[C;H0;D1;-1:10]",
        "[C;H0;D2;+0:10]#[C;H0;D2;+0:12]-[#6:11]",
        {"C_nuc"},
    ),
    Nucleophile(
        "terminal acetylide (RC≡CH, deprotonated)",
        "[C;H0;D1;-1:10]#[C;H1:11]",
        "[C;H0;D2;+0:10]#[C;H1:11]",
        {"C_nuc"},
    ),
    # ---- P nucleophiles ----
    Nucleophile(
        "phosphonate anion (Arbuzov / HWE)",
        "[#6:12]-[P;H0;D4;+0:10](=[O;H0;D1;+0:13])(-[O;H0;D2;+0:14]-[#6:15])-[C;H1,H2:11]",
        "[P;H0;D4;+0:10](=[O;H0;D1;+0:13])(-[O;H0;D2;+0:14]-[#6:15])(-[#6:12])-[C;H0,H1:11]",
        {"P_nuc"},
    ),
    Nucleophile(
        "trialkylphosphine (Mitsunobu / SN2)",
        "[P;H0;D3;+0:10](-[#6:11])(-[#6:12])-[#6:13]",
        "[P;H0;D4;+1:10](-[#6:11])(-[#6:12])-[#6:13]",
        {"P_nuc"},
    ),
]


# ---------------------------------------------------------------------------
# 3.  Assembly & validation
# ---------------------------------------------------------------------------


def assemble_smirks(
    center: ElectrophilicCenter,
    lg: LeavingGroup,
    nu: Nucleophile,
) -> Optional[str]:
    """
    Build a SMIRKS string from an electrophilic center, leaving group,
    and nucleophile by plugging fragments into the template placeholders.

    Args:
        center (ElectrophilicCenter): The electrophilic center template.
        lg (LeavingGroup): The leaving group fragment.
        nu (Nucleophile): The nucleophile definition.

    Returns:
        Optional[str]: A complete SMIRKS string, or None if the combination
            is incompatible or assembly fails.
    """
    # Tag compatibility check
    if not (center.compatible_lg_tags & lg.tags):
        return None
    if not (center.compatible_nu_tags & nu.tags):
        return None

    # Epoxide is a special case — the LG is structural, not a fragment
    if "epoxide" in lg.tags:
        if center.name != "epoxide":
            return None
        reactants = f"{center.reactant_smarts}.{nu.reactant_smarts}"
        products = center.product_smarts.replace("{NU}", nu.product_smarts)
        return f"{reactants}>>{products}"

    # Aziridine is handled the same way as epoxide
    if "aziridine" in lg.tags:
        if center.name != "aziridine":
            return None
        reactants = f"{center.reactant_smarts}.{nu.reactant_smarts}"
        products = center.product_smarts.replace("{NU}", nu.product_smarts)
        return f"{reactants}>>{products}"

    # Normal assembly
    reactant_center = center.reactant_smarts.replace("{LG}", lg.smarts)
    product_center = center.product_smarts.replace("{NU}", nu.product_smarts)

    reactants = f"{reactant_center}.{nu.reactant_smarts}"
    products = product_center

    return f"{reactants}>>{products}"


def validate_smirks(smirks: str) -> bool:
    """
    Validate a SMIRKS string by checking that:
    1. Both sides parse as valid SMARTS.
    2. rdchiral can initialize the retro-template.
    3. Atom maps are consistent (no transmutation, no orphans).

    Args:
        smirks (str): A reaction SMIRKS ``reactants>>products``.

    Returns:
        bool: True if the SMIRKS passes all checks.
    """
    parts = smirks.split(">>")
    if len(parts) != 2:
        return False

    reactant_str, product_str = parts

    # Parse each fragment
    for frag in reactant_str.split("."):
        if Chem.MolFromSmarts(frag) is None:
            return False
    for frag in product_str.split("."):
        if Chem.MolFromSmarts(frag) is None:
            return False

    # Collect atom-map → element on each side
    def _map_elements(smarts_str: str) -> Dict[int, str]:
        result = {}
        for frag in smarts_str.split("."):
            mol = Chem.MolFromSmarts(frag)
            if mol is None:
                continue
            for atom in mol.GetAtoms():
                m = atom.GetAtomMapNum()
                if m != 0:
                    result[m] = atom.GetSymbol()
        return result

    r_maps = _map_elements(reactant_str)
    p_maps = _map_elements(product_str)

    # Every mapped atom in products must also appear in reactants (and vice-versa)
    if set(r_maps.keys()) != set(p_maps.keys()):
        return False

    # No element transmutation
    for k in r_maps:
        # Allow wildcard '*' matching anything
        if r_maps[k] != p_maps[k] and r_maps[k] != "*" and p_maps[k] != "*":
            return False

    # rdchiral round-trip (retro direction)
    retro = f"{product_str}>>{reactant_str}"
    try:
        rdc.rdchiralReaction(retro)
    except Exception:
        return False

    return True


def generate_name(
    center: ElectrophilicCenter,
    lg: LeavingGroup,
    nu: Nucleophile,
) -> str:
    """
    Generate a human-readable name for the reaction pattern.

    Args:
        center (ElectrophilicCenter): The electrophilic center.
        lg (LeavingGroup): The leaving group.
        nu (Nucleophile): The nucleophile.

    Returns:
        str: A descriptive reaction name.
    """
    return f"Nucleophilic substitution {nu.name} at {center.name} ({lg.name} leaving group)"


RXNO_TERMS: Dict[str, Dict[str, str]] = {
    "RXNO:0000332": {
        "label": "aromatic substitution step",
        "definition": "A substitution step where one singly-bonded substituent on an aromatic skeleton is replaced by another singly-bonded substituent.",
    },
    "RXNO:0000357": {
        "label": "N-acylation to amide",
        "definition": "An acylation reaction where a nitrogen atom is acylated to form an amide.",
    },
    "RXNO:0000360": {
        "label": "O-acylation to ester",
        "definition": "An O-acylation reaction where an oxygen centre is acylated to result in an ester.",
    },
    "RXNO:0000359": {
        "label": "N-acylation to carbamate",
        "definition": "An N-acylation reaction where a nitrogen centre is acylated to form a carbamate.",
    },
    "RXNO:0000155": {
        "label": "Finkelstein reaction",
        "definition": "The conversion of an alkyl chloride, alkyl bromide or alkyl sulfonate ester to an alkyl iodide by SN2 substitution. The reaction relies upon the equilibrium being pushed to completion by the precipitation.",
    },
    "RXNO:0000060": {
        "label": "Arbuzov reaction",
        "definition": "The alkylation of a trialkyl phosphite with an alkyl halide or acyl halide to give an alkyl phosphonate.",
    },
    "RXNO:0000103": {
        "label": "Gabriel synthesis",
        "definition": "The reaction of primary alkyl halides with sodium or potassium phthalimide, followed by hydrolysis, to give the corresponding primary amine.",
    },
    "RXNO:0000090": {
        "label": "Williamson ether synthesis",
        "definition": "The reaction between an alkyl halide or alkyl sulfate and a metal alkoxide to give an ether.",
    },
    "RXNO:0000341": {
        "label": "aniline N-alkylation",
        "definition": "An N-alkylation where the reactive centre is an aniline nitrogen.",
    },
    "RXNO:0000340": {
        "label": "amide N-alkylation",
        "definition": "An N-alkylation where the reactive centre is an amide nitrogen.",
    },
    "RXNO:0000345": {
        "label": "heteroaryl N-alkylation",
        "definition": "An N-alkylation reaction where the reactive centre is an azacycle ring nitrogen.",
    },
    "RXNO:0000331": {
        "label": "substitution step",
        "definition": "A functional modification step in which one singly-bonded substituent, but not a hydrogen, is replaced by another singly-bonded substituent.",
    },
}


def classify_rxno(
    center: ElectrophilicCenter,
    lg: LeavingGroup,
    nu: Nucleophile,
) -> Dict[str, str]:
    """
    Assign the most specific RXNO ontology term for a nucleophilic substitution
    pattern based on the electrophilic center, leaving group, and nucleophile.

    The decision tree maps directly onto the metadata already encoded in
    ``center.name``, ``nu.tags``, and ``lg.tags``, requiring no additional
    SMARTS parsing.  When no specific term applies the function falls back to
    ``RXNO:0000331`` (substitution step), which is the correct RXNO parent for
    SN2, epoxide/aziridine ring-opening, and sulfonylation reactions that lack
    dedicated ontology entries.

    Args:
        center (ElectrophilicCenter): The electrophilic center template.
        lg (LeavingGroup): The leaving group fragment.
        nu (Nucleophile): The nucleophile definition.

    Returns:
        Dict[str, str]: A dict with keys ``rxno_id``, ``rxno_label``, and
            ``rxno_definition`` for the most specific applicable RXNO term.
    """
    # SNAr — aromatic substitution step
    if center.name == "SNAr aromatic":
        rxno_id = "RXNO:0000332"

    # Nucleophilic acyl substitution
    elif center.name == "acyl (carbonyl)":
        if "N_nuc" in nu.tags:
            rxno_id = "RXNO:0000357"  # N-acylation to amide
        elif "O_nuc" in nu.tags:
            rxno_id = "RXNO:0000360"  # O-acylation to ester
        else:
            rxno_id = "RXNO:0000331"

    # Chloroformate/carbonate — carbamate and carbonate synthesis
    elif center.name == "chloroformate/carbonate":
        if "N_nuc" in nu.tags:
            rxno_id = "RXNO:0000359"  # N-acylation to carbamate
        elif "O_nuc" in nu.tags:
            rxno_id = "RXNO:0000360"  # O-acylation to ester
        else:
            rxno_id = "RXNO:0000331"

    # Finkelstein: halide ↔ halide exchange
    elif "halide_nuc" in nu.tags and "halide" in lg.tags:
        rxno_id = "RXNO:0000155"

    # Arbuzov: P nucleophile + alkyl halide or sulfonate
    elif "P_nuc" in nu.tags:
        rxno_id = "RXNO:0000060"

    # Gabriel synthesis: phthalimide anion
    elif "phthalimide" in nu.name:
        rxno_id = "RXNO:0000103"

    # Williamson ether synthesis: O nucleophile + alkyl halide/sulfonate
    elif "O_nuc" in nu.tags and ("halide" in lg.tags or "sulfonate" in lg.tags):
        rxno_id = "RXNO:0000090"

    # N-alkylation specifics at sp3/benzylic/allylic/propargylic centres
    elif "N_nuc" in nu.tags:
        if "aniline" in nu.name:
            rxno_id = "RXNO:0000341"  # aniline N-alkylation
        elif "amide" in nu.name or "sulfonamide" in nu.name:
            rxno_id = "RXNO:0000340"  # amide N-alkylation
        elif "heterocyclic_N" in nu.tags:
            rxno_id = "RXNO:0000345"  # heteroaryl N-alkylation
        else:
            rxno_id = "RXNO:0000331"

    # Generic fallback: sp3 SN2, epoxide/aziridine opening, sulfonylation, etc.
    else:
        rxno_id = "RXNO:0000331"

    term = RXNO_TERMS[rxno_id]
    return {
        "rxno_id": rxno_id,
        "rxno_label": term["label"],
        "rxno_definition": term["definition"],
    }


CLASS_TREE_PATH = Path(__file__).parent / "data" / "reaction_classes.json"

# Nucleophiles the taxonomy resolves more precisely than their coarse tags allow.
AZOLE_NUCLEOPHILES = frozenset(
    {
        "imidazole N",
        "1,2,3-triazole NH",
        "1,2,4-triazole NH",
        "tetrazole NH",
        "pyrazole NH",
        "indole NH",
        "benzimidazole NH",
    }
)
PHTHALIMIDE_NUCLEOPHILE = "phthalimide anion (Gabriel)"
PHENOL_NUCLEOPHILE = "phenol (Ar-OH)"
HYDROXYLAMINE_O_NUCLEOPHILE = "hydroxylamine (H2N-OH)"
PHOSPHINE_NUCLEOPHILE = "trialkylphosphine (Mitsunobu / SN2)"
PHOSPHONATE_NUCLEOPHILE = "phosphonate anion (Arbuzov / HWE)"


@lru_cache(maxsize=1)
def _class_index() -> Dict[Tuple[int, ...], str]:
    """
    Index the authoritative reaction-class tree by integer address.

    Returns:
        Dict[Tuple[int, ...], str]: Mapping from an address tuple such as
            ``(1, 3, 1, 1)`` to that node's uuid. Every depth is present, so a
            superclass address like ``(1,)`` is a valid key.

    Note:
        Reads ``data/reaction_classes.json`` once per process and caches the result.
    """
    tree = json.loads(CLASS_TREE_PATH.read_text(encoding="utf-8"))
    child_keys = ("classes", "subclasses", "subsubclasses")
    index: Dict[Tuple[int, ...], str] = {}

    def visit(node: Dict, address: Tuple[int, ...], depth: int) -> None:
        index[address] = node["uuid"]
        if depth < len(child_keys):
            for child in node.get(child_keys[depth], []):
                visit(child, address + (child["id"],), depth + 1)

    for superclass in tree.get("superclasses", []):
        visit(superclass, (superclass["id"],), 0)
    return index


def classify_reaction_class(
    center: ElectrophilicCenter,
    lg: LeavingGroup,
    nu: Nucleophile,
) -> Tuple[int, ...]:
    """
    Assign the reaction-class address for a nucleophilic substitution pattern.

    Dispatches on the electrophilic center first, since the center decides the
    superclass, then refines on nucleophile identity and leaving group. Returns the
    deepest address the three building blocks actually determine: an alkyl halide
    plus an alcohol is a Williamson ether synthesis at depth four, whereas a
    thiol only resolves to the SN2 S-alkylation subclass at depth three.

    Reserved ``0`` (Unspecified) children are used where the taxonomy has a correct
    parent but no child matching this combination, so no pattern is asserted to be
    more specific than the inputs support.

    Args:
        center (ElectrophilicCenter): The electrophilic center template.
        lg (LeavingGroup): The leaving group fragment.
        nu (Nucleophile): The nucleophile definition.

    Returns:
        Tuple[int, ...]: Address into ``reaction_classes.json`` as a tuple of one to
            four ids, ordered superclass, class, subclass, subsubclass.

    Note:
        A few combinations have no home in the current taxonomy and deliberately
        resolve to an Unspecified node: halogen exchange (Finkelstein) has no
        subclass under Halide interconversion and displacement, and epoxide opening
        by carbon nucleophiles has no subclass under C-C Bond Forming Reactions.
    """
    is_n_nuc = "N_nuc" in nu.tags
    is_o_nuc = "O_nuc" in nu.tags
    is_s_nuc = "S_nuc" in nu.tags
    is_azole = nu.name in AZOLE_NUCLEOPHILES
    is_carboxylate = "carboxylate" in nu.tags
    is_halide_lg = "halide" in lg.tags
    is_sulfonate_lg = "sulfonate" in lg.tags

    # --- Acyl transfer: superclass 2, Acylation and Related Processes ---------
    if center.name == "acyl (carbonyl)":
        # A carboxylate attacking an acyl halide gives an anhydride, not an ester.
        if is_carboxylate:
            return (2, 8, 2)  # Mixed anhydride formation
        if is_azole:
            return (2, 1, 10)  # Imide and N-acyl heterocycle formation
        if is_n_nuc:
            if is_halide_lg:
                return (2, 1, 2)  # Acyl halide aminolysis
            if "anhydride" in lg.tags or "acyloxy" in lg.tags:
                return (2, 1, 3)  # Anhydride aminolysis
            if "activated_ester" in lg.tags:
                return (2, 1, 4, 1)  # Activated ester aminolysis
            return (2, 1, 0)
        if is_o_nuc:
            if is_halide_lg:
                return (2, 2, 3)  # Acyl halide esterification
            if "anhydride" in lg.tags or "acyloxy" in lg.tags:
                return (2, 2, 4)  # Anhydride esterification
            if "activated_ester" in lg.tags:
                return (2, 2, 6)  # Transesterification
            return (2, 2, 0)
        if is_s_nuc:
            if is_halide_lg:
                return (2, 3, 1)  # Acyl halide thioesterification
            return (2, 3, 0)
        return (2, 0)

    if center.name == "chloroformate/carbonate":
        if is_n_nuc or is_azole:
            return (2, 4, 1)  # Chloroformate and carbonate carbamoylation
        if is_o_nuc:
            return (2, 7, 1)  # Chloroformate carbonate formation
        return (2, 0)

    if center.name == "sulfonyl":
        # SuFEx is defined by the S-F bond, so the leaving group outranks the nucleophile.
        if "sulfonyl_halide" in lg.tags and "fluoride" in lg.name:
            return (2, 11, 3)  # Sulfonyl fluoride and SuFEx chemistry
        if is_n_nuc or is_azole:
            return (2, 11, 1)  # Sulfonamide formation
        if is_o_nuc:
            return (2, 11, 2)  # Sulfonate ester formation
        return (2, 11, 0)

    # --- Aromatic substitution: superclass 1, arylation classes ---------------
    if center.name == "SNAr aromatic":
        if "azide" in nu.tags:
            return (1, 2, 3)  # SNAr with nitrogen nucleophiles
        if "cyanide" in nu.tags:
            return (5, 5, 3)  # Halide to nitrile
        if is_n_nuc or is_azole:
            if "diazonium" in lg.tags:
                return (1, 2, 5)  # Diazonium N-arylation
            return (1, 2, 3)  # SNAr with nitrogen nucleophiles
        if is_o_nuc:
            return (1, 4, 3)  # SNAr with oxygen nucleophiles
        if is_s_nuc:
            return (1, 6, 2)  # SNAr with sulfur nucleophiles
        return (1, 0)

    # --- Strained-ring opening: superclass 1, ring opening subclasses ---------
    if center.name in ("epoxide", "aziridine"):
        amine_opening = (1, 1, 4, 1) if center.name == "epoxide" else (1, 1, 4, 2)
        if "cyanide" in nu.tags:
            return (4, 0)  # No carbon-nucleophile ring-opening subclass exists.
        if "azide" in nu.tags or is_azole:
            return (1, 1, 4, 0)
        if is_n_nuc:
            return amine_opening
        if is_o_nuc:
            return (1, 3, 3)  # Ring opening O-alkylation
        if is_s_nuc:
            return (1, 5, 2)  # Ring opening S-alkylation
        return (1, 1, 4, 0)

    # --- sp3 centers: alkyl, benzylic, allylic, propargylic, alpha to EWG -----
    # Carbon and phosphorus nucleophiles leave superclass 1 entirely.
    if "enolate" in nu.tags:
        return (4, 2, 6, 1)  # Malonate and beta-ketoester alkylation
    if "C_nuc" in nu.tags:
        return (4, 2, 9)  # Alkylation of terminal alkynes
    if nu.name == PHOSPHONATE_NUCLEOPHILE:
        return (1, 7, 2)  # Phosphonate and phosphinate C-P bond formation
    if nu.name == PHOSPHINE_NUCLEOPHILE:
        return (1, 7, 1)  # Phosphine alkylation and phosphonium salt formation

    # Azide, cyanide and halide displacement are interconversions, not alkylations.
    if "azide" in nu.tags:
        return (5, 5, 2) if is_halide_lg else (5, 4, 2)
    if "cyanide" in nu.tags:
        return (5, 5, 3)  # Halide to nitrile
    if "halide_nuc" in nu.tags:
        return (5, 5, 0)  # No halogen-exchange subclass exists.

    if is_azole:
        return (1, 1, 1, 3)  # Azole N1 vs N2 alkylation
    if nu.name == PHTHALIMIDE_NUCLEOPHILE:
        return (1, 1, 1, 4)  # Gabriel and Delepine amine synthesis
    if is_n_nuc:
        if is_halide_lg:
            return (1, 1, 1, 1)  # Alkyl halide N-alkylation
        if is_sulfonate_lg:
            return (1, 1, 1, 2)  # Sulfonate ester N-alkylation
        return (1, 1, 1, 0)
    if is_o_nuc:
        if is_carboxylate:
            return (1, 3, 1, 3)  # Carboxylate O-alkylation
        if nu.name == PHENOL_NUCLEOPHILE:
            return (1, 3, 1, 2)  # Phenol O-alkylation
        if nu.name == HYDROXYLAMINE_O_NUCLEOPHILE:
            return (1, 3, 1, 0)
        if is_halide_lg or is_sulfonate_lg:
            return (1, 3, 1, 1)  # Williamson ether synthesis
        return (1, 3, 1, 0)
    if is_s_nuc:
        return (1, 5, 1)  # SN2 S-alkylation
    return (1, 0)


def classification_fields(
    center: ElectrophilicCenter,
    lg: LeavingGroup,
    nu: Nucleophile,
) -> Dict[str, Optional[object]]:
    """
    Build the classification record fields for one pattern.

    Args:
        center (ElectrophilicCenter): The electrophilic center template.
        lg (LeavingGroup): The leaving group fragment.
        nu (Nucleophile): The nucleophile definition.

    Returns:
        Dict[str, Optional[object]]: The four integer id fields, padded to depth four
            with ``None``, plus ``classification_uuid`` holding the uuid of the
            addressed node.

    Raises:
        KeyError: If the assigned address does not exist in ``reaction_classes.json``,
            which means the classifier and the taxonomy have drifted apart.
    """
    address = classify_reaction_class(center, lg, nu)
    index = _class_index()
    if address not in index:
        raise KeyError(
            f"classify_reaction_class returned {address} for"
            f" center={center.name!r} lg={lg.name!r} nu={nu.name!r},"
            f" which is not a node in {CLASS_TREE_PATH.name}"
        )

    ids: List[Optional[int]] = list(address) + [None] * (4 - len(address))
    return {
        "superclass_id": ids[0],
        "class_id": ids[1],
        "subclass_id": ids[2],
        "subsubclass_id": ids[3],
        "classification_uuid": index[address],
    }


# ---------------------------------------------------------------------------
# 4.  Main enumeration
# ---------------------------------------------------------------------------


def enumerate_nuc_sub_smirks() -> List[Dict]:
    """
    Enumerate all chemically plausible nucleophilic substitution SMIRKS
    patterns from the combinatorial space of centers × leaving groups ×
    nucleophiles, applying compatibility and validation filters.

    Returns:
        List[Dict]: A list of pattern dicts matching the schema used in
    """
    patterns: List[Dict[str, object]] = []
    seen_smirks: Set[str] = set()

    for center, lg, nu in itertools.product(
        ELECTROPHILIC_CENTERS, LEAVING_GROUPS, NUCLEOPHILES
    ):
        smirks = assemble_smirks(center, lg, nu)
        if smirks is None:
            continue
        if smirks in seen_smirks:
            continue

        if not validate_smirks(smirks):
            print(f"  INVALID — skipping: {generate_name(center, lg, nu)}")
            continue

        seen_smirks.add(smirks)
        patterns.append(
            {
                "rxno_classification": [classify_rxno(center, lg, nu)],
                "name": generate_name(center, lg, nu),
                "priority": {"priority_class": None, "priority": None},
                "smirks": smirks,
                "uuid": str(uuid.uuid4()),
                **classification_fields(center, lg, nu),
            }
        )

    return patterns


if __name__ == "__main__":
    patterns = enumerate_nuc_sub_smirks()
    print(f"\nGenerated {len(patterns)} valid nucleophilic substitution patterns.\n")

    # Preview first few
    for p in patterns[:5]:
        print(f"  {p['name']}")
        print(f"    {p['smirks']}\n")

    # Write to file
    outpath = "nuc_sub_reactions.json"
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(patterns, f, indent=4, ensure_ascii=False)
    print(f"Wrote {len(patterns)} patterns to {outpath}")
