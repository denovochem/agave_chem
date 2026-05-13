"""
generate_nuc_sub_smirks.py
--------------------------
Enumerate plausible nucleophilic-substitution SMIRKS patterns by combining
electrophilic centers, leaving groups, and nucleophiles with chemical
compatibility filtering.
"""

import itertools
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

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
        if r_maps[k] != p_maps[k]:
            # Allow wildcard '*' matching anything
            if r_maps[k] != "*" and p_maps[k] != "*":
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
    return f"NucSub {nu.name} at {center.name} ({lg.name} LG)"


def classify_rxno(
    center: ElectrophilicCenter,
    lg: LeavingGroup,
    nu: Nucleophile,
) -> str:
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
        str: An RXNO term identifier string (e.g. ``"RXNO:0000331"``),
            representing the most specific applicable classification.
    """
    # SNAr — aromatic substitution step
    if center.name == "SNAr aromatic":
        return "RXNO:0000332"

    # Nucleophilic acyl substitution
    if center.name == "acyl (carbonyl)":
        if "N_nuc" in nu.tags:
            return "RXNO:0000357"  # N-acylation to amide
        if "O_nuc" in nu.tags:
            return "RXNO:0000360"  # O-acylation to ester

    # Chloroformate/carbonate — carbamate and carbonate synthesis
    if center.name == "chloroformate/carbonate":
        if "N_nuc" in nu.tags:
            return "RXNO:0000359"  # N-acylation to carbamate
        if "O_nuc" in nu.tags:
            return "RXNO:0000360"  # O-acylation to ester

    # Finkelstein: halide ↔ halide exchange
    if "halide_nuc" in nu.tags and "halide" in lg.tags:
        return "RXNO:0000155"

    # Arbuzov: P nucleophile + alkyl halide or sulfonate
    if "P_nuc" in nu.tags:
        return "RXNO:0000060"

    # Gabriel synthesis: phthalimide anion
    if "phthalimide" in nu.name:
        return "RXNO:0000103"

    # Williamson ether synthesis: O nucleophile + alkyl halide/sulfonate
    if "O_nuc" in nu.tags and ("halide" in lg.tags or "sulfonate" in lg.tags):
        return "RXNO:0000090"

    # N-alkylation specifics at sp3/benzylic/allylic/propargylic centres
    if "N_nuc" in nu.tags:
        if "aniline" in nu.name:
            return "RXNO:0000341"  # aniline N-alkylation
        if "amide" in nu.name or "sulfonamide" in nu.name:
            return "RXNO:0000340"  # amide N-alkylation
        if "heterocyclic_N" in nu.tags:
            return "RXNO:0000345"  # heteroaryl N-alkylation

    # Generic fallback: sp3 SN2, epoxide/aziridine opening, sulfonylation, etc.
    return "RXNO:0000331"


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
                "rxno_classification": [{"rxno_id": classify_rxno(center, lg, nu)}],
                "name": generate_name(center, lg, nu),
                "priority": {"priority_class": None, "priority": None},
                "smirks": smirks,
                "subclass_id": None,
                "subsubclass_id": None,
                "superclass_id": 1,  # Heteroatom Alkylation and Arylation
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
