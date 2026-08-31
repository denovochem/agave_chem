"""Visualisation utilities for mapper comparison.

Extracts rdchiral_plus templates from mapped reactions, identifies
atoms with differing mappings between mappers, and renders reaction
images with per-atom highlighting.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
from rdchiral import extract_from_reaction_smiles
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D


@dataclass
class TemplateInfo:
    """Result of extracting a template from a mapped reaction.

    Attributes:
        smarts: Canonical reaction SMARTS string for the template.
        atom_map_nums: Set of atom-map numbers that appear in the template
            (i.e. atoms involved in bond changes).
        products: SMARTS for the template product side.
        reactants: SMARTS for the template reactant side.
        success: Whether template extraction succeeded.
    """

    smarts: str
    atom_map_nums: Set[int]
    products: str
    reactants: str
    success: bool


def extract_template(mapped_rxn: str) -> TemplateInfo:
    """Extract a retrosynthetic template from a mapped reaction SMILES.

    Uses rdchiral_plus ``extract_from_reaction_smiles`` and parses the
    resulting SMARTS to identify which atom-map numbers participate in
    the bond-changing region of the reaction.

    Args:
        mapped_rxn: Atom-mapped reaction SMILES string.

    Returns:
        TemplateInfo with the template SMARTS and changed atom-map numbers.
        If extraction fails, returns a TemplateInfo with success=False and
        empty fields.
    """
    if not mapped_rxn or not mapped_rxn.strip():
        return TemplateInfo("", set(), "", "", success=False)

    try:
        result = extract_from_reaction_smiles(mapped_rxn)
    except Exception:
        return TemplateInfo("", set(), "", "", success=False)

    smarts = result.get("reaction_smarts", "") or ""
    if not smarts:
        return TemplateInfo("", set(), "", "", success=False)

    parts = smarts.split(">>")
    products = parts[0] if len(parts) == 2 else ""
    reactants = parts[1] if len(parts) == 2 else ""

    map_nums: Set[int] = {int(m) for m in re.findall(r":(\d+)", smarts)}

    return TemplateInfo(
        smarts=smarts,
        atom_map_nums=map_nums,
        products=products,
        reactants=reactants,
        success=True,
    )


def templates_differ(info_a: TemplateInfo, info_b: TemplateInfo) -> bool:
    """Check whether two TemplateInfo objects represent different templates.

    Args:
        info_a: First template info.
        info_b: Second template info.

    Returns:
        True if the templates differ (or one succeeded and the other didn't),
        False if both failed or both produced identical SMARTS.
    """
    if not info_a.success and not info_b.success:
        return False
    if info_a.success != info_b.success:
        return True
    return info_a.smarts != info_b.smarts


# ---------------------------------------------------------------------------
# Atom-level diffing between two mapped reactions
# ---------------------------------------------------------------------------


NodeID = Tuple[str, int, int]


@dataclass
class DifferingAtoms:
    """Atoms whose mapping differs between two mapped reactions.

    Attributes:
        reactant_a: Set of (frag_i, atom_i) tuples in rxn_a's reactants
            that have a different mapping in rxn_b.
        product_a: Set of (frag_i, atom_i) tuples in rxn_a's products
            that have a different mapping in rxn_b.
        reactant_b: Set of (frag_i, atom_i) tuples in rxn_b's reactants
            that have a different mapping in rxn_a.
        product_b: Set of (frag_i, atom_i) tuples in rxn_b's products
            that have a different mapping in rxn_a.
    """

    reactant_a: Set[NodeID] = field(default_factory=set)
    product_a: Set[NodeID] = field(default_factory=set)
    reactant_b: Set[NodeID] = field(default_factory=set)
    product_b: Set[NodeID] = field(default_factory=set)

    @property
    def has_differences(self) -> bool:
        """True if any atoms differ between the two mappings."""
        return bool(
            self.reactant_a or self.product_a or self.reactant_b or self.product_b
        )


def _rxn_to_mapping_graph(rxn_smiles: str) -> nx.Graph:
    """Build a mapping graph from a mapped reaction SMILES.

    Replicates the logic from ``agave_chem.utils.graph_utils.rxn_to_mapping_graph``
    but is kept self-contained to avoid importing the full package.

    Args:
        rxn_smiles: Mapped reaction SMILES parsable by RDKit.

    Returns:
        NetworkX graph with bond edges (kind="bond") and mapping
        edges (kind="map") connecting atoms with the same atom-map number.
    """
    rxn = AllChem.ReactionFromSmarts(rxn_smiles, useSmiles=True)  # type: ignore[attr-defined]
    G = nx.Graph()

    def add_side(mols, side_label: str) -> None:
        for frag_i, mol in enumerate(mols):
            mol.UpdatePropertyCache(strict=False)
            Chem.FastFindRings(mol)
            Chem.SanitizeMol(
                mol,
                Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
                | Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION,
                catchErrors=True,
            )
            Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
            for atom in mol.GetAtoms():
                node_id: NodeID = (side_label, frag_i, atom.GetIdx())
                G.add_node(
                    node_id,
                    Z=atom.GetAtomicNum(),
                    side=side_label,
                    charge=atom.GetFormalCharge(),
                    aromatic=atom.GetIsAromatic(),
                    in_ring=atom.IsInRing(),
                    hydrogen_count=atom.GetTotalNumHs(),
                    degree=atom.GetDegree(),
                    chiral_tag=atom.GetProp("_CIPCode")
                    if atom.HasProp("_CIPCode")
                    else "",
                )
            for bond in mol.GetBonds():
                a: NodeID = (side_label, frag_i, bond.GetBeginAtomIdx())
                b: NodeID = (side_label, frag_i, bond.GetEndAtomIdx())
                G.add_edge(
                    a,
                    b,
                    kind="bond",
                    order=int(bond.GetBondTypeAsDouble()),
                    stereo=int(bond.GetStereo()),
                )

    r_mols = [rxn.GetReactantTemplate(i) for i in range(rxn.GetNumReactantTemplates())]
    p_mols = [rxn.GetProductTemplate(i) for i in range(rxn.GetNumProductTemplates())]

    add_side(r_mols, "R")
    add_side(p_mols, "P")

    p_map: Dict[int, NodeID] = {}
    product_map_nums: Set[int] = set()
    for frag_i, mol in enumerate(p_mols):
        for atom in mol.GetAtoms():
            m = atom.GetAtomMapNum()
            product_map_nums.add(m)
            if m:
                p_map[m] = ("P", frag_i, atom.GetIdx())

    r_map: Dict[int, NodeID] = {}
    for frag_i, mol in enumerate(r_mols):
        for atom in mol.GetAtoms():
            m = atom.GetAtomMapNum()
            if m not in product_map_nums:
                atom.SetAtomMapNum(0)
            if m:
                r_map[m] = ("R", frag_i, atom.GetIdx())

    for m, r_node in r_map.items():
        p_node = p_map.get(m)
        if p_node is not None:
            G.add_edge(r_node, p_node, kind="map")

    return G


def _node_match(a: Dict, b: Dict) -> bool:
    """Check if two graph nodes are equivalent for isomorphism testing."""
    return (
        a["Z"] == b["Z"]
        and a["side"] == b["side"]
        and a["charge"] == b["charge"]
        and a["aromatic"] == b["aromatic"]
        and a["in_ring"] == b["in_ring"]
        and a["hydrogen_count"] == b["hydrogen_count"]
        and a["degree"] == b["degree"]
        and a["chiral_tag"] == b["chiral_tag"]
    )


def _edge_match_bond(a: Dict, b: Dict) -> bool:
    """Check if two bond edges are equivalent (ignoring map edges)."""
    if a.get("kind") != "bond" or b.get("kind") != "bond":
        return False
    return a.get("order") == b.get("order") and a.get("stereo", 0) == b.get("stereo", 0)


def _get_map_partner(G: nx.Graph, node: NodeID) -> Optional[NodeID]:
    """Find the atom connected to *node* via a 'map' edge."""
    for neighbor in G.neighbors(node):
        edge = G.edges[node, neighbor]
        if edge.get("kind") == "map":
            return neighbor
    return None


def get_differing_atoms(
    rxn_a: str,
    rxn_b: str,
) -> Optional[DifferingAtoms]:
    """Identify atoms with different atom-to-atom mappings between two reactions.

    Builds mapping graphs for both reactions, finds a structural isomorphism
    (bond-only, ignoring map edges), then compares the map edges under that
    isomorphism to find atoms whose mapping differs.

    Args:
        rxn_a: First mapped reaction SMILES.
        rxn_b: Second mapped reaction SMILES.

    Returns:
        DifferingAtoms with sets of differing atom node IDs for each side
        of each reaction, or None if the bond-only graphs are not
        isomorphic (i.e. the reactions have different connectivity and
        atom-by-atom comparison is not meaningful).
    """
    try:
        G_a = _rxn_to_mapping_graph(rxn_a)
        G_b = _rxn_to_mapping_graph(rxn_b)
    except Exception:
        return None

    # Build bond-only subgraphs for structural isomorphism
    G_a_bond = nx.Graph()
    G_a_bond.add_nodes_from(G_a.nodes(data=True))
    for u, v, d in G_a.edges(data=True):
        if d.get("kind") == "bond":
            G_a_bond.add_edge(u, v, **d)

    G_b_bond = nx.Graph()
    G_b_bond.add_nodes_from(G_b.nodes(data=True))
    for u, v, d in G_b.edges(data=True):
        if d.get("kind") == "bond":
            G_b_bond.add_edge(u, v, **d)

    gm = nx.isomorphism.GraphMatcher(
        G_a_bond, G_b_bond, node_match=_node_match, edge_match=_edge_match_bond
    )
    if not gm.is_isomorphic():
        return None

    iso: Dict[NodeID, NodeID] = gm.mapping

    result = DifferingAtoms()

    # Compare map edges: for each node in G_a, check if its map partner
    # corresponds to the map partner of the isomorphic node in G_b.
    all_a_nodes: Set[NodeID] = set(G_a.nodes())
    for node_a in all_a_nodes:
        partner_a = _get_map_partner(G_a, node_a)
        node_b = iso.get(node_a)
        if node_b is None:
            continue
        partner_b = _get_map_partner(G_b, node_b)

        if partner_a is not None and partner_b is not None:
            # Both mapped — check if they map to corresponding atoms
            expected_partner_b = iso.get(partner_a)
            if partner_b != expected_partner_b:
                side = node_a[0]
                if side == "R":
                    result.reactant_a.add(node_a)
                    result.reactant_b.add(node_b)
                else:
                    result.product_a.add(node_a)
                    result.product_b.add(node_b)
        elif partner_a is not None and partner_b is None:
            # A maps this atom, B doesn't
            side = node_a[0]
            if side == "R":
                result.reactant_a.add(node_a)
                result.reactant_b.add(node_b)
            else:
                result.product_a.add(node_a)
                result.product_b.add(node_b)
        elif partner_a is None and partner_b is not None:
            # B maps this atom, A doesn't
            side = node_a[0]
            if side == "R":
                result.reactant_a.add(node_a)
                result.reactant_b.add(node_b)
            else:
                result.product_a.add(node_a)
                result.product_b.add(node_b)

    return result


# ---------------------------------------------------------------------------
# Reaction drawing with per-atom highlighting
# ---------------------------------------------------------------------------


# Colors for highlighting (R, G, B) in 0–1 range
_HIGHLIGHT_DIFF = (1.0, 0.5, 0.0)  # orange for differing atoms
_HIGHLIGHT_AGREE = (0.5, 0.7, 1.0)  # light blue for agreeing atoms


def _node_ids_to_atom_indices(
    node_ids: Set[NodeID],
    mols: List,
) -> Set[int]:
    """Convert (side, frag_i, atom_i) node IDs to flat atom indices.

    The flat index is the cumulative atom index across all fragments,
    matching the indexing used by RDKit's reaction drawing.
    """
    result: Set[int] = set()
    offset = 0
    for frag_i, mol in enumerate(mols):
        n_atoms = mol.GetNumAtoms()
        for node in node_ids:
            if node[1] == frag_i:
                result.add(offset + node[2])
        offset += n_atoms
    return result


def draw_reaction_highlighted(
    mapped_rxn: str,
    highlight_atoms: Optional[Set[NodeID]] = None,
    width: int = 1600,
    height: int = 500,
    show_agreeing: bool = False,
) -> Optional[bytes]:
    """Render a mapped reaction with per-atom highlighting.

    Draws each reactant and product molecule individually using
    ``DrawMolecule`` with ``highlightAtoms``, then composes them into
    a single reaction image with an arrow.

    Args:
        mapped_rxn: Atom-mapped reaction SMILES string.
        highlight_atoms: Set of (side, frag_i, atom_i) node IDs to
            highlight as differing.  If None, no highlighting is applied.
        width: Total image width in pixels.
        height: Image height in pixels.
        show_agreeing: If True, atoms that are mapped but not in
            highlight_atoms are shown in a neutral highlight color.

    Returns:
        PNG image bytes, or None if the reaction cannot be parsed.
    """
    if not mapped_rxn or not mapped_rxn.strip():
        return None

    try:
        rxn = AllChem.ReactionFromSmarts(mapped_rxn, useSmiles=True)  # type: ignore[attr-defined]
    except Exception:
        return None

    if rxn is None:
        return None

    r_mols = [rxn.GetReactantTemplate(i) for i in range(rxn.GetNumReactantTemplates())]
    p_mols = [rxn.GetProductTemplate(i) for i in range(rxn.GetNumProductTemplates())]

    # Compute flat atom indices for highlighting
    r_highlight: Set[int] = set()
    p_highlight: Set[int] = set()
    r_agree: Set[int] = set()
    p_agree: Set[int] = set()

    if highlight_atoms is not None:
        r_diff_nodes = {n for n in highlight_atoms if n[0] == "R"}
        p_diff_nodes = {n for n in highlight_atoms if n[0] == "P"}
        r_highlight = _node_ids_to_atom_indices(r_diff_nodes, r_mols)
        p_highlight = _node_ids_to_atom_indices(p_diff_nodes, p_mols)

        if show_agreeing:
            # Find all mapped atoms (those with atom-map > 0)
            r_all_mapped: Set[NodeID] = set()
            p_all_mapped: Set[NodeID] = set()
            for frag_i, mol in enumerate(r_mols):
                for atom in mol.GetAtoms():
                    if atom.GetAtomMapNum() > 0:
                        r_all_mapped.add(("R", frag_i, atom.GetIdx()))
            for frag_i, mol in enumerate(p_mols):
                for atom in mol.GetAtoms():
                    if atom.GetAtomMapNum() > 0:
                        p_all_mapped.add(("P", frag_i, atom.GetIdx()))
            r_agree = _node_ids_to_atom_indices(r_all_mapped - r_diff_nodes, r_mols)
            p_agree = _node_ids_to_atom_indices(p_all_mapped - p_diff_nodes, p_mols)

    # Draw each molecule into its own panel, then crop to bounding box.
    # This eliminates all whitespace — the final image is exactly as
    # wide/tall as the drawn content plus a small arrow gap.
    from PIL import Image, ImageChops, ImageDraw

    ARROW_GAP = 60  # pixels for the reaction arrow

    def _crop_to_content(img: "Image.Image") -> "Image.Image":
        """Crop an image to its non-white bounding box."""
        img_rgb = img.convert("RGB")
        bg = Image.new("RGB", img_rgb.size, (255, 255, 255))
        diff = ImageChops.difference(img_rgb, bg)
        bbox = diff.getbbox()
        if bbox:
            return img.crop(bbox)
        return img

    def _draw_mol_panel(
        mol: object,
        highlight: Set[int],
        agree: Set[int],
        panel_w: int,
        panel_h: int,
    ) -> Optional["Image.Image"]:
        sub_hl, sub_colors = _build_color_map(highlight, agree)
        sub_drawer = rdMolDraw2D.MolDraw2DCairo(panel_w, panel_h)
        sub_opts = sub_drawer.drawOptions()
        sub_opts.bondLineWidth = 2.0  # type: ignore[assignment]
        sub_opts.minFontSize = 12  # type: ignore[assignment]
        sub_opts.padding = 0.0  # type: ignore[assignment]
        sub_opts.useBWAtomPalette()  # type: ignore[assignment]
        try:
            sub_drawer.DrawMolecule(
                mol,  # type: ignore[arg-type]
                highlightAtoms=list(sub_hl) if sub_hl else None,
                highlightAtomColors=sub_colors if sub_colors else None,
            )
            sub_drawer.FinishDrawing()
            img = Image.open(io.BytesIO(sub_drawer.GetDrawingText()))
            return _crop_to_content(img)
        except Exception:
            return None

    # Build highlight color dicts for DrawMolecule
    def _build_color_map(
        highlight: Set[int], agree: Set[int]
    ) -> Tuple[Set[int], Dict[int, Tuple[float, float, float]]]:
        atoms: Set[int] = set()
        colors: Dict[int, Tuple[float, float, float]] = {}
        for idx in highlight:
            atoms.add(idx)
            colors[idx] = _HIGHLIGHT_DIFF
        for idx in agree:
            atoms.add(idx)
            colors[idx] = _HIGHLIGHT_AGREE
        return atoms, colors

    # Use a large panel for drawing, then crop — the final size is
    # determined by the content, not the panel.
    draw_panel_w = width
    draw_panel_h = height

    cropped_images: List["Image.Image"] = []
    max_h = 0

    r_offset = 0
    for frag_i, mol in enumerate(r_mols):
        n_atoms = mol.GetNumAtoms()
        hl = {
            idx - r_offset
            for idx in r_highlight
            if r_offset <= idx < r_offset + n_atoms
        }
        ag = {idx - r_offset for idx in r_agree if r_offset <= idx < r_offset + n_atoms}
        img = _draw_mol_panel(mol, hl, ag, draw_panel_w, draw_panel_h)
        if img is None:
            return None
        cropped_images.append(img)
        max_h = max(max_h, img.height)
        r_offset += n_atoms

    p_offset = 0
    for frag_i, mol in enumerate(p_mols):
        n_atoms = mol.GetNumAtoms()
        hl = {
            idx - p_offset
            for idx in p_highlight
            if p_offset <= idx < p_offset + n_atoms
        }
        ag = {idx - p_offset for idx in p_agree if p_offset <= idx < p_offset + n_atoms}
        img = _draw_mol_panel(mol, hl, ag, draw_panel_w, draw_panel_h)
        if img is None:
            return None
        cropped_images.append(img)
        max_h = max(max_h, img.height)
        p_offset += n_atoms

    # Compute total width: sum of all cropped images + arrow gap
    total_w = sum(img.width for img in cropped_images) + ARROW_GAP
    canvas = Image.new("RGBA", (total_w, max_h), (255, 255, 255, 255))

    x_offset = 0
    n_r = len(r_mols)
    for i, img in enumerate(cropped_images):
        # Vertically center
        y = (max_h - img.height) // 2
        canvas.paste(img, (x_offset, y))
        x_offset += img.width
        # Insert arrow after the last reactant
        if i == n_r - 1:
            arrow_y = max_h // 2
            draw = ImageDraw.Draw(canvas)
            arrow_start = x_offset + 10
            arrow_end = x_offset + ARROW_GAP - 10
            draw.line(
                [(arrow_start, arrow_y), (arrow_end, arrow_y)],
                fill=(0, 0, 0, 255),
                width=3,
            )
            draw.polygon(
                [
                    (arrow_end, arrow_y),
                    (arrow_end - 12, arrow_y - 8),
                    (arrow_end - 12, arrow_y + 8),
                ],
                fill=(0, 0, 0, 255),
            )
            x_offset += ARROW_GAP

    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    return buf.getvalue()


def draw_reaction_png(
    mapped_rxn: str,
    width: int = 1600,
    height: int = 500,
    highlight_map_nums: Optional[Set[int]] = None,
) -> Optional[bytes]:
    """Render a mapped reaction SMILES as a PNG image.

    Args:
        mapped_rxn: Atom-mapped reaction SMILES string.
        width: Image width in pixels.
        height: Image height in pixels.
        highlight_map_nums: Unused, kept for backward compatibility.

    Returns:
        PNG image bytes, or None if the reaction cannot be parsed.
    """
    return draw_reaction_highlighted(mapped_rxn, None, width, height)


def draw_reaction_to_bytes_io(
    mapped_rxn: str,
    width: int = 900,
    height: int = 350,
    highlight_map_nums: Optional[Set[int]] = None,
) -> io.BytesIO:
    """Render a mapped reaction and return a BytesIO for IPython.display.

    Args:
        mapped_rxn: Atom-mapped reaction SMILES string.
        width: Image width in pixels.
        height: Image height in pixels.
        highlight_map_nums: Set of atom-map numbers to highlight (placeholder).

    Returns:
        BytesIO containing PNG data.  Empty if rendering fails.
    """
    png = draw_reaction_png(mapped_rxn, width, height, highlight_map_nums)
    buf = io.BytesIO()
    if png:
        buf.write(png)
    return buf


@dataclass
class MapperEntry:
    """A single mapper's output for one reaction.

    Attributes:
        mapped_rxn: Atom-mapped reaction SMILES.
        confidence: Confidence score from the mapper (0.0–1.0), or None.
        mapper_name: Name of the mapper.
    """

    mapped_rxn: str
    confidence: Optional[float]
    mapper_name: str


def prepare_comparison_data(
    rows: List[Dict[str, str]],
    mapper_columns: Dict[str, str],
    confidence_columns: Optional[Dict[str, str]] = None,
) -> List[Tuple[int, List[MapperEntry]]]:
    """Prepare comparison data from CSV-like rows.

    Filters to only reactions where at least two mappers produce different
    templates, and returns per-reaction mapper entries.

    Args:
        rows: List of dicts (e.g. from csv.DictReader), one per reaction.
        mapper_columns: Mapping from mapper name to the CSV column name
            containing that mapper's mapped reaction SMILES.
        confidence_columns: Optional mapping from mapper name to the CSV
            column name containing confidence scores.

    Returns:
        List of (row_index, [MapperEntry, ...]) tuples for reactions where
        at least two mappers disagree on the template.
    """
    if confidence_columns is None:
        confidence_columns = {}

    results: List[Tuple[int, List[MapperEntry]]] = []

    for idx, row in enumerate(rows):
        entries: List[MapperEntry] = []
        templates: List[TemplateInfo] = []

        for mapper_name, col in mapper_columns.items():
            mapped_rxn = row.get(col, "")
            conf_col = confidence_columns.get(mapper_name)
            conf_str = row.get(conf_col, "") if conf_col else ""
            confidence: Optional[float] = None
            if conf_str:
                try:
                    confidence = float(conf_str)
                except ValueError:
                    confidence = None

            entries.append(
                MapperEntry(
                    mapped_rxn=mapped_rxn,
                    confidence=confidence,
                    mapper_name=mapper_name,
                )
            )
            templates.append(extract_template(mapped_rxn))

        # Keep only reactions where at least two mappers disagree
        has_disagreement = False
        for i in range(len(templates)):
            for j in range(i + 1, len(templates)):
                if templates_differ(templates[i], templates[j]):
                    has_disagreement = True
                    break
            if has_disagreement:
                break

        if has_disagreement:
            results.append((idx, entries))

    return results
