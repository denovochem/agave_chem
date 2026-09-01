"""Unit tests for _offset_map_nums and _combine_child_smirks in template_mapper."""

from __future__ import annotations

from agave_chem.mappers.template.template_mapper import (
    _combine_child_smirks,
    _offset_map_nums,
)

# ---------------------------------------------------------------------------
# _offset_map_nums
# ---------------------------------------------------------------------------


class TestOffsetMapNums:
    """Tests for _offset_map_nums."""

    def test_simple_atom_map(self):
        assert _offset_map_nums("[C:1]>>[C:1]", 100) == "[C:101]>>[C:101]"

    def test_multiple_atom_maps(self):
        smirks = "[C:1]-[O:2]>>[C:1]-[O:2]"
        expected = "[C:101]-[O:102]>>[C:101]-[O:102]"
        assert _offset_map_nums(smirks, 100) == expected

    def test_complex_atom_expression(self):
        smirks = "[O;H0;D2;+0:102]-[*:103]"
        expected = "[O;H0;D2;+0:202]-[*:203]"
        assert _offset_map_nums(smirks, 100) == expected

    def test_reserved_map_nums_not_offset(self):
        smirks = "[*:900]-[*:950]>>[*:900]-[*:950]"
        assert _offset_map_nums(smirks, 100) == smirks

    def test_mixed_reserved_and_non_reserved(self):
        smirks = "[*:1]-[*:900]>>[*:1]-[*:900]"
        expected = "[*:101]-[*:900]>>[*:101]-[*:900]"
        assert _offset_map_nums(smirks, 100) == expected

    def test_aromatic_ring_closure_not_offset(self):
        """Ring closure digits after aromatic bonds must not be touched."""
        smirks = "[c:201]1:[c:202]:[c:203]:[c:204]:[c:205]:[c:206]:1"
        expected = "[c:301]1:[c:302]:[c:303]:[c:304]:[c:305]:[c:306]:1"
        assert _offset_map_nums(smirks, 100) == expected

    def test_fused_ring_closures_not_offset(self):
        """Fused-ring closure digits (:1:2) must not be touched."""
        smirks = "[c:11]1:[c:12]:[c:13]2:[c:14]:[c:15]:[c:16]:[c:17]:[c:18]:1:2"
        expected = (
            "[c:111]1:[c:112]:[c:113]2:[c:114]:[c:115]:[c:116]:[c:117]:[c:118]:1:2"
        )
        assert _offset_map_nums(smirks, 100) == expected

    def test_user_example_ester_plus_aromatic(self):
        """The exact example from the bug report."""
        smirks = (
            "[*]-[C;H0;D3;+0](=[O;D1;H0])-[O;H0;D2;+0:102]-[*:103]."
            "[c:201]1:[c:202]:[c:203]:[c:204]:[c:205]:[c:206]:1"
        )
        expected = (
            "[*]-[C;H0;D3;+0](=[O;D1;H0])-[O;H0;D2;+0:202]-[*:203]."
            "[c:301]1:[c:302]:[c:303]:[c:304]:[c:305]:[c:306]:1"
        )
        assert _offset_map_nums(smirks, 100) == expected

    def test_heterocycle_with_aromatic_bonds(self):
        """Pyridine-like heterocycle with explicit aromatic bonds."""
        smirks = "[n:10]1:[c:11]:[n:12]:[c:13]:[c:14]:1"
        expected = "[n:110]1:[c:111]:[n:112]:[c:113]:[c:114]:1"
        assert _offset_map_nums(smirks, 100) == expected

    def test_offset_zero(self):
        smirks = "[C:1]>>[C:1]"
        assert _offset_map_nums(smirks, 0) == smirks

    def test_empty_string(self):
        assert _offset_map_nums("", 100) == ""

    def test_no_atom_maps(self):
        smirks = "c1ccccc1>>c1ccccc1"
        assert _offset_map_nums(smirks, 100) == smirks

    def test_ring_closure_digit_only_no_atom_maps(self):
        """Pure aromatic ring with no atom maps should be unchanged."""
        smirks = "c1:c:c:c:c:c:1"
        assert _offset_map_nums(smirks, 100) == smirks


# ---------------------------------------------------------------------------
# _combine_child_smirks
# ---------------------------------------------------------------------------


class TestCombineChildSmirks:
    """Tests for _combine_child_smirks."""

    def test_single_smirks(self):
        smirks = "[C:1]-[O:2]>>[C:1]-[O:2]"
        # offset = (0+1)*100 = 100
        expected = "[C:101]-[O:102]>>[C:101]-[O:102]"
        assert _combine_child_smirks([smirks]) == expected

    def test_two_smirks(self):
        s1 = "[C:1]>>[C:1]"
        s2 = "[N:1]>>[N:1]"
        # s1 offset=100, s2 offset=200
        result = _combine_child_smirks([s1, s2])
        assert result == "[C:101].[N:201]>>[C:101].[N:201]"

    def test_aromatic_ring_smirks_combined(self):
        """Two aromatic templates combined should preserve ring closures."""
        s1 = "[c:1]1:[c:2]:[c:3]:[c:4]:[c:5]:[c:6]:1>>[c:1]1:[c:2]:[c:3]:[c:4]:[c:5]:[c:6]:1"
        s2 = "[n:1]1:[c:2]:[n:3]:[c:4]:[c:5]:1>>[n:1]1:[c:2]:[n:3]:[c:4]:[c:5]:1"
        result = _combine_child_smirks([s1, s2])
        # s1 offset=100, s2 offset=200; ring closure :1 must be preserved
        expected = (
            "[c:101]1:[c:102]:[c:103]:[c:104]:[c:105]:[c:106]:1."
            "[n:201]1:[c:202]:[n:203]:[c:204]:[c:205]:1"
            ">>"
            "[c:101]1:[c:102]:[c:103]:[c:104]:[c:105]:[c:106]:1."
            "[n:201]1:[c:202]:[n:203]:[c:204]:[c:205]:1"
        )
        assert result == expected

    def test_fused_ring_smirks_combined(self):
        """Fused-ring templates combined should preserve :1:2 closures."""
        s1 = (
            "[c:11]1:[c:12]:[c:13]2:[c:14]:[c:15]:[c:16]:[c:17]:[c:18]:1:2"
            ">>"
            "[c:11]1:[c:12]:[c:13]2:[c:14]:[c:15]:[c:16]:[c:17]:[c:18]:1:2"
        )
        result = _combine_child_smirks([s1])
        expected = (
            "[c:111]1:[c:112]:[c:113]2:[c:114]:[c:115]:[c:116]:[c:117]:[c:118]:1:2"
            ">>"
            "[c:111]1:[c:112]:[c:113]2:[c:114]:[c:115]:[c:116]:[c:117]:[c:118]:1:2"
        )
        assert result == expected

    def test_reserved_maps_preserved_in_combine(self):
        s1 = "[*:900]>>[*:900]"
        s2 = "[*:1]>>[*:1]"
        result = _combine_child_smirks([s1, s2])
        assert result == "[*:900].[*:201]>>[*:900].[*:201]"
