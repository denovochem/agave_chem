import pytest

from workflows.template_processing.generate_child_smarts_from_smirks import (
    expand_all_brackets,
    has_top_level_comma,
)


class TestHasTopLevelComma:
    @pytest.mark.parametrize(
        "s,expected",
        [
            ("Cl,Br,I", True),
            ("H1,H2", True),
            ("F,Cl,Br,I", True),
            ("#6,#7,#8", True),
            ("Cl", False),
            ("H1", False),
            ("", False),
            ("#6;H0;D3;+0", False),
        ],
    )
    def test_simple_atom_lists(self, s: str, expected: bool) -> None:
        assert has_top_level_comma(s) is expected

    @pytest.mark.parametrize(
        "s,expected",
        [
            ("$([#6]:[#6]),$([#6]=[#6]),$([#6]#[#6])", False),
            ("$([#6]=[#6]),$([#6]~[#6]:[#6]),$([#6]~n)", False),
            ("$(NC),H2&$(Nc1:[c,n]:[c,n]:1)", False),
        ],
    )
    def test_recursive_smarts_or_groups(self, s: str, expected: bool) -> None:
        assert has_top_level_comma(s) is expected

    @pytest.mark.parametrize(
        "s,expected",
        [
            ("Cl,Br,I,$([#6]=[#6])", True),
            ("$([#6]=[#6]),Cl,Br", True),
            ("H1,H2,$([#6]:[#6]),$([#6]=[#6])", True),
        ],
    )
    def test_mixed_lists_and_recursive(self, s: str, expected: bool) -> None:
        assert has_top_level_comma(s) is expected

    def test_nested_recursive_smarts(self) -> None:
        s = "$([#6;$([#6]:[#6])]),$([#6]=[#6])"
        assert has_top_level_comma(s) is False


class TestExpandAllBrackets:
    def test_simple_atom_list(self) -> None:
        result = expand_all_brackets("[Cl,Br,I]")
        assert len(result) == 3
        assert "[Cl]" in result
        assert "[Br]" in result
        assert "[I]" in result

    def test_no_commas(self) -> None:
        s = "[#6;H0;D3;+0:2]"
        result = expand_all_brackets(s)
        assert result == [s]

    def test_suzuki_pattern_produces_few_children(self) -> None:
        s = (
            "[#6;$([#6]:[#6]),$([#6]=[#6]),$([#6]#[#6]);+0:2]"
            "-[B;H0;D3;+0](-[O;H1;D1;+0])-[O;H1;D1;+0]"
            ".[#6;$([#6]=[#6]),$([#6]~[#6]:[#6]),$([#6]~n);+0:3]"
            "[Cl,Br,I]"
            ">>"
            "[#6;$([#6]:[#6]),$([#6]=[#6]),$([#6]#[#6]);+0:2]"
            "-[#6;$([#6]=[#6]),$([#6]~[#6]:[#6]),$([#6]~n);+0:3]"
        )
        result = expand_all_brackets(s)
        assert len(result) == 3
        for child in result:
            assert "[Cl]" in child or "[Br]" in child or "[I]" in child

    def test_property_alternatives_expanded(self) -> None:
        s = "[N;H1,H2;D1,D2;+0:4]"
        result = expand_all_brackets(s)
        assert len(result) == 4
        assert any("H1" in r and "D1" in r for r in result)
        assert any("H1" in r and "D2" in r for r in result)
        assert any("H2" in r and "D1" in r for r in result)
        assert any("H2" in r and "D2" in r for r in result)
