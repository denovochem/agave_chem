"""Unit tests for the Elo/Bradley-Terry rating system.

Tests cover:
- Expected score calculation
- Rating updates for all four outcome types
- Rankings and statistics
- Pair selection (round-robin)
- JSON save/load round-trip persistence
- Error handling (invalid outcome, unknown model)
"""

import json
from pathlib import Path

import pytest

from workflows.compare_mappers.elo_rating import (
    DEFAULT_OUTCOME_SCORES,
    OUTCOME_A_CORRECT,
    OUTCOME_B_CORRECT,
    OUTCOME_BOTH_CORRECT,
    OUTCOME_BOTH_WRONG,
    VALID_OUTCOMES,
    EloRatingSystem,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def models() -> list[str]:
    return ["mapper_a", "mapper_b", "mapper_c"]


@pytest.fixture
def elo(models: list[str]) -> EloRatingSystem:
    return EloRatingSystem(model_names=models, initial_rating=1500.0, k_factor=32.0)


# ---------------------------------------------------------------------------
# expected_score
# ---------------------------------------------------------------------------


class TestExpectedScore:
    def test_equal_ratings(self) -> None:
        ea, eb = EloRatingSystem.expected_score(1500.0, 1500.0)
        assert ea == pytest.approx(0.5)
        assert eb == pytest.approx(0.5)

    def test_a_higher(self) -> None:
        ea, eb = EloRatingSystem.expected_score(1900.0, 1500.0)
        assert ea > 0.5
        assert eb < 0.5
        assert ea + eb == pytest.approx(1.0)

    def test_b_higher(self) -> None:
        ea, eb = EloRatingSystem.expected_score(1500.0, 1900.0)
        assert ea < 0.5
        assert eb > 0.5
        assert ea + eb == pytest.approx(1.0)

    def test_sum_always_one(self) -> None:
        for diff in [-800, -400, -100, 0, 100, 400, 800]:
            ea, eb = EloRatingSystem.expected_score(1500.0, 1500.0 + diff)
            assert ea + eb == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# update_ratings
# ---------------------------------------------------------------------------


class TestUpdateRatings:
    def test_a_correct_raises_a(self, elo: EloRatingSystem) -> None:
        result = elo.update_ratings("mapper_a", "mapper_b", OUTCOME_A_CORRECT)
        assert result["mapper_a"] > 0
        assert result["mapper_b"] < 0

    def test_b_correct_raises_b(self, elo: EloRatingSystem) -> None:
        result = elo.update_ratings("mapper_a", "mapper_b", OUTCOME_B_CORRECT)
        assert result["mapper_a"] < 0
        assert result["mapper_b"] > 0

    def test_both_correct_small_change(self, elo: EloRatingSystem) -> None:
        result = elo.update_ratings("mapper_a", "mapper_b", OUTCOME_BOTH_CORRECT)
        # Both get 0.65 actual vs 0.5 expected → both gain a little
        assert result["mapper_a"] > 0
        assert result["mapper_b"] > 0

    def test_both_wrong_small_loss(self, elo: EloRatingSystem) -> None:
        result = elo.update_ratings("mapper_a", "mapper_b", OUTCOME_BOTH_WRONG)
        # Both get 0.35 actual vs 0.5 expected → both lose a little
        assert result["mapper_a"] < 0
        assert result["mapper_b"] < 0

    def test_invalid_outcome_raises(self, elo: EloRatingSystem) -> None:
        with pytest.raises(ValueError, match="Invalid outcome"):
            elo.update_ratings("mapper_a", "mapper_b", "invalid_outcome")

    def test_unknown_model_raises(self, elo: EloRatingSystem) -> None:
        with pytest.raises(ValueError, match="Unknown model"):
            elo.update_ratings("unknown", "mapper_b", OUTCOME_A_CORRECT)

    def test_comparison_recorded(self, elo: EloRatingSystem) -> None:
        elo.update_ratings("mapper_a", "mapper_b", OUTCOME_A_CORRECT, reaction_index=42)
        assert len(elo.comparisons) == 1
        comp = elo.comparisons[0]
        assert comp.model_a == "mapper_a"
        assert comp.model_b == "mapper_b"
        assert comp.outcome == OUTCOME_A_CORRECT
        assert comp.reaction_index == 42
        assert comp.timestamp == 0

    def test_timestamp_increments(self, elo: EloRatingSystem) -> None:
        elo.update_ratings("mapper_a", "mapper_b", OUTCOME_A_CORRECT)
        elo.update_ratings("mapper_b", "mapper_c", OUTCOME_B_CORRECT)
        assert elo.comparisons[0].timestamp == 0
        assert elo.comparisons[1].timestamp == 1

    def test_comparison_count_increments(self, elo: EloRatingSystem) -> None:
        elo.update_ratings("mapper_a", "mapper_b", OUTCOME_A_CORRECT)
        elo.update_ratings("mapper_b", "mapper_a", OUTCOME_B_CORRECT)
        # Same pair, sorted → count should be 2
        pair = ("mapper_a", "mapper_b")
        assert elo.comparison_counts[pair] == 2


# ---------------------------------------------------------------------------
# get_rankings / get_statistics
# ---------------------------------------------------------------------------


class TestRankingsAndStats:
    def test_initial_rankings_all_equal(self, elo: EloRatingSystem) -> None:
        rankings = elo.get_rankings()
        assert len(rankings) == 3
        for _, rating in rankings:
            assert rating == 1500.0

    def test_rankings_sorted_descending(self, elo: EloRatingSystem) -> None:
        # Make mapper_a win several times
        for _ in range(5):
            elo.update_ratings("mapper_a", "mapper_b", OUTCOME_A_CORRECT)
        rankings = elo.get_rankings()
        assert rankings[0][0] == "mapper_a"
        assert rankings[0][1] > rankings[1][1]

    def test_statistics_keys(self, elo: EloRatingSystem) -> None:
        elo.update_ratings("mapper_a", "mapper_b", OUTCOME_A_CORRECT)
        stats = elo.get_statistics()
        assert "total_comparisons" in stats
        assert "rankings" in stats
        assert "rating_spread" in stats
        assert "comparisons_per_pair" in stats
        assert "outcome_distribution" in stats
        assert stats["total_comparisons"] == 1

    def test_outcome_distribution(self, elo: EloRatingSystem) -> None:
        elo.update_ratings("mapper_a", "mapper_b", OUTCOME_A_CORRECT)
        elo.update_ratings("mapper_b", "mapper_c", OUTCOME_BOTH_CORRECT)
        dist = elo._outcome_distribution()
        assert dist[OUTCOME_A_CORRECT] == 1
        assert dist[OUTCOME_BOTH_CORRECT] == 1

    def test_empty_statistics(self) -> None:
        elo = EloRatingSystem(model_names=["a", "b"])
        stats = elo.get_statistics()
        assert stats["total_comparisons"] == 0
        assert stats["rating_spread"] == 0.0


# ---------------------------------------------------------------------------
# get_next_pair
# ---------------------------------------------------------------------------


class TestGetNextPair:
    def test_returns_valid_pair(self, elo: EloRatingSystem) -> None:
        pair = elo.get_next_pair()
        assert len(pair) == 2
        assert pair[0] in elo.ratings
        assert pair[1] in elo.ratings
        assert pair[0] != pair[1]

    def test_prioritises_least_compared(self, elo: EloRatingSystem) -> None:
        # Compare a vs b once
        elo.update_ratings("mapper_a", "mapper_b", OUTCOME_A_CORRECT)
        # Next pair should be from the zero-count pairs (a,c) or (b,c)
        for _ in range(20):
            pair = elo.get_next_pair()
            sorted_pair = tuple(sorted(pair))
            assert sorted_pair != ("mapper_a", "mapper_b")

    def test_custom_available_pairs(self, elo: EloRatingSystem) -> None:
        pairs = [("mapper_a", "mapper_c")]
        pair = elo.get_next_pair(available_pairs=pairs)
        sorted_pair = tuple(sorted(pair))
        assert sorted_pair == ("mapper_a", "mapper_c")


# ---------------------------------------------------------------------------
# Persistence (save / load)
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_save_load_round_trip(self, elo: EloRatingSystem, tmp_path: Path) -> None:
        elo.update_ratings("mapper_a", "mapper_b", OUTCOME_A_CORRECT, reaction_index=5)
        elo.update_ratings("mapper_b", "mapper_c", OUTCOME_BOTH_WRONG, reaction_index=10)

        path = tmp_path / "test_elo.json"
        elo.save(path)
        assert path.exists()

        loaded = EloRatingSystem.load(path)
        assert loaded.ratings == elo.ratings
        assert len(loaded.comparisons) == 2
        assert loaded.comparisons[0].reaction_index == 5
        assert loaded.comparisons[1].reaction_index == 10
        assert loaded.k_factor == elo.k_factor
        assert loaded.initial_rating == elo.initial_rating

    def test_load_preserves_comparison_counts(self, elo: EloRatingSystem, tmp_path: Path) -> None:
        elo.update_ratings("mapper_a", "mapper_b", OUTCOME_A_CORRECT)
        elo.update_ratings("mapper_a", "mapper_b", OUTCOME_B_CORRECT)

        path = tmp_path / "test_elo.json"
        elo.save(path)
        loaded = EloRatingSystem.load(path)
        pair = ("mapper_a", "mapper_b")
        assert loaded.comparison_counts[pair] == 2

    def test_save_creates_valid_json(self, elo: EloRatingSystem, tmp_path: Path) -> None:
        elo.update_ratings("mapper_a", "mapper_b", OUTCOME_A_CORRECT)
        path = tmp_path / "test_elo.json"
        elo.save(path)
        data = json.loads(path.read_text())
        assert "ratings" in data
        assert "comparisons" in data
        assert "model_names" in data

    def test_resume_increments_timestamp(self, elo: EloRatingSystem, tmp_path: Path) -> None:
        elo.update_ratings("mapper_a", "mapper_b", OUTCOME_A_CORRECT)
        path = tmp_path / "test_elo.json"
        elo.save(path)

        loaded = EloRatingSystem.load(path)
        loaded.update_ratings("mapper_a", "mapper_c", OUTCOME_A_CORRECT)
        assert loaded.comparisons[-1].timestamp == 1


# ---------------------------------------------------------------------------
# leaderboard_str
# ---------------------------------------------------------------------------


class TestLeaderboardStr:
    def test_contains_all_models(self, elo: EloRatingSystem) -> None:
        s = elo.leaderboard_str()
        for name in elo.ratings:
            assert name in s


    def test_contains_header(self, elo: EloRatingSystem) -> None:
        s = elo.leaderboard_str()
        assert "LEADERBOARD" in s


# ---------------------------------------------------------------------------
# Constants and defaults
# ---------------------------------------------------------------------------


class TestConstants:
    def test_valid_outcomes_has_four(self) -> None:
        assert len(VALID_OUTCOMES) == 4

    def test_default_outcome_scores(self) -> None:
        assert DEFAULT_OUTCOME_SCORES[OUTCOME_A_CORRECT] == (1.0, 0.0)
        assert DEFAULT_OUTCOME_SCORES[OUTCOME_B_CORRECT] == (0.0, 1.0)
        assert DEFAULT_OUTCOME_SCORES[OUTCOME_BOTH_CORRECT] == (0.65, 0.65)
        assert DEFAULT_OUTCOME_SCORES[OUTCOME_BOTH_WRONG] == (0.35, 0.35)

    def test_outcome_scores_sum_leq_one(self) -> None:
        for sa, sb in DEFAULT_OUTCOME_SCORES.values():
            assert 0.0 <= sa <= 1.0
            assert 0.0 <= sb <= 1.0
