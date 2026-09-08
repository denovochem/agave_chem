"""Modified Elo rating system for comparing atom-mapping models.

Supports multi-outcome comparisons (a_correct, b_correct, both_correct,
both_wrong) and persists results to JSON so evaluation can be paused
and resumed across sessions.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Outcome type alias — one of the four comparison results.
Outcome = str

# Valid outcome strings.
OUTCOME_A_CORRECT = "a_correct"
OUTCOME_B_CORRECT = "b_correct"
OUTCOME_BOTH_CORRECT = "both_correct"
OUTCOME_BOTH_WRONG = "both_wrong"

VALID_OUTCOMES: frozenset[str] = frozenset(
    {OUTCOME_A_CORRECT, OUTCOME_B_CORRECT, OUTCOME_BOTH_CORRECT, OUTCOME_BOTH_WRONG}
)

# Default actual-score mapping for each outcome: (score_a, score_b).
DEFAULT_OUTCOME_SCORES: Dict[str, Tuple[float, float]] = {
    OUTCOME_A_CORRECT: (1.0, 0.0),
    OUTCOME_B_CORRECT: (0.0, 1.0),
    OUTCOME_BOTH_CORRECT: (0.65, 0.65),
    OUTCOME_BOTH_WRONG: (0.35, 0.35),
}


@dataclass
class Comparison:
    """Record of a single pairwise comparison between two mappers.

    Attributes:
        model_a: Name of the first mapper in the comparison.
        model_b: Name of the second mapper in the comparison.
        outcome: One of OUTCOME_A_CORRECT, OUTCOME_B_CORRECT,
            OUTCOME_BOTH_CORRECT, OUTCOME_BOTH_WRONG.
        reaction_index: Index of the reaction in the source dataset, or -1
            if not associated with a specific reaction.
        timestamp: Sequential comparison number (0-based).
    """

    model_a: str
    model_b: str
    outcome: Outcome
    reaction_index: int = -1
    timestamp: int = 0


@dataclass
class EloRatingSystem:
    """Modified Elo rating system for model comparison with multi-outcome support.

    Attributes:
        ratings: Current Elo rating per mapper name.
        k_factor: Sensitivity of rating updates (16–32 typical).
        comparisons: Ordered list of all recorded comparisons.
        comparison_counts: Number of comparisons per (sorted) mapper pair.
        outcome_scores: Mapping from outcome string to (score_a, score_b).
    """

    model_names: List[str]
    initial_rating: float = 1500.0
    k_factor: float = 32.0
    outcome_scores: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: dict(DEFAULT_OUTCOME_SCORES)
    )
    ratings: Dict[str, float] = field(default_factory=dict)
    comparisons: List[Comparison] = field(default_factory=list)
    comparison_counts: Dict[Tuple[str, str], int] = field(
        default_factory=lambda: defaultdict(int)
    )
    _next_timestamp: int = 0

    def __post_init__(self) -> None:
        if not self.ratings:
            self.ratings = {name: self.initial_rating for name in self.model_names}
        # Backfill any missing models that appear in model_names but not ratings.
        for name in self.model_names:
            if name not in self.ratings:
                self.ratings[name] = self.initial_rating
        if self.comparisons:
            self._next_timestamp = max(c.timestamp for c in self.comparisons) + 1
        else:
            self._next_timestamp = 0

    # ------------------------------------------------------------------
    # Core Elo computation
    # ------------------------------------------------------------------

    @staticmethod
    def expected_score(rating_a: float, rating_b: float) -> Tuple[float, float]:
        """Calculate expected scores for both models.

        Args:
            rating_a: Current Elo rating of model A.
            rating_b: Current Elo rating of model B.

        Returns:
            (expected_a, expected_b): Probabilities that sum to 1.0.
        """
        expected_a = 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))
        return expected_a, 1.0 - expected_a

    def update_ratings(
        self,
        model_a: str,
        model_b: str,
        outcome: str,
        reaction_index: int = -1,
    ) -> Dict[str, float]:
        """Update Elo ratings based on a comparison outcome.

        Args:
            model_a: First mapper name.
            model_b: Second mapper name.
            outcome: One of the VALID_OUTCOMES strings.
            reaction_index: Index of the reaction in the source dataset.

        Returns:
            Dictionary with rating changes and intermediate values:
            model_a → delta, model_b → delta, plus expected and actual scores.

        Raises:
            ValueError: If outcome is not a valid outcome string, or if
                model_a or model_b is not in the ratings dict.
        """
        if outcome not in VALID_OUTCOMES:
            raise ValueError(
                f"Invalid outcome: {outcome!r}. Must be one of {sorted(VALID_OUTCOMES)}."
            )
        if model_a not in self.ratings:
            raise ValueError(f"Unknown model: {model_a!r}")
        if model_b not in self.ratings:
            raise ValueError(f"Unknown model: {model_b!r}")

        rating_a = self.ratings[model_a]
        rating_b = self.ratings[model_b]

        expected_a, expected_b = self.expected_score(rating_a, rating_b)
        score_a, score_b = self.outcome_scores[outcome]

        delta_a = self.k_factor * (score_a - expected_a)
        delta_b = self.k_factor * (score_b - expected_b)

        self.ratings[model_a] = rating_a + delta_a
        self.ratings[model_b] = rating_b + delta_b

        pair_key: Tuple[str, str] = tuple(sorted([model_a, model_b]))  # type: ignore[assignment]
        self.comparison_counts[pair_key] += 1
        ts = self._next_timestamp
        self._next_timestamp += 1
        self.comparisons.append(
            Comparison(
                model_a=model_a,
                model_b=model_b,
                outcome=outcome,
                reaction_index=reaction_index,
                timestamp=ts,
            )
        )

        return {
            model_a: delta_a,
            model_b: delta_b,
            "expected_a": expected_a,
            "expected_b": expected_b,
            "score_a": score_a,
            "score_b": score_b,
        }

    # ------------------------------------------------------------------
    # Rankings / statistics
    # ------------------------------------------------------------------

    def get_rankings(self) -> List[Tuple[str, float]]:
        """Get current model rankings sorted by rating (highest first).

        Returns:
            List of (model_name, rating) tuples.
        """
        return sorted(self.ratings.items(), key=lambda x: x[1], reverse=True)

    def get_statistics(self) -> Dict[str, object]:
        """Get summary statistics about the rating system.

        Returns:
            Dictionary with total_comparisons, rankings, rating_spread,
            comparisons_per_pair, and outcome_distribution.
        """
        rankings = self.get_rankings()
        return {
            "total_comparisons": len(self.comparisons),
            "rankings": rankings,
            "rating_spread": rankings[0][1] - rankings[-1][1] if rankings else 0.0,
            "comparisons_per_pair": dict(self.comparison_counts),
            "outcome_distribution": self._outcome_distribution(),
        }

    def _outcome_distribution(self) -> Dict[str, int]:
        """Count occurrences of each outcome type."""
        dist: Dict[str, int] = defaultdict(int)
        for comp in self.comparisons:
            dist[comp.outcome] += 1
        return dict(dist)

    # ------------------------------------------------------------------
    # Pair selection
    # ------------------------------------------------------------------

    def get_next_pair(
        self,
        available_pairs: Optional[List[Tuple[str, str]]] = None,
    ) -> Tuple[str, str]:
        """Select next pair to compare using a round-robin approach.

        Prioritises pairs that have been compared less frequently.  When
        multiple pairs share the minimum count, one is chosen at random.

        Args:
            available_pairs: Optional list of (model_a, model_b) pairs to
                choose from.  If None, all possible pairs from the current
                ratings are generated.

        Returns:
            (model_a, model_b) tuple with randomised order.
        """
        if available_pairs is None:
            models = list(self.ratings.keys())
            available_pairs = list(combinations(models, 2))

        normalised: List[Tuple[str, str]] = [
            tuple(sorted(p))  # type: ignore[misc]
            for p in available_pairs
        ]

        min_count = min(self.comparison_counts.get(p, 0) for p in normalised)
        undersampled = [
            p for p in normalised if self.comparison_counts.get(p, 0) == min_count
        ]
        selected = random.choice(undersampled)
        if random.random() < 0.5:
            return selected
        return (selected[1], selected[0])

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Persist the rating system state to a JSON file.

        Args:
            path: File path to write.
        """
        data = {
            "model_names": self.model_names,
            "initial_rating": self.initial_rating,
            "k_factor": self.k_factor,
            "outcome_scores": self.outcome_scores,
            "ratings": self.ratings,
            "comparisons": [asdict(c) for c in self.comparisons],
            "comparison_counts": {
                f"{a}||{b}": cnt for (a, b), cnt in self.comparison_counts.items()
            },
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> EloRatingSystem:
        """Load a rating system state from a JSON file.

        Args:
            path: File path to read.

        Returns:
            A reconstructed EloRatingSystem instance.
        """
        data = json.loads(Path(path).read_text())
        comparisons = [Comparison(**c) for c in data.get("comparisons", [])]
        comparison_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        for key, cnt in data.get("comparison_counts", {}).items():
            a, b = key.split("||")
            comparison_counts[(a, b)] = cnt

        return cls(
            model_names=data["model_names"],
            initial_rating=data.get("initial_rating", 1500.0),
            k_factor=data.get("k_factor", 32.0),
            outcome_scores=data.get("outcome_scores", dict(DEFAULT_OUTCOME_SCORES)),
            ratings=data.get("ratings", {}),
            comparisons=comparisons,
            comparison_counts=comparison_counts,
        )

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def leaderboard_str(self) -> str:
        """Return a formatted leaderboard string."""
        lines = ["=" * 55, "MODEL LEADERBOARD", "=" * 55]
        for rank, (model, rating) in enumerate(self.get_rankings(), 1):
            lines.append(f"{rank}. {model:25s} | Rating: {rating:7.1f}")
        lines.append("=" * 55)
        stats = self.get_statistics()
        lines.append(f"Total Comparisons: {stats['total_comparisons']}")
        lines.append(f"Rating Spread: {stats['rating_spread']:.1f}")
        lines.append("Outcome Distribution:")
        outcome_dist = stats["outcome_distribution"]
        if isinstance(outcome_dist, dict):
            for outcome, count in outcome_dist.items():
                lines.append(f"  {outcome:20s}: {count}")
        lines.append("=" * 55)
        return "\n".join(lines)
