"""Transparent matching between a preferences-and-supports profile and a verified job listing.

The match is a distance between what a candidate prefers and how a verified listing describes the
workplace, on a small set of dimensions. It is explainable on purpose: every number can be traced
back to one dimension. It is a fit between preferences and a description, never an assessment of a
person's ability, and it never uses a diagnosis.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# key: (label, meaning of 1, meaning of 5)
DIMENSIONS = {
    "noise": ("Noise level", "Quiet", "Lively"),
    "social": ("Social interaction", "Mostly solo", "Lots of teamwork"),
    "structure": ("Task structure", "Clear steps", "Open-ended"),
    "schedule": ("Schedule", "Fixed hours", "Flexible"),
    "communication": ("Communication", "Written", "Verbal"),
}

SUPPORTS = {
    "transport": "Public transport route guide",
    "firstday": "Step-by-step first-day guide",
    "interview": "Interview practice with a chatbot",
}

VERIFICATION = {
    0: "Not verified",
    1: "Documents checked",
    2: "Workplace shown",
    3: "Visited",
}

MUST_HAVE_MIN_FIT = 0.75  # a must-have dimension is unmet below this per-dimension fit
MUST_HAVE_WEIGHT = 2.0  # must-have dimensions count double in the total


def _check_values(values: dict, what: str) -> None:
    missing = set(DIMENSIONS) - set(values)
    if missing:
        raise ValueError(f"{what} is missing dimensions: {sorted(missing)}")
    for key, value in values.items():
        if key not in DIMENSIONS:
            raise ValueError(f"{what} has an unknown dimension: {key}")
        if not 1 <= value <= 5:
            raise ValueError(f"{what}: {key} must be between 1 and 5, got {value}")


@dataclass
class Candidate:
    id: str
    prefs: dict
    must_have: frozenset = frozenset()
    supports: frozenset = frozenset()
    group: str = ""  # only used to check the matching for bias on synthetic data

    def __post_init__(self) -> None:
        _check_values(self.prefs, "Candidate prefs")
        if not set(self.must_have) <= set(DIMENSIONS):
            raise ValueError("must_have contains an unknown dimension")
        if not set(self.supports) <= set(SUPPORTS):
            raise ValueError("supports contains an unknown support")


@dataclass
class Listing:
    id: str
    title: str
    values: dict
    city: str = ""
    supports: frozenset = frozenset()
    verification: int = 0

    def __post_init__(self) -> None:
        _check_values(self.values, "Listing values")
        if self.verification not in VERIFICATION:
            raise ValueError(f"verification must be one of {sorted(VERIFICATION)}")


@dataclass
class MatchResult:
    total: float  # 0 to 100
    per_dim: dict = field(default_factory=dict)  # fit per dimension, 0 to 1
    unmet_must_haves: list = field(default_factory=list)
    supports_met: list = field(default_factory=list)
    supports_missing: list = field(default_factory=list)


def dim_fit(preference: int, value: int) -> float:
    """1.0 when preference and listing agree, 0.0 at the largest possible gap (4 steps)."""
    return 1 - abs(preference - value) / 4


def score(candidate: Candidate, listing: Listing) -> MatchResult:
    numerator = 0.0
    denominator = 0.0
    per_dim = {}
    unmet = []
    for key in DIMENSIONS:
        fit = dim_fit(candidate.prefs[key], listing.values[key])
        per_dim[key] = fit
        is_must = key in candidate.must_have
        weight = MUST_HAVE_WEIGHT if is_must else 1.0
        numerator += weight * fit
        denominator += weight
        if is_must and fit < MUST_HAVE_MIN_FIT:
            unmet.append(key)
    wanted = sorted(candidate.supports)
    return MatchResult(
        total=100 * numerator / denominator,
        per_dim=per_dim,
        unmet_must_haves=unmet,
        supports_met=[s for s in wanted if s in listing.supports],
        supports_missing=[s for s in wanted if s not in listing.supports],
    )


def rank(candidate: Candidate, listings: list) -> list:
    """Listings sorted best first. A listing with an unmet must-have is ranked after all listings
    without one (it is not hidden). Ties go to the better verified listing, then to the id."""
    results = [(listing, score(candidate, listing)) for listing in listings]
    results.sort(
        key=lambda pair: (
            len(pair[1].unmet_must_haves) > 0,
            -pair[1].total,
            -pair[0].verification,
            pair[0].id,
        )
    )
    return results
