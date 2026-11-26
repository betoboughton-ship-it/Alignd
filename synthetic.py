"""Synthetic candidates and listings for testing the matching logic.

All data here is generated from assumptions written below. It is NOT evidence about real people or
real employers: it can show whether the logic behaves sensibly and where it may be unfair to some
kinds of profiles, but it cannot show that real users will be satisfied.
"""
from __future__ import annotations

import random

from matching import DIMENSIONS, SUPPORTS, Candidate, Listing

# group name: (share of candidates, mean preference per dimension). Assumptions, not research.
ARCHETYPES = {
    "quiet_structured": (0.30, {"noise": 1.5, "social": 2.0, "structure": 1.5, "schedule": 2.0, "communication": 1.5}),
    "balanced": (0.25, {"noise": 3.0, "social": 3.0, "structure": 3.0, "schedule": 3.0, "communication": 3.0}),
    "quiet_flexible": (0.20, {"noise": 1.5, "social": 2.0, "structure": 3.0, "schedule": 4.5, "communication": 2.0}),
    "social_flexible": (0.15, {"noise": 4.0, "social": 4.5, "structure": 3.5, "schedule": 4.5, "communication": 4.5}),
    "rare_mixed": (0.10, {"noise": 4.5, "social": 1.5, "structure": 1.5, "schedule": 1.5, "communication": 4.5}),
}

# Assumed average of the job market: lively, team-based, fixed hours. Not measured.
MARKET_MEANS = {"noise": 3.3, "social": 3.5, "structure": 3.0, "schedule": 2.5, "communication": 3.4}
CITIES = ["Cologne", "Bonn", "Düsseldorf"]
SUPPORT_OFFER_RATES = {"transport": 0.7, "firstday": 0.5, "interview": 0.3}


def _clip(x: float) -> int:
    return max(1, min(5, round(x)))


def generate_candidates(n: int, rng: random.Random) -> list:
    names = list(ARCHETYPES)
    shares = [ARCHETYPES[name][0] for name in names]
    candidates = []
    for i in range(n):
        group = rng.choices(names, weights=shares)[0]
        means = ARCHETYPES[group][1]
        prefs = {d: _clip(rng.gauss(means[d], 0.7)) for d in DIMENSIONS}
        extreme = [d for d, v in prefs.items() if v <= 2 or v >= 4]
        k = rng.choice([0, 1, 1, 2])
        must = frozenset(rng.sample(extreme, min(k, len(extreme))))
        supports = frozenset(s for s in SUPPORTS if rng.random() < 0.6)
        candidates.append(Candidate(f"c{i}", prefs, must, supports, group))
    return candidates


def generate_listings(m: int, rng: random.Random) -> list:
    listings = []
    for i in range(m):
        values = {d: _clip(rng.gauss(MARKET_MEANS[d], 1.1)) for d in DIMENSIONS}
        offered = frozenset(s for s, rate in SUPPORT_OFFER_RATES.items() if rng.random() < rate)
        verification = rng.choices([0, 1, 2, 3], weights=[0.1, 0.3, 0.35, 0.25])[0]
        listings.append(Listing(f"l{i}", f"Synthetic listing {i}", values, rng.choice(CITIES), offered, verification))
    return listings


def hidden_weights(rng: random.Random) -> dict:
    """How much each dimension REALLY matters to a synthetic person. The matching does not see this."""
    return {d: rng.uniform(0.5, 1.5) for d in DIMENSIONS}


def true_satisfaction(candidate: Candidate, listing: Listing, weights: dict) -> float:
    """Synthetic ground truth, 0 to 1. Deliberately different from the model: a steeper penalty for
    a gap (3 steps already means zero) and person-specific weights the model does not know."""
    num = 0.0
    den = 0.0
    for d in DIMENSIONS:
        fit = max(0.0, 1 - abs(candidate.prefs[d] - listing.values[d]) / 3)
        num += weights[d] * fit
        den += weights[d]
    return num / den
