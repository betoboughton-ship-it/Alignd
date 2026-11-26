"""Evaluate the matching on synthetic data: does it beat simple alternatives, and is it fair to every
kind of profile?  Run:  python3 evaluate.py --candidates 2000 --listings 300 --seed 42

Read the limits in the output: this uses synthetic data and a synthetic ground truth, so it tests the
logic, not real-world satisfaction.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics

from matching import rank
from synthetic import (
    ARCHETYPES,
    generate_candidates,
    generate_listings,
    hidden_weights,
    true_satisfaction,
)


def evaluate(n_candidates: int = 2000, n_listings: int = 300, seed: int = 42, k: int = 3, ok_threshold: float = 0.7) -> dict:
    rng = random.Random(seed)
    candidates = generate_candidates(n_candidates, rng)
    listings = generate_listings(n_listings, rng)
    weights_rng = random.Random(seed + 1)
    pick_rng = random.Random(seed + 2)

    scores = {"matching": [], "noise_filter": [], "random": []}
    top1_ok = {"matching": 0, "noise_filter": 0, "random": 0}
    groups = {name: {"top_k": [], "top1_ok": 0, "no_clean_match": 0, "n": 0} for name in ARCHETYPES}

    for c in candidates:
        w = hidden_weights(weights_rng)
        ranked = rank(c, listings)
        picks = {
            "matching": [pair[0] for pair in ranked[:k]],
            "noise_filter": sorted(listings, key=lambda l: (abs(c.prefs["noise"] - l.values["noise"]), l.id))[:k],
            "random": pick_rng.sample(listings, k),
        }
        for method, chosen in picks.items():
            sats = [true_satisfaction(c, l, w) for l in chosen]
            scores[method].append(statistics.mean(sats))
            if sats[0] >= ok_threshold:
                top1_ok[method] += 1
            if method == "matching":
                g = groups[c.group]
                g["top_k"].append(statistics.mean(sats))
                g["n"] += 1
                if sats[0] >= ok_threshold:
                    g["top1_ok"] += 1
                if ranked[0][1].unmet_must_haves:
                    g["no_clean_match"] += 1

    methods = {
        m: {"mean_top_k_satisfaction": round(statistics.mean(v), 3), "top1_ok_rate": round(top1_ok[m] / n_candidates, 3)}
        for m, v in scores.items()
    }
    group_rows = {}
    for name, g in groups.items():
        if g["n"] == 0:
            continue
        group_rows[name] = {
            "n": g["n"],
            "mean_top_k_satisfaction": round(statistics.mean(g["top_k"]), 3),
            "top1_ok_rate": round(g["top1_ok"] / g["n"], 3),
            "no_clean_match_rate": round(g["no_clean_match"] / g["n"], 3),
        }
    means = [row["mean_top_k_satisfaction"] for row in group_rows.values()]
    return {
        "settings": {"candidates": n_candidates, "listings": n_listings, "seed": seed, "k": k, "ok_threshold": ok_threshold},
        "methods": methods,
        "groups": group_rows,
        "group_gap": round(max(means) - min(means), 3),
    }


def print_report(result: dict) -> None:
    s = result["settings"]
    print(f"SYNTHETIC DATA: {s['candidates']} candidates, {s['listings']} listings, seed {s['seed']}")
    print(f"Quality = true satisfaction (0-1) of the top {s['k']}; 'ok' = top-1 at or above {s['ok_threshold']}\n")
    print(f"{'Method':<16}{'Mean top-k':>12}{'Top-1 ok':>12}")
    for name, row in result["methods"].items():
        print(f"{name:<16}{row['mean_top_k_satisfaction']:>12.3f}{row['top1_ok_rate']:>12.3f}")
    print(f"\n{'Group (matching)':<20}{'n':>6}{'Mean top-k':>12}{'Top-1 ok':>10}{'No clean':>10}")
    for name, row in result["groups"].items():
        print(f"{name:<20}{row['n']:>6}{row['mean_top_k_satisfaction']:>12.3f}{row['top1_ok_rate']:>10.3f}{row['no_clean_match_rate']:>10.3f}")
    print(f"\nGap between best and worst group (mean top-k): {result['group_gap']:.3f}")
    print("Limits: synthetic profiles, assumed job market and an assumed ground truth. This tests the logic and")
    print("looks for unfairness; it does not show that real users will be satisfied.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=int, default=2000)
    parser.add_argument("--listings", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="results.json")
    args = parser.parse_args()
    outcome = evaluate(args.candidates, args.listings, args.seed)
    print_report(outcome)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(outcome, handle, indent=2)
    print(f"\nSaved {args.out}")
