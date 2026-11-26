import unittest

from evaluate import evaluate
from matching import DIMENSIONS, Candidate, Listing, dim_fit, rank, score

ALL3 = {d: 3 for d in DIMENSIONS}


class MatchingTests(unittest.TestCase):
    def test_identical_profile_scores_100(self):
        c = Candidate("c", dict(ALL3), frozenset({"noise"}))
        l = Listing("l", "Same", dict(ALL3))
        result = score(c, l)
        self.assertAlmostEqual(result.total, 100.0)
        self.assertEqual(result.unmet_must_haves, [])

    def test_fit_is_symmetric_and_in_range(self):
        for a in range(1, 6):
            for b in range(1, 6):
                self.assertEqual(dim_fit(a, b), dim_fit(b, a))
                self.assertTrue(0.0 <= dim_fit(a, b) <= 1.0)

    def test_must_have_counts_double(self):
        c = Candidate("c", dict(ALL3), frozenset({"noise"}))
        values = dict(ALL3, noise=5)
        result = score(c, Listing("l", "Loud", values))
        # noise fit 0.5 weighted 2, four other dimensions fit 1.0 weighted 1 each
        self.assertAlmostEqual(result.total, 100 * (2 * 0.5 + 4) / 6)
        self.assertEqual(result.unmet_must_haves, ["noise"])

    def test_unmet_must_have_ranks_after_clean_match(self):
        c = Candidate("c", {**ALL3, "noise": 1}, frozenset({"noise"}))
        # A is close on everything else but loud; B is a little worse overall but quiet
        loud = Listing("a", "Loud but otherwise perfect", {**ALL3, "noise": 5})
        quiet = Listing("b", "Quiet, rest off by 2", {"noise": 1, "social": 5, "structure": 5, "schedule": 5, "communication": 5})
        self.assertGreater(score(c, loud).total, score(c, quiet).total - 100)  # sanity
        order = [l.id for l, _ in rank(c, [loud, quiet])]
        self.assertEqual(order[0], "b")

    def test_ties_prefer_better_verification(self):
        c = Candidate("c", dict(ALL3))
        a = Listing("a", "Unverified", dict(ALL3), verification=0)
        b = Listing("b", "Visited", dict(ALL3), verification=3)
        self.assertEqual([l.id for l, _ in rank(c, [a, b])], ["b", "a"])

    def test_supports_are_reported(self):
        c = Candidate("c", dict(ALL3), supports=frozenset({"transport", "interview"}))
        l = Listing("l", "x", dict(ALL3), supports=frozenset({"transport"}))
        result = score(c, l)
        self.assertEqual(result.supports_met, ["transport"])
        self.assertEqual(result.supports_missing, ["interview"])

    def test_invalid_values_are_rejected(self):
        with self.assertRaises(ValueError):
            Candidate("c", {**ALL3, "noise": 6})
        with self.assertRaises(ValueError):
            Listing("l", "x", {k: v for k, v in ALL3.items() if k != "noise"})
        with self.assertRaises(ValueError):
            Candidate("c", dict(ALL3), must_have=frozenset({"unknown"}))
        with self.assertRaises(ValueError):
            Listing("l", "x", dict(ALL3), verification=7)


class EvaluationTests(unittest.TestCase):
    def test_same_seed_gives_same_result(self):
        self.assertEqual(evaluate(200, 60, seed=3), evaluate(200, 60, seed=3))

    def test_matching_beats_random_on_synthetic_data(self):
        result = evaluate(300, 100, seed=5)
        methods = result["methods"]
        self.assertGreater(methods["matching"]["mean_top_k_satisfaction"], methods["random"]["mean_top_k_satisfaction"])


if __name__ == "__main__":
    unittest.main()
