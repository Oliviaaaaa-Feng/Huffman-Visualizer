import math
import random
import unittest

from scl.compressors.limited_depth_huffman import (
    LimitedDepthHuffmanDecoder,
    LimitedDepthHuffmanEncoder,
    _compute_code_lengths,
)
from scl.core.data_block import DataBlock
from scl.core.prob_dist import ProbabilityDist


def _bruteforce_optimal_lengths(prob_dist: ProbabilityDist, max_depth: int):
    """Exhaustively search all length assignments satisfying Kraft equality."""
    symbols = prob_dist.alphabet
    num_symbols = len(symbols)
    best_cost = math.inf
    best_lengths = None

    def dfs(idx, lengths, kraft_sum):
        nonlocal best_cost, best_lengths
        if kraft_sum > 1.0 + 1e-9:
            return
        if idx == num_symbols:
            if abs(kraft_sum - 1.0) > 1e-9:
                return
            cost = sum(
                prob_dist.probability(symbols[i]) * lengths[i]
                for i in range(num_symbols)
            )
            if cost < best_cost - 1e-12:
                best_cost = cost
                best_lengths = dict(zip(symbols, lengths))
            return

        for length in range(1, max_depth + 1):
            dfs(idx + 1, lengths + [length], kraft_sum + 2.0 ** (-length))

    dfs(0, [], 0.0)

    if best_lengths is None:
        raise ValueError("No valid length assignment satisfies Kraft equality.")
    return best_cost, best_lengths


class LimitedDepthHuffmanTest(unittest.TestCase):
    def test_code_lengths_respect_depth_bound(self):
        prob_dist = ProbabilityDist(
            {"A": 0.35, "B": 0.25, "C": 0.2, "D": 0.1, "E": 0.1}
        )
        max_depth = 3

        code_lengths = _compute_code_lengths(prob_dist, max_depth=max_depth)

        self.assertEqual(set(code_lengths), set(prob_dist.alphabet))
        self.assertTrue(all(length <= max_depth for length in code_lengths.values()))

    def test_round_trip_encode_decode(self):
        prob_dist = ProbabilityDist(
            {"A": 0.4, "B": 0.25, "C": 0.2, "D": 0.1, "E": 0.05}
        )
        max_depth = 3
        symbols = list("ABBCCDDEAAAABBBCCDDEE")
        data_block = DataBlock(symbols)

        encoder = LimitedDepthHuffmanEncoder(prob_dist, max_depth=max_depth)
        decoder = LimitedDepthHuffmanDecoder(prob_dist, max_depth=max_depth)

        encoded = encoder.encode_block(data_block)
        decoded_block, bits_consumed = decoder.decode_block(encoded)

        self.assertEqual(bits_consumed, len(encoded))
        self.assertEqual(decoded_block.data_list, symbols)

    def test_invalid_depth_raises(self):
        prob_dist = ProbabilityDist(
            {"A": 0.25, "B": 0.2, "C": 0.2, "D": 0.15, "E": 0.1, "F": 0.1}
        )

        with self.assertRaises(ValueError):
            _compute_code_lengths(prob_dist, max_depth=2)


class LimitedDepthHuffmanExtraTests(unittest.TestCase):
    def test_single_symbol(self):
        prob_dist = ProbabilityDist({"X": 1.0})
        lengths = _compute_code_lengths(prob_dist, max_depth=5)
        # single symbol should get length 1 by convention here
        self.assertEqual(lengths, {"X": 1})

        enc = LimitedDepthHuffmanEncoder(prob_dist, max_depth=5)
        dec = LimitedDepthHuffmanDecoder(prob_dist, max_depth=5)
        blk = DataBlock(["X"] * 10)
        enc_bits = enc.encode_block(blk)
        dec_blk, consumed = dec.decode_block(enc_bits)
        self.assertEqual(consumed, len(enc_bits))
        self.assertEqual(dec_blk.data_list, blk.data_list)

    def test_kraft_equality(self):
        prob_dist = ProbabilityDist({"A":0.35, "B":0.25, "C":0.2, "D":0.1, "E":0.1})
        max_depth = 3
        lengths = _compute_code_lengths(prob_dist, max_depth)
        kraft = sum(2.0 ** (-l) for l in lengths.values())
        self.assertTrue(abs(kraft - 1.0) < 1e-9)

    def test_monotone_lengths_vs_probability(self):
        # Higher probability ⇒ no longer code than a lower probability symbol
        prob_dist = ProbabilityDist({"A":0.4, "B":0.3, "C":0.2, "D":0.1})
        L = 3
        lengths = _compute_code_lengths(prob_dist, L)
        # sort by probability desc
        items = sorted(prob_dist.alphabet, key=lambda s: -prob_dist.probability(s))
        for i in range(len(items)-1):
            p_i = prob_dist.probability(items[i])
            p_j = prob_dist.probability(items[i+1])
            l_i = lengths[items[i]]
            l_j = lengths[items[i+1]]
            # if p_i >= p_j then l_i <= l_j
            self.assertLessEqual(l_i, l_j, msg=f"{items[i]} (p={p_i}) has length {l_i} > {l_j} of {items[i+1]} (p={p_j})")

    def test_balanced_when_equal_probs(self):
        # With equal probabilities and sufficient depth, lengths should be as balanced as possible
        prob_dist = ProbabilityDist({"A":0.25, "B":0.25, "C":0.25, "D":0.25})
        L = 2
        lengths = _compute_code_lengths(prob_dist, L)
        # For n=4 and L=2, optimal is all length-2
        self.assertEqual(set(lengths.values()), {2})

    def test_exact_capacity_n_equals_2_pow_L(self):
        # n == 2^L forces all lengths == L
        n = 8
        alphabet = {f"s{i}": 1.0/n for i in range(n)}
        prob_dist = ProbabilityDist(alphabet)
        L = 3
        lengths = _compute_code_lengths(prob_dist, L)
        self.assertTrue(all(l == L for l in lengths.values()))

    def test_ties_are_stable(self):
        # Two symbols with identical probability: stable order shouldn’t break canonical build
        prob_dist = ProbabilityDist({"A":0.4, "B":0.2, "C":0.2, "D":0.2})
        L = 3
        lengths = _compute_code_lengths(prob_dist, L)
        # Just ensure it returns all symbols and respects bound
        self.assertEqual(set(lengths.keys()), set(prob_dist.alphabet))
        self.assertTrue(all(1 <= l <= L for l in lengths.values()))

    def test_depth_limit_binding(self):
        # This forces many symbols near the limit; ensure bound respected and decodes roundtrip
        probs = {"A":0.3, "B":0.2, "C":0.15, "D":0.12, "E":0.1, "F":0.08, "G":0.05}
        prob_dist = ProbabilityDist(probs)
        L = 3
        lengths = _compute_code_lengths(prob_dist, L)
        self.assertTrue(all(l <= L for l in lengths.values()))
        kraft = sum(2.0 ** (-l) for l in lengths.values())
        self.assertTrue(abs(kraft - 1.0) < 1e-9)

        enc = LimitedDepthHuffmanEncoder(prob_dist, max_depth=L)
        dec = LimitedDepthHuffmanDecoder(prob_dist, max_depth=L)
        seq = list("AABBBCCCDDDEEEFFFGGGAAAABBBCCCDDE")
        blk = DataBlock(seq)
        enc_bits = enc.encode_block(blk)
        dec_blk, consumed = dec.decode_block(enc_bits)
        self.assertEqual(consumed, len(enc_bits))
        self.assertEqual(dec_blk.data_list, seq)

    def test_raises_when_capacity_too_small(self):
        # n = 6, L = 2 ⇒ capacity 4 < 6, must raise
        prob_dist = ProbabilityDist({"s1":0.2,"s2":0.2,"s3":0.2,"s4":0.2,"s5":0.1,"s6":0.1})
        with self.assertRaises(ValueError):
            _compute_code_lengths(prob_dist, max_depth=2)
    
    def test_kraft_equality_and_optimality_lock(self):
        # Crafted so that slack-at-last-level could be (wrongly) accepted
        prob_dist = ProbabilityDist({"A":0.34, "B":0.33, "C":0.18, "D":0.08, "E":0.07})
        L = 3
        lengths = _compute_code_lengths(prob_dist, L)
        kraft = sum(2.0 ** (-l) for l in lengths.values())
        # With the equality fix, we hit equality exactly.
        assert abs(kraft - 1.0) < 1e-9


class LimitedDepthHuffmanComplexTests(unittest.TestCase):
    def test_bruteforce_optimality_small_case(self):
        prob_dist = ProbabilityDist({"A": 0.4, "B": 0.3, "C": 0.2, "D": 0.1})
        max_depth = 3
        dp_lengths = _compute_code_lengths(prob_dist, max_depth)
        dp_cost = sum(
            prob_dist.probability(s) * dp_lengths[s] for s in prob_dist.alphabet
        )
        brute_cost, brute_lengths = _bruteforce_optimal_lengths(prob_dist, max_depth)

        self.assertAlmostEqual(dp_cost, brute_cost, places=12)
        self.assertEqual(dp_lengths, brute_lengths)

    def test_random_distributions_match_bruteforce(self):
        random.seed(0)
        symbols = ["A", "B", "C", "D", "E"]
        max_depth = 4
        for _ in range(5):
            weights = [random.random() + 0.2 for _ in symbols]
            total = sum(weights)
            prob_dist = ProbabilityDist({s: w / total for s, w in zip(symbols, weights)})

            dp_lengths = _compute_code_lengths(prob_dist, max_depth)
            dp_cost = sum(prob_dist.probability(s) * dp_lengths[s] for s in symbols)

            brute_cost, _ = _bruteforce_optimal_lengths(prob_dist, max_depth)
            self.assertAlmostEqual(dp_cost, brute_cost, places=10)



if __name__ == "__main__":
    unittest.main()
