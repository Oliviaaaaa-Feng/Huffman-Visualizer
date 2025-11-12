import json
import math
from dataclasses import dataclass, field
from itertools import count
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from scl.compressors.prefix_free_compressors import (
    PrefixFreeDecoder,
    PrefixFreeEncoder,
    PrefixFreeTree,
)
from scl.core.prob_dist import ProbabilityDist
from scl.utils.bitarray_utils import BitArray, uint_to_bitarray


@dataclass(order=True)
class _Coin:
    """Helper structure used by the package-merge/coin-collector procedure."""

    cost: float
    order: int
    symbols: List[Any] = field(compare=False)


def _build_canonical_codes(
    lengths: Mapping[Any, int], symbol_order: Sequence[Any]
) -> Dict[Any, BitArray]:
    """Return canonical prefix codes given a dictionary of code lengths."""

    rank = {symbol: idx for idx, symbol in enumerate(symbol_order)}
    sorted_symbols = sorted(lengths, key=lambda s: (lengths[s], rank[s]))

    codes: Dict[Any, BitArray] = {}
    codeword = 0
    prev_length = 0

    for symbol in sorted_symbols:
        length = lengths[symbol]
        if length <= 0:
            raise ValueError(f"Invalid code length {length} for symbol {symbol}")

        shift = length - prev_length
        if shift < 0:
            raise ValueError("Code lengths must be non-decreasing after sorting")
        codeword <<= shift
        codes[symbol] = uint_to_bitarray(codeword, bit_width=length)
        codeword += 1
        prev_length = length

    return codes


def _select_length_counts(coins: Iterable[_Coin], num_symbols: int) -> Dict[Any, int]:
    """Pick the cheapest 2(n-1) coins at level 1 and tally occurrences."""

    level_one_coins = sorted(coins)
    required = 2 * (num_symbols - 1)

    if required <= 0:
        raise ValueError("Selection requires at least one symbol")
    if len(level_one_coins) < required:
        raise ValueError("Insufficient coins to satisfy Kraft equality")

    lengths: Dict[Any, int] = {}
    for coin in level_one_coins[:required]:
        for symbol in coin.symbols:
            lengths[symbol] = lengths.get(symbol, 0) + 1
    return lengths


def _compute_code_lengths(prob_dist: ProbabilityDist, max_depth: int) -> Dict[Any, int]:
    """
    Compute limited-depth Huffman code lengths via the package-merge algorithm.
    """

    if max_depth < 1:
        raise ValueError("max_depth must be at least 1")

    symbols = prob_dist.alphabet
    num_symbols = len(symbols)

    if num_symbols == 0:
        raise ValueError("Probability distribution must contain at least one symbol")

    if num_symbols == 1:
        # Degenerate case – use a single-bit code to keep the tree well-defined.
        return {symbols[0]: 1}

    if num_symbols > (1 << max_depth):
        raise ValueError(
            f"Depth {max_depth} cannot support {num_symbols} symbols (capacity {1 << max_depth})"
        )

    probabilities = {s: prob_dist.probability(s) for s in symbols}
    order_counter = count()

    def make_coin(cost: float, symbols_list: Iterable[Any]) -> _Coin:
        return _Coin(cost=cost, order=next(order_counter), symbols=list(symbols_list))

    # Maintain coins for each level (1-indexed). Each level starts with the n singleton coins.
    coins_per_level: List[List[_Coin]] = [[] for _ in range(max_depth + 1)]
    for level in range(1, max_depth + 1):
        for symbol in symbols:
            coins_per_level[level].append(make_coin(probabilities[symbol], [symbol]))

    # Package-merge: move from deepest level upwards, pairing cheapest coins.
    for level in range(max_depth, 1, -1):
        current_level = sorted(coins_per_level[level])
        if len(current_level) % 2 == 1:
            current_level = current_level[:-1]  # leave the heaviest coin unused

        for i in range(0, len(current_level), 2):
            left = current_level[i]
            right = current_level[i + 1]
            merged_symbols = left.symbols + right.symbols
            merged_cost = left.cost + right.cost
            coins_per_level[level - 1].append(make_coin(merged_cost, merged_symbols))

    lengths = _select_length_counts(coins_per_level[1], num_symbols)

    # Ensure every symbol was assigned a length.
    missing = set(symbols) - set(lengths)
    if missing:
        raise ValueError(f"Failed to assign code lengths for symbols: {missing}")

    # Validate bounds and Kraft equality (full binary tree).
    for symbol, length in lengths.items():
        if length <= 0 or length > max_depth:
            raise ValueError(
                f"Invalid length {length} for symbol {symbol}; violates depth constraint"
            )

    kraft_sum = sum(2.0 ** (-length) for length in lengths.values())
    if not math.isclose(kraft_sum, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("Computed lengths do not satisfy Kraft equality")

    return lengths


class LimitedDepthHuffmanEncoder(PrefixFreeEncoder):
    """Limited-depth Huffman encoder built from the package-merge codebook."""

    def __init__(self, prob_dist: ProbabilityDist, max_depth: int):
        self.prob_dist = prob_dist
        self.max_depth = max_depth
        lengths = _compute_code_lengths(prob_dist, max_depth)
        self.encoding_table = _build_canonical_codes(lengths, prob_dist.alphabet)
        self.tree = PrefixFreeTree.build_prefix_free_tree_from_code(
            self.encoding_table
        )

    def encode_symbol(self, s) -> BitArray:
        return self.encoding_table[s]

    def export_tree_json(
        self,
        output_path: str,
        symbol_weights: Optional[Mapping[Any, float]] = None,
    ) -> None:
        """
        Serialize the Huffman tree so the frontend can visualize it.

        Args:
            output_path: destination path for the JSON payload.
            symbol_weights: optional explicit weights per symbol. By default we
                reuse the probabilities passed into the encoder.
        """
        if self.tree is None:
            raise ValueError("Tree not available; build the encoder before exporting.")

        weights = (
            dict(symbol_weights)
            if symbol_weights is not None
            else {s: self.prob_dist.probability(s) for s in self.prob_dist.alphabet}
        )

        missing = set(self.prob_dist.alphabet) - set(weights)
        if missing:
            raise ValueError(f"Missing weights for symbols: {missing}")

        nodes: List[Dict[str, Any]] = []

        def _collect(node, depth: int):
            if node is None:
                return None, 0.0

            if node.is_leaf_node:
                symbol = node.id
                label = str(symbol)
                weight = float(weights[symbol])
                nodes.append(
                    {
                        "id": label,
                        "weight": weight,
                        "depth": depth,
                    }
                )
                return label, weight

            left_label, left_weight = _collect(node.left_child, depth + 1)
            right_label, right_weight = _collect(node.right_child, depth + 1)

            child_labels = [lbl for lbl in [left_label, right_label] if lbl is not None]
            label = "".join(sorted(child_labels)) if child_labels else f"node_{depth}"
            weight = left_weight + right_weight
            nodes.append(
                {
                    "id": label,
                    "weight": float(weight),
                    "depth": depth,
                    "left": left_label,
                    "right": right_label,
                }
            )
            return label, weight

        _collect(self.tree.root_node, 0)
        ordered_nodes = list(reversed(nodes))
        payload = {"tree": ordered_nodes}

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


class LimitedDepthHuffmanDecoder(PrefixFreeDecoder):
    """Matching decoder for the limited-depth Huffman code."""

    def __init__(self, prob_dist: ProbabilityDist, max_depth: int):
        lengths = _compute_code_lengths(prob_dist, max_depth)
        encoding_table = _build_canonical_codes(lengths, prob_dist.alphabet)
        self.tree = PrefixFreeTree.build_prefix_free_tree_from_code(encoding_table)

    def decode_symbol(self, encoded_bitarray: BitArray):
        decoded_symbol, bits_used = self.tree.decode_symbol(encoded_bitarray)
        return decoded_symbol, bits_used


__all__ = [
    "_compute_code_lengths",
    "LimitedDepthHuffmanEncoder",
    "LimitedDepthHuffmanDecoder",
]
