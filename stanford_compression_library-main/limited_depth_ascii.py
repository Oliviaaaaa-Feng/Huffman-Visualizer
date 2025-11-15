#!/usr/bin/env python3
"""
Build and visualize a limited-depth Huffman tree.

Example:
    python limited_depth_ascii.py "[0.8, 0.1, 0.1]" 2
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from scl.compressors.limited_depth_huffman import LimitedDepthHuffmanEncoder
from scl.core.prob_dist import ProbabilityDist


def _parse_probability_list(raw: str) -> List[float]:
    """Return a list of floats from a JSON/pythonic/comma-separated string."""

    text = raw.strip()
    for loader in (json.loads, ast.literal_eval):
        try:
            parsed = loader(text)
        except Exception:  # noqa: BLE001 - fallback parsing below
            continue
        if isinstance(parsed, Sequence) and not isinstance(parsed, (str, bytes)):
            return [float(value) for value in parsed]

    separators = "," if "," in text else None
    tokens = [tok.strip() for tok in text.split(separators) if tok.strip()]
    if not tokens:
        raise ValueError("No probabilities provided.")
    return [float(tok) for tok in tokens]


def _default_symbols(count: int) -> List[str]:
    """Generate symbol labels A, B, ... , Z, A1, B1, ... as needed."""

    base = [chr(code) for code in range(ord("A"), ord("Z") + 1)]
    symbols: List[str] = []
    suffix = 0
    while len(symbols) < count:
        for letter in base:
            label = letter if suffix == 0 else f"{letter}{suffix}"
            symbols.append(label)
            if len(symbols) == count:
                break
        suffix += 1
    return symbols


def _format_node_label(node: Dict[str, object]) -> str:
    weight = node.get("weight")
    depth = node.get("depth")
    if weight is None:
        return str(node["id"])
    return f"{node['id']} (w={float(weight):.4f}, depth={depth})"


def _render_ascii_tree(root_id: str, nodes: Dict[str, Dict[str, object]]) -> str:
    """Render the exported JSON tree as ASCII art."""

    lines: List[str] = []
    root = nodes[root_id]
    lines.append(_format_node_label(root))

    def _walk(node_id: str, prefix: str, is_tail: bool) -> None:
        node = nodes[node_id]
        label = _format_node_label(node)
        connector = "└── " if is_tail else "├── "
        lines.append(f"{prefix}{connector}{label}")

        children = [
            child_id
            for child_id in (node.get("left"), node.get("right"))
            if child_id is not None
        ]
        for idx, child in enumerate(children):
            child_prefix = prefix + ("    " if is_tail else "│   ")
            _walk(child, child_prefix, idx == len(children) - 1)

    children = [
        child_id
        for child_id in (root.get("left"), root.get("right"))
        if child_id is not None
    ]
    for idx, child in enumerate(children):
        _walk(child, "", idx == len(children) - 1)

    return "\n".join(lines)


def _build_probability_dict(symbols: Iterable[str], probs: List[float]) -> Dict[str, float]:
    total = sum(probs)
    if total <= 0:
        raise ValueError("Probabilities must sum to a positive value.")

    normalized = [value / total for value in probs]
    return {symbol: prob for symbol, prob in zip(symbols, normalized)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a limited-depth Huffman tree, export its JSON, and print an ASCII visualization."
    )
    parser.add_argument(
        "probabilities",
        help="Probability list. Accepts JSON/pythonic syntax or comma-separated floats, e.g. \"[0.8,0.1,0.1]\".",
    )
    parser.add_argument(
        "depth",
        type=int,
        help="Maximum depth constraint for the Huffman tree.",
    )
    parser.add_argument(
        "--symbols",
        nargs="*",
        help="Optional explicit symbol labels. Defaults to A, B, C, ...",
    )
    parser.add_argument(
        "--json-output",
        default="limited_depth_tree.json",
        help="Destination file for the exported tree JSON.",
    )

    args = parser.parse_args()

    probabilities = _parse_probability_list(args.probabilities)
    if args.depth < 1:
        raise ValueError("Depth must be at least 1.")
    if not probabilities:
        raise ValueError("At least one probability value is required.")

    symbols = args.symbols if args.symbols else _default_symbols(len(probabilities))
    if len(symbols) != len(probabilities):
        raise ValueError("Number of symbols must match number of probabilities.")

    prob_dict = _build_probability_dict(symbols, probabilities)
    prob_dist = ProbabilityDist(prob_dict)

    encoder = LimitedDepthHuffmanEncoder(prob_dist, args.depth)

    output_path = Path(args.json_output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    encoder.export_tree_json(str(output_path), symbol_weights=prob_dict)
    print(f"Exported tree JSON to {output_path}")

    with output_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    nodes = payload.get("tree")
    if not nodes:
        raise ValueError("Tree export did not produce any nodes.")

    node_lookup = {node["id"]: node for node in nodes}
    root_id = nodes[0]["id"]
    ascii_tree = _render_ascii_tree(root_id, node_lookup)
    print("\nASCII tree:\n")
    print(ascii_tree)


if __name__ == "__main__":
    main()
