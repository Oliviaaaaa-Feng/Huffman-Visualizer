#!/usr/bin/env python3
"""
Baseline experiment: compress an input text file with standard Huffman coding.

Upcoming extension: sweep limited-depth Huffman to compare depth vs compression ratio.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure local SCL library is importable when running from repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCL_PATH = PROJECT_ROOT / "stanford_compression_library-main"
if str(SCL_PATH) not in sys.path:
    sys.path.insert(0, str(SCL_PATH))

from scl.compressors.huffman_coder import HuffmanDecoder, HuffmanEncoder, HuffmanTree
from scl.compressors.limited_depth_huffman import (
    LimitedDepthHuffmanDecoder,
    LimitedDepthHuffmanEncoder,
)
from scl.core.data_block import DataBlock
from scl.core.prob_dist import ProbabilityDist

# Default depth sweep for limited-depth Huffman (must be >= ceil(log2(|alphabet|))).
DEFAULT_DEPTHS = [81, 80, 70, 60, 50, 40, 30, 20, 10]


def load_text_as_block(path: Path) -> DataBlock:
    """Read text file and return characters as a DataBlock."""
    content = path.read_text(encoding="utf-8")
    return DataBlock(list(content))


def build_prob_dist(data_block: DataBlock) -> ProbabilityDist:
    """Compute empirical symbol probabilities for the data block."""
    counts = data_block.get_counts()
    total = data_block.size
    prob_dict: Dict[str, float] = {sym: count / total for sym, count in counts.items()}
    return ProbabilityDist(prob_dict)


def huffman_compress_bits(
    data_block: DataBlock, prob_dist: ProbabilityDist, verify: bool = False
) -> Tuple[int, float]:
    """Encode with naive Huffman; optionally verify lossless decode.

    Returns:
        total_bits: length of compressed bitstream.
        avg_bits_per_symbol: total_bits / num_symbols.
    """
    encoder = HuffmanEncoder(prob_dist)
    bitstream = encoder.encode_block(data_block)
    total_bits = len(bitstream)
    avg_bits = total_bits / data_block.size

    if verify:
        decoder = HuffmanDecoder(prob_dist)
        decoded_block, bits_used = decoder.decode_block(bitstream)
        if bits_used != total_bits:
            raise ValueError(f"Decoder consumed {bits_used} bits, expected {total_bits}")
        if decoded_block.data_list != data_block.data_list:
            raise ValueError("Decoded data does not match original input")

    return total_bits, avg_bits


def format_ratio(compressed_bits: int, raw_bits: int) -> str:
    ratio = compressed_bits / raw_bits
    savings = 1.0 - ratio
    return f"{ratio:.4f} (savings {savings*100:.2f}%)"


def run_baseline(input_path: Path, verify: bool = True) -> None:
    data_block = load_text_as_block(input_path)
    raw_bits = data_block.size * 8  # assume 1 byte per character
    prob_dist = build_prob_dist(data_block)

    compressed_bits, avg_bits = huffman_compress_bits(data_block, prob_dist, verify=verify)
    ratio_str = format_ratio(compressed_bits, raw_bits)

    print(f"Input: {input_path}")
    print(f"Symbols: {data_block.size}")
    print(f"Alphabet size: {len(prob_dist.alphabet)}")
    print(f"Empirical entropy: {prob_dist.entropy:.4f} bits/symbol")
    print(f"Huffman avg bits/symbol: {avg_bits:.4f}")
    print(f"Raw size (bits): {raw_bits}")
    print(f"Compressed size (bits): {compressed_bits}")
    print(f"Compression ratio (compressed/raw): {ratio_str}")

    # Dump summary and full code table to a file for later use.
    stem = input_path.stem
    output_path = Path(f"experiment/output/huffman_baseline_{stem}.txt")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tree = HuffmanTree(prob_dist)
    encoding_table = tree.get_encoding_table()
    lengths = {sym: len(code) for sym, code in encoding_table.items()}

    def _fmt_symbol(sym: str) -> str:
        # Keep control characters readable.
        return sym.encode("unicode_escape").decode("ascii")

    with output_path.open("w", encoding="utf-8") as f:
        f.write("Huffman baseline results\n")
        f.write(f"Input: {input_path}\n")
        f.write(f"Symbols: {data_block.size}\n")
        f.write(f"Alphabet size: {len(prob_dist.alphabet)}\n")
        f.write(f"Empirical entropy: {prob_dist.entropy:.4f} bits/symbol\n")
        f.write(f"Huffman avg bits/symbol: {avg_bits:.4f}\n")
        f.write(f"Raw size (bits): {raw_bits}\n")
        f.write(f"Compressed size (bits): {compressed_bits}\n")
        f.write(f"Compression ratio (compressed/raw): {ratio_str}\n")
        f.write(f"Min code length: {min(lengths.values())}\n")
        f.write(f"Max code length: {max(lengths.values())}\n\n")
        f.write("Code table (symbol -> bits [len]):\n")

        for sym, code in sorted(encoding_table.items(), key=lambda kv: (len(kv[1]), kv[0])):
            bitstring = code.to01()
            f.write(f"{_fmt_symbol(sym)} -> {bitstring} [{len(code)}]\n")

    print(f"Wrote summary and code table to {output_path}")
    return {
        "raw_bits": raw_bits,
        "compressed_bits": compressed_bits,
        "avg_bits_per_symbol": avg_bits,
        "ratio": compressed_bits / raw_bits,
        "data_block": data_block,
        "prob_dist": prob_dist,
    }


def limited_depth_compress_bits(
    data_block: DataBlock,
    prob_dist: ProbabilityDist,
    max_depth: int,
    verify: bool = False,
) -> Tuple[int, float, dict]:
    encoder = LimitedDepthHuffmanEncoder(prob_dist, max_depth=max_depth)
    bitstream = encoder.encode_block(data_block)
    total_bits = len(bitstream)
    avg_bits = total_bits / data_block.size

    if verify:
        decoder = LimitedDepthHuffmanDecoder(prob_dist, max_depth=max_depth)
        decoded_block, bits_used = decoder.decode_block(bitstream)
        if bits_used != total_bits:
            raise ValueError(f"[depth={max_depth}] Decoder consumed {bits_used} bits, expected {total_bits}")
        if decoded_block.data_list != data_block.data_list:
            raise ValueError(f"[depth={max_depth}] Decoded data does not match original input")

    lengths = {sym: len(code) for sym, code in encoder.encoding_table.items()}
    return total_bits, avg_bits, lengths


def run_depth_sweep(input_path: Path, depths: List[int], verify: bool = True) -> None:
    data_block = load_text_as_block(input_path)
    raw_bits = data_block.size * 8
    prob_dist = build_prob_dist(data_block)

    stem = input_path.stem
    output_dir = Path(f"experiment/output/depth_runs_{stem}")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for depth in depths:
        compressed_bits, avg_bits, lengths = limited_depth_compress_bits(
            data_block, prob_dist, max_depth=depth, verify=verify
        )
        ratio_str = format_ratio(compressed_bits, raw_bits)
        results.append(
            {
                "depth": depth,
                "compressed_bits": compressed_bits,
                "avg_bits_per_symbol": avg_bits,
                "ratio": compressed_bits / raw_bits,
                "min_len": min(lengths.values()),
                "max_len": max(lengths.values()),
            }
        )

        output_path = output_dir / f"depth_{depth}.txt"

        def _fmt_symbol(sym: str) -> str:
            return sym.encode("unicode_escape").decode("ascii")

        with output_path.open("w", encoding="utf-8") as f:
            f.write(f"Limited-depth Huffman results (max_depth={depth})\n")
            f.write(f"Input: {input_path}\n")
            f.write(f"Symbols: {data_block.size}\n")
            f.write(f"Alphabet size: {len(prob_dist.alphabet)}\n")
            f.write(f"Empirical entropy: {prob_dist.entropy:.4f} bits/symbol\n")
            f.write(f"Max depth constraint: {depth}\n")
            f.write(f"Avg bits/symbol: {avg_bits:.4f}\n")
            f.write(f"Raw size (bits): {raw_bits}\n")
            f.write(f"Compressed size (bits): {compressed_bits}\n")
            f.write(f"Compression ratio (compressed/raw): {ratio_str}\n")
            f.write(f"Min code length: {min(lengths.values())}\n")
            f.write(f"Max code length: {max(lengths.values())}\n\n")
            f.write("Code table (symbol -> bits [len]):\n")
            table = LimitedDepthHuffmanEncoder(prob_dist, max_depth=depth).encoding_table
            for sym, code in sorted(table.items(), key=lambda kv: (len(kv[1]), kv[0])):
                bitstring = code.to01()
                f.write(f"{_fmt_symbol(sym)} -> {bitstring} [{len(code)}]\n")

        print(f"[depth={depth}] Avg bits/symbol: {avg_bits:.4f}, ratio: {ratio_str}")
        print(f"[depth={depth}] Wrote summary and code table to {output_path}")
    return results


def plot_depth_vs_ratio(results: List[dict], baseline_ratio: float, label: str) -> None:
    """Plot compression ratio vs depth and save to output/plot using matplotlib."""
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:
        print("matplotlib not available; install it to generate plots (pip install matplotlib).")
        return

    if not results:
        print("No depth results to plot.")
        return

    results_sorted = sorted(results, key=lambda r: r["depth"])
    depths = [r["depth"] for r in results_sorted]
    ratios = [r["ratio"] for r in results_sorted]

    plot_dir = Path("experiment/output/plot")
    plot_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(depths, ratios, marker="o", label="Limited-depth Huffman", linewidth=2)
    plt.axhline(baseline_ratio, color="red", linestyle="--", label="Baseline Huffman", linewidth=2)
    plt.xlabel("Depth limit")
    plt.ylabel("Compression ratio (compressed/raw)")
    plt.title(f"Depth limit vs compression ratio ({label})")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    # Annotate
    for d, r in zip(depths, ratios):
        plt.annotate(f"{r:.3f}", (d, r), textcoords="offset points", xytext=(0, -12), ha="center")

    out_path = plot_dir / f"depth_vs_ratio_{label}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved plot to {out_path}")


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Baseline and limited-depth Huffman compression experiments."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("experiment/data/sherlock_ascii.txt"),
        help="Path to the input text file (default: experiment/data/sherlock_ascii.txt).",
    )
    parser.add_argument(
        "--depths",
        type=str,
        default=",".join(str(d) for d in DEFAULT_DEPTHS),
        help=f"Comma-separated depth limits to run (default: {DEFAULT_DEPTHS}).",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip decode verification (enabled by default).",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    verify = not args.no_verify
    depths = [int(d.strip()) for d in args.depths.split(",") if d.strip()]
    stem = args.input.stem

    baseline = run_baseline(args.input, verify=verify)
    depth_results = run_depth_sweep(args.input, depths=depths, verify=verify)
    plot_depth_vs_ratio(depth_results, baseline_ratio=baseline["ratio"], label=stem)


if __name__ == "__main__":
    main()
