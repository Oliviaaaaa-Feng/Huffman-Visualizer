"""
Offline image compressor/decompressor that reuses the limited-depth Huffman code.

The encoder mirrors the JPEG pipeline up to the entropy stage:
  * YCbCr conversion + block based DCT
  * Quantization using standard tables (configurable quality)
  * Zigzag/RLE producing DC difference categories and AC (run, size) symbols
  * Canonical Huffman tables produced via the limited-depth package-merge code

The resulting bitstreams are stored inside a tiny custom container so we can
reconstruct the image later without touching the JPEG container format.

Usage:
    python -m scl.compressors.limited_depth_image_codec encode lena.png lena.ldhc --quality 85
    python -m scl.compressors.limited_depth_image_codec decode lena.ldhc lena_recon.png
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover - dependency hint
    raise ImportError(
        "Pillow is required for image I/O. Install it via `pip install pillow`."
    ) from exc

from scl.compressors.limited_depth_huffman import (
    LimitedDepthHuffmanEncoder,
    _build_canonical_codes,
)
from scl.compressors.prefix_free_compressors import PrefixFreeTree
from scl.core.prob_dist import ProbabilityDist
from scl.utils.bitarray_utils import BitArray, bitarray_to_uint, uint_to_bitarray

BLOCK_SIZE = 8
MAGIC = b"LDHC"
CONTAINER_VERSION = 1
COMPONENTS = ("Y", "Cb", "Cr")

BASE_LUMA_TABLE = np.array(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ],
    dtype=np.int32,
)

BASE_CHROMA_TABLE = np.array(
    [
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
    ],
    dtype=np.int32,
)

ZIGZAG_ORDER = np.array(
    [
        0,
        1,
        8,
        16,
        9,
        2,
        3,
        10,
        17,
        24,
        32,
        25,
        18,
        11,
        4,
        5,
        12,
        19,
        26,
        33,
        40,
        48,
        41,
        34,
        27,
        20,
        13,
        6,
        7,
        14,
        21,
        28,
        35,
        42,
        49,
        56,
        57,
        50,
        43,
        36,
        29,
        22,
        15,
        23,
        30,
        37,
        44,
        51,
        58,
        59,
        52,
        45,
        38,
        31,
        39,
        46,
        53,
        60,
        61,
        54,
        47,
        55,
        62,
        63,
    ],
    dtype=np.int32,
)

INV_ZIGZAG_ORDER = np.argsort(ZIGZAG_ORDER)


class CodecError(RuntimeError):
    """Raised when the container or bitstreams are malformed."""


def _create_dct_matrix(n: int) -> np.ndarray:
    matrix = np.zeros((n, n), dtype=np.float64)
    factor = math.pi / (2 * n)
    for k in range(n):
        alpha = math.sqrt(1 / n) if k == 0 else math.sqrt(2 / n)
        for i in range(n):
            matrix[k, i] = alpha * math.cos((2 * i + 1) * k * factor)
    return matrix


DCT_MATRIX = _create_dct_matrix(BLOCK_SIZE)


@dataclass
class HuffmanSpec:
    symbol_order: List[int]
    code_lengths: Dict[int, int]


def _quality_to_tables(quality: int) -> Dict[str, np.ndarray]:
    quality = max(1, min(quality, 100))
    if quality < 50:
        scale = 5000 / quality
    else:
        scale = 200 - quality * 2

    def _scale_table(base: np.ndarray) -> np.ndarray:
        scaled = np.floor((base * scale + 50) / 100).astype(np.int32)
        scaled[scaled < 1] = 1
        scaled[scaled > 255] = 255
        return scaled

    return {
        "Y": _scale_table(BASE_LUMA_TABLE),
        "Cb": _scale_table(BASE_CHROMA_TABLE),
        "Cr": _scale_table(BASE_CHROMA_TABLE),
    }


def _lossless_tables() -> Dict[str, np.ndarray]:
    identity = np.ones_like(BASE_LUMA_TABLE, dtype=np.int32)
    return {
        "Y": identity.copy(),
        "Cb": identity.copy(),
        "Cr": identity.copy(),
    }


def _pad_channel(channel: np.ndarray) -> np.ndarray:
    h, w = channel.shape
    pad_h = (BLOCK_SIZE - (h % BLOCK_SIZE)) % BLOCK_SIZE
    pad_w = (BLOCK_SIZE - (w % BLOCK_SIZE)) % BLOCK_SIZE
    if pad_h == 0 and pad_w == 0:
        return channel
    return np.pad(channel, ((0, pad_h), (0, pad_w)), mode="edge")


def _channel_to_blocks(channel: np.ndarray) -> np.ndarray:
    h, w = channel.shape
    channel = _pad_channel(channel)
    padded_h, padded_w = channel.shape
    blocks = (
        channel.reshape(padded_h // BLOCK_SIZE, BLOCK_SIZE, padded_w // BLOCK_SIZE, BLOCK_SIZE)
        .swapaxes(1, 2)
        .reshape(-1, BLOCK_SIZE, BLOCK_SIZE)
    )
    return blocks


def _blocks_to_channel(blocks: np.ndarray, padded_h: int, padded_w: int) -> np.ndarray:
    rows = padded_h // BLOCK_SIZE
    cols = padded_w // BLOCK_SIZE
    channel = np.zeros((rows * BLOCK_SIZE, cols * BLOCK_SIZE), dtype=np.float64)
    idx = 0
    for r in range(rows):
        for c in range(cols):
            channel[
                r * BLOCK_SIZE : (r + 1) * BLOCK_SIZE,
                c * BLOCK_SIZE : (c + 1) * BLOCK_SIZE,
            ] = blocks[idx]
            idx += 1
    return channel[:padded_h, :padded_w]


def _zigzag(block: np.ndarray) -> np.ndarray:
    return block.reshape(-1)[ZIGZAG_ORDER]


def _inverse_zigzag(vector: np.ndarray) -> np.ndarray:
    # Map values from zigzag order back to row-major order.
    return vector[INV_ZIGZAG_ORDER]


def _dct(block: np.ndarray) -> np.ndarray:
    return DCT_MATRIX @ block @ DCT_MATRIX.T


def _idct(block: np.ndarray) -> np.ndarray:
    return DCT_MATRIX.T @ block @ DCT_MATRIX


def _value_to_category_and_bits(value: int) -> Tuple[int, BitArray]:
    if value == 0:
        return 0, BitArray()

    magnitude = abs(value)
    size = int(math.floor(math.log2(magnitude))) + 1

    if value > 0:
        bits = uint_to_bitarray(magnitude, bit_width=size)
    else:
        bits = uint_to_bitarray((1 << size) - 1 + value, bit_width=size)
    return size, bits


def _bits_to_value(bits: BitArray, size: int) -> int:
    if size == 0:
        return 0
    if len(bits) != size:
        raise CodecError(f"Expected {size} amplitude bits, received {len(bits)}")
    value = bitarray_to_uint(bits)
    if bits[0] == 0:
        return value - ((1 << size) - 1)
    return value


def _scaled_probability(counter: Counter) -> ProbabilityDist:
    if not counter:
        raise CodecError("No symbols collected for Huffman coding")

    total = sum(counter.values())
    scale = max(1, math.ceil(total / 1_000_000))
    scaled_counts = {sym: max(1, count // scale) for sym, count in counter.items()}
    scaled_total = float(sum(scaled_counts.values()))
    prob_dict = {sym: count / scaled_total for sym, count in scaled_counts.items()}
    return ProbabilityDist(prob_dict)


def _build_huffman_spec(counter: Counter, max_depth: int) -> Tuple[HuffmanSpec, LimitedDepthHuffmanEncoder]:
    prob_dist = _scaled_probability(counter)
    encoder = LimitedDepthHuffmanEncoder(prob_dist, max_depth=max_depth)
    code_lengths = {symbol: len(code) for symbol, code in encoder.encoding_table.items()}
    return HuffmanSpec(symbol_order=[int(s) for s in prob_dist.alphabet], code_lengths=code_lengths), encoder


def _serialize_lengths(lengths: Dict[int, int]) -> List[List[int]]:
    return [[int(symbol), int(lengths[symbol])] for symbol in sorted(lengths)]


def _deserialize_lengths(entries: Iterable[Sequence[int]]) -> Dict[int, int]:
    result: Dict[int, int] = {}
    for symbol, length in entries:
        result[int(symbol)] = int(length)
    return result


def _tree_from_spec(spec: HuffmanSpec) -> PrefixFreeTree:
    codes = _build_canonical_codes(spec.code_lengths, spec.symbol_order)
    return PrefixFreeTree.build_prefix_free_tree_from_code(codes)


def encode_image(
    input_path: Path,
    output_path: Path,
    *,
    quality: int = 75,
    max_depth: int = 16,
    lossless: bool = False,
) -> Dict:
    with Image.open(input_path) as image:
        ycbcr = np.asarray(image.convert("YCbCr"), dtype=np.float64)

    height, width, _ = ycbcr.shape
    padded_height = int(math.ceil(height / BLOCK_SIZE) * BLOCK_SIZE)
    padded_width = int(math.ceil(width / BLOCK_SIZE) * BLOCK_SIZE)
    quant_tables = _lossless_tables() if lossless else _quality_to_tables(quality)

    dc_symbols: List[int] = []
    dc_values: List[int] = []
    ac_symbols: List[int] = []
    ac_values: List[int] = []
    blocks_per_component: Dict[str, int] = {}

    for idx, component in enumerate(COMPONENTS):
        channel = ycbcr[:, :, idx].copy()
        if not lossless:
            channel -= 128.0
        blocks = _channel_to_blocks(channel)
        prev_dc = 0
        quant_table = quant_tables[component].astype(np.float64)

        blocks_per_component[component] = int(blocks.shape[0])

        for block in blocks:
            if lossless:
                quantized = block.astype(np.int32)
            else:
                dct_coeffs = _dct(block)
                quantized = np.round(dct_coeffs / quant_table).astype(np.int32)
            zigzag = _zigzag(quantized)

            dc_diff = int(zigzag[0] - prev_dc)
            prev_dc = int(zigzag[0])
            category, _ = _value_to_category_and_bits(dc_diff)
            dc_symbols.append(category)
            dc_values.append(dc_diff)

            run = 0
            for coeff in zigzag[1:]:
                coeff = int(coeff)
                if coeff == 0:
                    run += 1
                    if run == 16:
                        ac_symbols.append(0xF0)
                        ac_values.append(0)
                        run = 0
                    continue

                while run > 15:
                    ac_symbols.append(0xF0)
                    ac_values.append(0)
                    run -= 16

                size, _ = _value_to_category_and_bits(coeff)
                symbol = (run << 4) | size
                ac_symbols.append(symbol)
                ac_values.append(coeff)
                run = 0

            if run > 0:
                ac_symbols.append(0x00)
                ac_values.append(0)

    if not dc_symbols or not ac_symbols:
        raise CodecError("Input image did not generate any symbols")

    dc_counter = Counter(dc_symbols)
    ac_counter = Counter(ac_symbols)

    dc_spec, dc_encoder = _build_huffman_spec(dc_counter, max_depth)
    ac_spec, ac_encoder = _build_huffman_spec(ac_counter, max_depth)

    dc_stream = BitArray()
    for symbol, value in zip(dc_symbols, dc_values):
        dc_stream.extend(dc_encoder.encoding_table[symbol])
        _, amplitude_bits = _value_to_category_and_bits(value)
        dc_stream.extend(amplitude_bits)

    ac_stream = BitArray()
    for symbol, value in zip(ac_symbols, ac_values):
        ac_stream.extend(ac_encoder.encoding_table[symbol])
        if symbol not in (0x00, 0xF0):
            _, amplitude_bits = _value_to_category_and_bits(value)
            ac_stream.extend(amplitude_bits)

    dc_bytes = dc_stream.tobytes()
    ac_bytes = ac_stream.tobytes()

    header = {
        "version": CONTAINER_VERSION,
        "width": width,
        "height": height,
        "padded_width": padded_width,
        "padded_height": padded_height,
        "block_size": BLOCK_SIZE,
        "blocks_per_row": padded_width // BLOCK_SIZE,
        "blocks_per_col": padded_height // BLOCK_SIZE,
        "component_order": list(COMPONENTS),
        "blocks_per_component": blocks_per_component,
        "quality": int(quality),
        "max_depth": int(max_depth),
        "lossless": bool(lossless),
        "quant_tables": {
            comp: quant_tables[comp].astype(np.int32).reshape(-1).tolist() for comp in COMPONENTS
        },
        "dc": {
            "symbol_order": dc_spec.symbol_order,
            "code_lengths": _serialize_lengths(dc_spec.code_lengths),
        },
        "ac": {
            "symbol_order": ac_spec.symbol_order,
            "code_lengths": _serialize_lengths(ac_spec.code_lengths),
        },
        "streams": {
            "dc_bits": int(len(dc_stream)),
            "ac_bits": int(len(ac_stream)),
            "dc_bytes": len(dc_bytes),
            "ac_bytes": len(ac_bytes),
        },
    }

    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")

    with open(output_path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack(">I", len(header_bytes)))
        f.write(header_bytes)
        f.write(dc_bytes)
        f.write(ac_bytes)

    return header


def _read_container(path: Path) -> Tuple[Dict, bytes, bytes]:
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != MAGIC:
            raise CodecError("Not a limited-depth Huffman container")
        header_len = struct.unpack(">I", f.read(4))[0]
        header = json.loads(f.read(header_len).decode("utf-8"))
        dc_bytes = f.read(header["streams"]["dc_bytes"])
        ac_bytes = f.read(header["streams"]["ac_bytes"])
    return header, dc_bytes, ac_bytes


def _bitarray_from_bytes(data: bytes, bit_length: int) -> BitArray:
    arr = BitArray()
    arr.frombytes(data)
    if bit_length > len(arr):
        raise CodecError("Declared bit length exceeds payload size")
    if bit_length < len(arr):
        return arr[:bit_length]
    return arr


def _decode_symbol_from_tree(tree: PrefixFreeTree, bits: BitArray, start: int, limit: int) -> Tuple[int, int]:
    node = tree.root_node
    idx = start
    while not node.is_leaf_node:
        if idx >= limit:
            raise CodecError("Ran out of bits while decoding symbol")
        bit = bits[idx]
        node = node.right_child if bit else node.left_child
        idx += 1
    return node.id, idx - start


def decode_image(input_path: Path, output_path: Path) -> Dict:
    header, dc_bytes, ac_bytes = _read_container(input_path)

    lossless = bool(header.get("lossless", False))

    dc_spec = HuffmanSpec(
        symbol_order=[int(s) for s in header["dc"]["symbol_order"]],
        code_lengths=_deserialize_lengths(header["dc"]["code_lengths"]),
    )
    ac_spec = HuffmanSpec(
        symbol_order=[int(s) for s in header["ac"]["symbol_order"]],
        code_lengths=_deserialize_lengths(header["ac"]["code_lengths"]),
    )

    dc_tree = _tree_from_spec(dc_spec)
    ac_tree = _tree_from_spec(ac_spec)

    dc_bits = _bitarray_from_bytes(dc_bytes, header["streams"]["dc_bits"])
    ac_bits = _bitarray_from_bytes(ac_bytes, header["streams"]["ac_bits"])

    dc_idx = 0
    ac_idx = 0

    padded_width = header["padded_width"]
    padded_height = header["padded_height"]
    width = header["width"]
    height = header["height"]

    component_channels: Dict[str, np.ndarray] = {}

    for component in header["component_order"]:
        num_blocks = header["blocks_per_component"][component]
        quant_table = (
            None
            if lossless
            else np.array(header["quant_tables"][component], dtype=np.float64).reshape(
                BLOCK_SIZE, BLOCK_SIZE
            )
        )
        prev_dc = 0
        blocks = np.zeros((num_blocks, BLOCK_SIZE, BLOCK_SIZE), dtype=np.float64)

        for block_idx in range(num_blocks):
            category, consumed = _decode_symbol_from_tree(dc_tree, dc_bits, dc_idx, len(dc_bits))
            dc_idx += consumed
            if category < 0 or category > 11:
                raise CodecError(f"Invalid DC category {category}")
            amplitude_bits = dc_bits[dc_idx : dc_idx + category]
            dc_idx += category
            dc_diff = _bits_to_value(amplitude_bits, category)
            dc_value = prev_dc + dc_diff
            prev_dc = dc_value

            coeffs = np.zeros(64, dtype=np.float64)
            coeffs[0] = dc_value
            pos = 1

            while pos < 64:
                symbol, consumed = _decode_symbol_from_tree(ac_tree, ac_bits, ac_idx, len(ac_bits))
                ac_idx += consumed

                if symbol == 0x00:
                    break
                if symbol == 0xF0:
                    pos += 16
                    continue

                run = symbol >> 4
                size = symbol & 0x0F
                pos += run
                amplitude_bits = ac_bits[ac_idx : ac_idx + size]
                ac_idx += size
                coeff_value = _bits_to_value(amplitude_bits, size)
                if pos >= 64:
                    raise CodecError("AC run exceeded block bounds")
                coeffs[pos] = coeff_value
                pos += 1

            block = _inverse_zigzag(coeffs).reshape(BLOCK_SIZE, BLOCK_SIZE)
            if lossless:
                spatial = block
            else:
                assert quant_table is not None
                dequant = block * quant_table
                spatial = _idct(dequant) + 128.0
            blocks[block_idx] = spatial

        channel = _blocks_to_channel(blocks, padded_height, padded_width)
        component_channels[component] = np.clip(channel, 0, 255)

    stacked = np.stack(
        [component_channels[comp][:height, :width] for comp in header["component_order"]],
        axis=2,
    ).astype(np.uint8)

    rebuilt = Image.fromarray(stacked, mode="YCbCr").convert("RGB")
    rebuilt.save(output_path)
    return header


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Limited-depth Huffman image coder")
    subparsers = parser.add_subparsers(dest="command", required=True)

    enc_parser = subparsers.add_parser("encode", help="Encode an image into the custom container")
    enc_parser.add_argument("input_image", type=Path)
    enc_parser.add_argument("output_file", type=Path)
    enc_parser.add_argument("--quality", type=int, default=75, help="JPEG-like quality factor (1-100)")
    enc_parser.add_argument("--max-depth", type=int, default=16, help="Max Huffman tree depth")
    enc_parser.add_argument(
        "--lossless",
        action="store_true",
        help="Skip DCT/quantization and encode raw YCbCr blocks for perfect reconstruction",
    )

    dec_parser = subparsers.add_parser("decode", help="Decode a container back into an image")
    dec_parser.add_argument("input_file", type=Path)
    dec_parser.add_argument("output_image", type=Path)

    return parser.parse_args()


def main():
    args = _parse_args()

    if args.command == "encode":
        header = encode_image(
            args.input_image,
            args.output_file,
            quality=args.quality,
            max_depth=args.max_depth,
            lossless=args.lossless,
        )
        raw_channels = len(header["component_order"])
        raw_bytes = header["width"] * header["height"] * raw_channels
        compressed_bytes = args.output_file.stat().st_size
        print(
            f"Raw YCbCr data size: {raw_bytes} bytes "
            f"({raw_bytes / (1024**2):.2f} MiB) | "
            f"Compressed container: {compressed_bytes} bytes "
            f"({compressed_bytes / (1024**2):.2f} MiB)"
        )
    elif args.command == "decode":
        decode_image(args.input_file, args.output_image)


if __name__ == "__main__":
    main()
