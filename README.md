# Huffman Visualizer

This repo extends the original Stanford Compression Library (SCL) with limited-depth and adaptive Huffman tooling, plus a frontend visualizer.

## Repository Layout
- `stanford_compression_library-main/` – upstream SCL code. Our additions live in `stanford_compression_library-main/scl/compressors/`:
  - `limited_depth_huffman.py` – package-merge implementation that enforces a maximum codeword depth. Exposes `_compute_code_lengths`, `LimitedDepthHuffmanEncoder`, and `LimitedDepthHuffmanDecoder`, and can export trees for visualization.
  - `limited_depth_image_codec.py` – offline JPEG-style image coder that reuses the limited-depth Huffman tables. CLI examples:
    - `python -m scl.compressors.limited_depth_image_codec encode input.jpg output.ldhc --quality 85`
    - `python -m scl.compressors.limited_depth_image_codec decode output.ldhc recon.png`
  - `vitter_adaptive_huffman.py` – streaming adaptive Huffman coder using Vitter’s algorithm with linked-list block maintenance for incremental updates.
  - `test_limited_depth_huffman.py` – unit tests that validate depth bounds, Kraft equality, canonical code generation, and round-trip encoding/decoding for the limited-depth coder.
- `experiment/` – scripts and data for quick experiments (e.g., `experiment/huffman_depth_experiment.py` to compare standard vs limited-depth Huffman on text inputs).
- `huffman-visualizer/` – frontend for visualizing Huffman trees and encoding steps (Vite/TypeScript). Run with `cd huffman-visualizer && npm install && npm run dev`.

## Environment
- Python 3.10+ recommended.
- Install Python dependencies (SCL requirements plus Pillow for the image codec):
```
pip install -r requirements.txt
```

## Quick Start (Visualizer)
1) From repo root: `cd huffman-visualizer`
2) Install deps and start dev server:
```
npm install
npm run dev
```