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

You can experience the visualization tool in two ways: via our deployed website or by running it locally.

### Option 1: Live Demo
Access the deployed version directly in your browser:
**[https://huffman-visualizer-three.vercel.app/](https://huffman-visualizer-three.vercel.app/)**

*(Note: The server is hosted on a free instance, so it might take a minute to wake up upon the first request.)*

---

### Option 2: Run Locally
To run the project locally, you will need to start the **Backend** and **Frontend** services separately in two terminal windows.

#### 1. Start the Backend Server
The backend handles the Huffman algorithms.

```bash
# In Terminal 1: Navigate to the backend directory
cd stanford_compression_library-main

# Install Python dependencies
pip install fastapi uvicorn numpy pillow python-multipart

# Start the API server
python -m uvicorn app:app --reload --port 8000
```
The backend will start running at http://127.0.0.1:8000

#### 2. Start the Frontend Application
The frontend provides the interactive web interface.

```bash
# In Terminal 2: Navigate to the frontend directory
cd huffman-visualizer

# Install Node.js dependencies
npm install

# Start the development server
npm run dev
```
The frontend will start running at http://localhost:5173 (or similar). Open this link in your browser to use the tool.

> **Configuration Note:**
> By default, the frontend is configured to point to the deployed production server. If you want it to communicate with your **local backend**, please open `huffman-visualizer/src/App.tsx` and update the `API_BASE` variable:
>
> ```typescript
> // huffman-visualizer/src/App.tsx
>
> // Change this line:
> // const API_BASE = 'https://huffman-visualizer.onrender.com'
>
> // To this:
> const API_BASE = 'http://127.0.0.1:8000'
> ```
