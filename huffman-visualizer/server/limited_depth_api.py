"""
Simple FastAPI service exposing limited-depth Huffman encode/decode endpoints.

This file mirrors the API helpers built alongside the python codec so the
frontend repo can host the backend service directly.

Run locally from the huffman-visualizer root:

    uvicorn server.limited_depth_api:app --reload --port 8080
"""

from __future__ import annotations

import base64
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCL_PATH = REPO_ROOT / "stanford_compression_library-main"
if str(SCL_PATH) not in sys.path:
    sys.path.insert(0, str(SCL_PATH))

try:
    from fastapi import FastAPI, File, HTTPException, UploadFile
    from fastapi.responses import JSONResponse
except ImportError as exc:  # pragma: no cover - dependency hint
    raise ImportError(
        "FastAPI is required for the HTTP service. Install it via `pip install fastapi uvicorn`."
    ) from exc

from scl.compressors.limited_depth_image_codec import decode_image, encode_image

app = FastAPI(title="Limited Depth Huffman Image Codec")


def _base64_encode(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


@app.post("/encode")
async def encode_endpoint(
    image: UploadFile = File(...),
    quality: int = 75,
    max_depth: int = 16,
    lossless: bool = False,
) -> JSONResponse:
    payload = await image.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Empty image payload")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        input_path = tmpdir_path / (image.filename or "upload.img")
        output_path = tmpdir_path / "compressed.ldhc"
        input_path.write_bytes(payload)

        header = encode_image(
            input_path,
            output_path,
            quality=quality,
            max_depth=max_depth,
            lossless=lossless,
        )

        raw_channels = len(header["component_order"])
        raw_bytes = header["width"] * header["height"] * raw_channels
        compressed_bytes = output_path.stat().st_size
        ldch_bytes = output_path.read_bytes()

    response = {
        "filename": image.filename,
        "raw_bytes": raw_bytes,
        "raw_mebibytes": raw_bytes / (1024**2),
        "compressed_bytes": compressed_bytes,
        "compressed_mebibytes": compressed_bytes / (1024**2),
        "lossless": lossless,
        "quality": quality,
        "max_depth": max_depth,
        "message": (
            f"Raw YCbCr data size: {raw_bytes} bytes ({raw_bytes / (1024**2):.2f} MiB) | "
            f"Compressed container: {compressed_bytes} bytes ({compressed_bytes / (1024**2):.2f} MiB)"
        ),
        "ldhc_base64": _base64_encode(ldch_bytes),
    }
    return JSONResponse(response)


@app.post("/decode")
async def decode_endpoint(bitstream: UploadFile = File(...)) -> JSONResponse:
    payload = await bitstream.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Empty bitstream payload")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        ldch_path = tmpdir_path / (bitstream.filename or "upload.ldhc")
        output_path = tmpdir_path / "decoded.jpg"
        ldch_path.write_bytes(payload)

        header = decode_image(ldch_path, output_path)
        jpeg_bytes = output_path.read_bytes()

    response = {
        "filename": bitstream.filename,
        "width": header["width"],
        "height": header["height"],
        "lossless": header.get("lossless", False),
        "quality": header.get("quality"),
        "decoded_base64": _base64_encode(jpeg_bytes),
        "message": "Decoded image is returned as JPEG; base64 payload can be downloaded by the frontend.",
    }
    return JSONResponse(response)
