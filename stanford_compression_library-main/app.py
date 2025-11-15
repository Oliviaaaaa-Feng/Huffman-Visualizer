from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict
import json
import ast

from scl.compressors.limited_depth_huffman import LimitedDepthHuffmanEncoder
from scl.core.prob_dist import ProbabilityDist
from scl.compressors.vitter_adaptive_huffman import VitterAdaptiveHuffmanEncoder

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_headers=["*"],
    allow_methods=["*"],
)


def _parse_prob_list(raw: str) -> List[float]:
    """Parse a probability list from JSON / Python literal / comma-separated string."""
    raw = raw.strip()
    for loader in (json.loads, ast.literal_eval):
        try:
            v = loader(raw)
            if isinstance(v, (list, tuple)):
                return [float(x) for x in v]
        except Exception:
            pass
    toks = [t.strip() for t in raw.split(",") if t.strip()]
    if not toks:
        raise ValueError("empty probabilities")
    return [float(t) for t in toks]


def _default_symbols(n: int) -> List[str]:
    """Generate default symbol labels: A..Z, A1..Z1, ..."""
    base = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
    out: List[str] = []
    suf = 0
    while len(out) < n:
        for ch in base:
            out.append(ch if suf == 0 else f"{ch}{suf}")
            if len(out) == n:
                break
        suf += 1
    return out


@app.post("/api/limited-depth/preview")
def limited_depth_preview(
    probabilities: str = Form(...),
    lmax: int = Form(...),
):
    """Build a limited-depth Huffman tree and return it as JSON (no animation)."""
    if lmax < 1:
        return JSONResponse({"error": "lmax must be >= 1"}, status_code=400)

    probs = _parse_prob_list(probabilities)
    if not probs:
        return JSONResponse({"error": "empty probs"}, status_code=400)

    syms = _default_symbols(len(probs))
    total = sum(probs)
    probs = [p / total for p in probs]
    prob_dict: Dict[str, float] = {s: p for s, p in zip(syms, probs)}

    enc = LimitedDepthHuffmanEncoder(ProbabilityDist(prob_dict), lmax)

    payload = enc.export_tree_json(symbol_weights=prob_dict)

    return JSONResponse(payload)


@app.post("/api/adaptive/preview")
def adaptive_preview(
    text: str = Form(...),
):
    """
    Adaptive Huffman preview:
    - Input: text (utf-8)
    - Output: steps array with tree snapshot after each symbol,
      plus encoded bitstream.
    """
    data = text.encode("utf-8")

    encoder = VitterAdaptiveHuffmanEncoder()
    trace = encoder.encode_with_trace(data)

    # Prettify node ids: if id is "65", convert to "A"
    for step in trace.get("steps", []):
        id_map = {}
        for node in step.get("tree", []):
            nid = node.get("id")
            if isinstance(nid, str) and nid.isdigit():
                val = int(nid)
                if 32 <= val <= 126:
                    new_id = chr(val)
                    id_map[nid] = new_id
                    node["id"] = new_id
                else:
                    id_map[nid] = nid
            else:
                if isinstance(nid, str):
                    id_map[nid] = nid

        for node in step.get("tree", []):
            left = node.get("left")
            right = node.get("right")
            if isinstance(left, str) and left in id_map:
                node["left"] = id_map[left]
            if isinstance(right, str) and right in id_map:
                node["right"] = id_map[right]

        if "root" in step and isinstance(step["root"], str) and step["root"] in id_map:
            step["root"] = id_map[step["root"]]

    return JSONResponse(trace)
