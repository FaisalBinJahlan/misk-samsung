"""Build the kids word index for the expanded recognizer.

Reads tools/kids-words.txt, embeds every word with the MobileCLIP-S0 text
encoder (an average over several photo prompts), and writes the gzip'd label
list and int8 index into vendor/expanded-lite, updating assets.json.

Usage:
  python tools/build_index.py <folder with text_model.onnx and tokenizer.json>

The text encoder comes from https://huggingface.co/Xenova/mobileclip_s0
(onnx/text_model.onnx and tokenizer.json). It is only needed here, not in the app.
Requires: numpy, onnxruntime, tokenizers.
"""
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
WORDS = ROOT / "tools" / "kids-words.txt"
OUT = ROOT / "vendor" / "expanded-lite"
PART_BYTES = 512 * 1024
TEMPLATES = [
    "a photo of a {}.",
    "a photo of the {}.",
    "a close-up photo of a {}.",
    "a photo of a small {}.",
    "a photo of a large {}.",
    "a bright photo of a {}.",
    "a dark photo of the {}.",
    "a blurry photo of a {}.",
    "a cropped photo of the {}.",
    "a photo of my {}.",
]


def read_words():
    labels, prompts, levels = [], [], []
    level = None
    for raw in WORDS.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("## level"):
            level = int(line.split()[-1])
            continue
        if not line or line.startswith("#"):
            continue
        word, _, prompt = (part.strip() for part in line.partition("|"))
        labels.append(word)
        prompts.append(prompt or word)
        levels.append(level)
    if len(set(labels)) != len(labels):
        raise SystemExit("Duplicate words in kids-words.txt")
    order = sorted(range(len(labels)), key=lambda i: levels[i])
    return [labels[i] for i in order], [prompts[i] for i in order], [levels[i] for i in order]


def embed(texts, tokenizer, session, batch=64):
    pad = tokenizer.token_to_id("!") or 0
    out = []
    for start in range(0, len(texts), batch):
        ids = [tokenizer.encode(t).ids[:77] for t in texts[start:start + batch]]
        ids = np.array([row + [pad] * (77 - len(row)) for row in ids], np.int64)
        vectors = session.run(None, {"input_ids": ids})[0]
        out.append(vectors / np.linalg.norm(vectors, axis=1, keepdims=True))
    return np.concatenate(out)


def write_parts(name, data, assets, key):
    for old in OUT.glob(f"{name}-*.bin"):
        old.unlink()
    packed = gzip.compress(data, compresslevel=9, mtime=0)
    parts = []
    for i in range(0, len(packed), PART_BYTES):
        part = f"{name}-{i // PART_BYTES:02d}.bin"
        (OUT / part).write_bytes(packed[i:i + PART_BYTES])
        parts.append({"name": part, "bytes": len(packed[i:i + PART_BYTES])})
    assets[key] = {"bytes": len(packed), "expandedBytes": len(data), "parts": parts}


def main():
    model_dir = Path(sys.argv[1])
    labels, prompts, levels = read_words()
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    session = ort.InferenceSession(str(model_dir / "text_model.onnx"))

    total = np.zeros((len(labels), 512), np.float64)
    for template in TEMPLATES:
        total += embed([template.format(p) for p in prompts], tokenizer, session)
        print("embedded", template)
    total /= np.linalg.norm(total, axis=1, keepdims=True)

    # int8 per row; the worker renormalises, so the per-row scale is not stored.
    scale = 127 / np.abs(total).max(axis=1, keepdims=True)
    index = np.round(total * scale).astype(np.int8)
    check = index.astype(np.float64)
    check /= np.linalg.norm(check, axis=1, keepdims=True)
    print("int8 fidelity (min cosine):", round(float((check * total).sum(1).min()), 4))

    meta = {
        "format": "int8",
        "dimensions": 512,
        "labels": labels,
        "levels": levels,
    }
    assets = json.loads((OUT / "assets.json").read_text(encoding="utf-8"))
    write_parts("labels", json.dumps(meta, ensure_ascii=False).encode("utf-8"), assets, "labels")
    write_parts("index", index.tobytes(), assets, "index")
    (OUT / "assets.json").write_text(json.dumps(assets, separators=(",", ":")), encoding="utf-8")
    print(f"{len(labels)} words:", {lv: levels.count(lv) for lv in sorted(set(levels))})


if __name__ == "__main__":
    main()
