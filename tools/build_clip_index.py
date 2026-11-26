"""Build the kids vocabulary index with CLIP ViT-B/32 text embeddings.

Usage:
  python tools/build_clip_index.py tools/clip-vit-base-patch32

Requires: numpy, onnxruntime, tokenizers.
The output is written to vendor/clip-vit-base-patch32/index/.
It is intentionally separate from the existing MobileCLIP index.
"""
import gzip
import json
from pathlib import Path
import sys

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
WORDS = ROOT / "tools" / "kids-words.txt"
OUT = ROOT / "vendor" / "clip-vit-base-patch32" / "index"
PART_BYTES = 512 * 1024
MAX_LENGTH = 77
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
    return ([labels[i] for i in order], [prompts[i] for i in order],
            [levels[i] for i in order])


def make_inputs(session, rows, tokenizer):
    # CLIP uses the end-of-text token as its padding token.
    pad_id = tokenizer.token_to_id("<|endoftext|>")
    if pad_id is None:
        pad_id = tokenizer.token_to_id("</s>")
    if pad_id is None:
        raise RuntimeError("Could not find CLIP end-of-text token in tokenizer.json")

    ids = []
    for text in rows:
        token_ids = tokenizer.encode(text).ids[:MAX_LENGTH]
        ids.append(token_ids + [pad_id] * (MAX_LENGTH - len(token_ids)))
    ids = np.asarray(ids, dtype=np.int64)

    feeds = {}
    for item in session.get_inputs():
        if item.name == "input_ids":
            feeds[item.name] = ids
        elif item.name == "attention_mask":
            feeds[item.name] = (ids != pad_id).astype(np.int64)
        else:
            raise RuntimeError(f"Unexpected CLIP text input: {item.name}")
    return feeds


def embed(texts, tokenizer, session, batch=32):
    chunks = []
    for start in range(0, len(texts), batch):
        feeds = make_inputs(session, texts[start:start + batch], tokenizer)
        outputs = session.run(None, feeds)
        vectors = np.asarray(outputs[0], dtype=np.float32)
        if vectors.ndim != 2:
            raise RuntimeError(
                f"Expected pooled text embeddings [batch, dim], got {vectors.shape}"
            )
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        chunks.append(vectors / np.maximum(norms, 1e-12))
    return np.concatenate(chunks, axis=0)


def write_parts(name, data):
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob(f"{name}-*.bin"):
        old.unlink()
    packed = gzip.compress(data, compresslevel=9, mtime=0)
    parts = []
    for i in range(0, len(packed), PART_BYTES):
        filename = f"{name}-{i // PART_BYTES:02d}.bin"
        chunk = packed[i:i + PART_BYTES]
        (OUT / filename).write_bytes(chunk)
        parts.append({"name": filename, "bytes": len(chunk)})
    return {"bytes": len(packed), "expandedBytes": len(data), "parts": parts}


def main():
    model_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "tools" / "clip-vit-base-patch32"
    text_model = model_dir / "onnx" / "text_model_int8.onnx"
    tokenizer_file = model_dir / "tokenizer.json"
    if not text_model.exists() or not tokenizer_file.exists():
        raise SystemExit(
            "Missing CLIP files. Run: python tools/download_clip.py"
        )

    labels, prompts, levels = read_words()
    tokenizer = Tokenizer.from_file(str(tokenizer_file))
    session = ort.InferenceSession(str(text_model), providers=["CPUExecutionProvider"])
    total = None
    for template in TEMPLATES:
        vectors = embed([template.format(p) for p in prompts], tokenizer, session)
        total = vectors if total is None else total + vectors
        print("embedded", template, flush=True)

    total /= np.maximum(np.linalg.norm(total, axis=1, keepdims=True), 1e-12)
    dimensions = int(total.shape[1])
    scale = 127 / np.maximum(np.abs(total).max(axis=1, keepdims=True), 1e-12)
    index = np.round(total * scale).astype(np.int8)
    check = index.astype(np.float32)
    check /= np.maximum(np.linalg.norm(check, axis=1, keepdims=True), 1e-12)
    fidelity = float((check * total).sum(axis=1).min())

    metadata = {
        "format": "int8",
        "model": "CLIP ViT-B/32",
        "dimensions": dimensions,
        "labels": labels,
        "levels": levels,
    }
    manifest = {
        "model": "CLIP ViT-B/32",
        "source": "Xenova/clip-vit-base-patch32",
        "dimensions": dimensions,
        "labels": write_parts("labels", json.dumps(metadata, ensure_ascii=False).encode("utf-8")),
        "index": write_parts("index", index.tobytes()),
    }
    (OUT / "assets.json").write_text(
        json.dumps(manifest, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(f"\nBuilt {len(labels)} labels; embedding dimension={dimensions}")
    print(f"Minimum INT8 cosine fidelity: {fidelity:.4f}")
    print("Output:", OUT)


if __name__ == "__main__":
    main()
