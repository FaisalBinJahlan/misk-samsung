from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).absolute().parents[2]
INDEX_DIR = ROOT / "vendor" / "siglip" / "index"

embeddings = np.load(INDEX_DIR / "embeddings.npy")

with open(INDEX_DIR / "labels.json", "r", encoding="utf-8") as f:
    labels = json.load(f)

with open(INDEX_DIR / "metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

print("Labels:", len(labels))
print("Embeddings shape:", embeddings.shape)
print("Metadata:", metadata)

assert len(labels) == 1000
assert embeddings.shape == (1000, 768)
assert np.isfinite(embeddings).all()

print("\nIndex verification successful!")