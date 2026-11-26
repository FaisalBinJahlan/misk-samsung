
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).absolute().parents[2]

INDEX_DIR = ROOT / "vendor" / "siglip" / "index"

embeddings = np.load(INDEX_DIR / "embeddings.npy")
labels = json.loads(
    (INDEX_DIR / "labels.json").read_text(encoding="utf-8")
)

if embeddings.shape != (len(labels), 768):
    raise ValueError("Embeddings and labels dimensions do not match.")

# Browser-friendly little-endian float32
embeddings = np.asarray(embeddings, dtype="<f4")

output_path = INDEX_DIR / "embeddings.bin"
embeddings.tofile(output_path)

metadata = {
    "model": "google/siglip-base-patch16-224",
    "number_of_words": len(labels),
    "embedding_dimension": 768,
    "dtype": "float32",
    "byte_order": "little-endian",
    "format": "raw-binary"
}

(INDEX_DIR / "web_metadata.json").write_text(
    json.dumps(metadata, indent=2),
    encoding="utf-8"
)

print("Web index created successfully!")
print("Labels:", len(labels))
print("Shape:", embeddings.shape)
print("Binary size:", output_path.stat().st_size)
print("Output:", output_path)
