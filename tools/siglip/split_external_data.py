
from pathlib import Path
import json
import hashlib

ROOT = Path(__file__).absolute().parents[2]

SOURCE = ROOT / "vendor/siglip/onnx/image_encoder.onnx.data"
OUTPUT_DIR = ROOT / "vendor/siglip/onnx/external_parts"

CHUNK_SIZE = 10 * 1024 * 1024  # 10 MB


def sha256_file(path):
    digest = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def main():
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_size = SOURCE.stat().st_size
    parts = []

    print("Source size:", total_size)
    print("Splitting external weights...")

    with open(SOURCE, "rb") as source:
        index = 0

        while True:
            data = source.read(CHUNK_SIZE)

            if not data:
                break

            name = f"weights-{index:03d}.bin"
            output_path = OUTPUT_DIR / name

            with open(output_path, "wb") as output:
                output.write(data)

            parts.append({
                "name": name,
                "bytes": len(data)
            })

            print(name, len(data), "bytes")
            index += 1

    manifest = {
        "source": "image_encoder.onnx.data",
        "total_bytes": total_size,
        "chunk_size": CHUNK_SIZE,
        "parts": parts,
        "sha256": sha256_file(SOURCE)
    }

    with open(OUTPUT_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\nSplitting completed.")
    print("Number of parts:", len(parts))
    print("Manifest:", OUTPUT_DIR / "manifest.json")


if __name__ == "__main__":
    main()
