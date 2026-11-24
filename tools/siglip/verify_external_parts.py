
from pathlib import Path
import json
import hashlib

ROOT = Path(__file__).absolute().parents[2]

SOURCE = ROOT / "vendor/siglip/onnx/image_encoder.onnx.data"
PARTS_DIR = ROOT / "vendor/siglip/onnx/external_parts"
MANIFEST = PARTS_DIR / "manifest.json"


def sha256_file(path):
    digest = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def main():
    with open(MANIFEST, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    expected_hash = manifest["sha256"]
    expected_size = manifest["total_bytes"]

    combined_hash = hashlib.sha256()
    total_size = 0

    print("Verifying parts...")

    for part in manifest["parts"]:
        path = PARTS_DIR / part["name"]

        if not path.exists():
            raise FileNotFoundError(path)

        actual_size = path.stat().st_size

        if actual_size != part["bytes"]:
            raise ValueError(f"Size mismatch: {part['name']}")

        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                combined_hash.update(chunk)

        total_size += actual_size
        print("OK:", part["name"])

    actual_hash = combined_hash.hexdigest()
    original_hash = sha256_file(SOURCE)

    print("\nOriginal size:", expected_size)
    print("Combined size:", total_size)

    print("\nManifest SHA256:", expected_hash)
    print("Original SHA256:", original_hash)
    print("Combined SHA256:", actual_hash)

    assert total_size == expected_size
    assert original_hash == expected_hash
    assert actual_hash == original_hash

    print("\nExternal weights verification successful!")


if __name__ == "__main__":
    main()
