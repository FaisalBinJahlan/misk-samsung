"""Download only the files needed to build and run CLIP ViT-B/32 locally.

Usage:
  pip install huggingface_hub
  python tools/download_clip.py

The vision encoder is used by the browser. The text encoder is only needed
when building the vocabulary index and does not need to be deployed.
"""
from pathlib import Path
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "tools" / "clip-vit-base-patch32"
REPO = "Xenova/clip-vit-base-patch32"
FILES = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "preprocessor_config.json",
    "onnx/text_model_int8.onnx",
    "onnx/vision_model_int8.onnx",
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for filename in FILES:
        print(f"Downloading {filename} ...", flush=True)
        path = hf_hub_download(
            repo_id=REPO,
            filename=filename,
            local_dir=OUT,
        )
        print(f"  saved: {path}")
    print("\nCLIP files are ready in:", OUT)


if __name__ == "__main__":
    main()
