
from pathlib import Path
import json

import numpy as np
import torch
from transformers import AutoProcessor, SiglipModel


# Project paths
ROOT = Path(__file__).absolute().parents[2]

MODEL_DIR = ROOT / "tools" / "siglip" / "model"
WORDS_FILE = ROOT / "tools" / "kids-words.txt"
OUTPUT_DIR = ROOT / "vendor" / "siglip" / "index"

# Model settings
DEVICE = "cpu"
BATCH_SIZE = 32

# Text templates
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
    "a photo of my {}."
]


def load_words():
    labels = []
    prompts = []

    with open(WORDS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # Skip empty lines and comments/headings
            if not line or line.startswith("#"):
                continue

            # Parse label | prompt
            if "|" in line:
                label, prompt = line.split("|", 1)
                label = label.strip()
                prompt = prompt.strip()
            else:
                label = line
                prompt = line

            if not label or not prompt:
                continue

            labels.append(label)
            prompts.append(prompt)

    if not labels:
        raise ValueError("The vocabulary file is empty.")

    if len(labels) != len(set(labels)):
        raise ValueError("Duplicate labels found in vocabulary.")

    return labels, prompts



def main():

    print("Loading vocabulary...")

    labels, prompts = load_words()

    print(f"Number of words: {len(labels)}")
    print("First 10 labels:", labels[:10])

    print("\nLoading SigLIP...")

    processor = AutoProcessor.from_pretrained(
        MODEL_DIR,
        local_files_only=True
    )

    model = SiglipModel.from_pretrained(
        MODEL_DIR,
        local_files_only=True
    )

    model.to(DEVICE)
    model.eval()

    print("\nGenerating text embeddings...")

    all_embeddings = []

    for template in TEMPLATES:

        print(f"Processing template: {template}")

        template_embeddings = []

        for start in range(0, len(labels), BATCH_SIZE):

            batch_words = prompts[start:start + BATCH_SIZE]

            texts = [
                template.format(word)
                for word in batch_words
            ]

            inputs = processor(
                text=texts,
                padding="max_length",
                return_tensors="pt"
            )

            inputs = {
                key: value.to(DEVICE)
                for key, value in inputs.items()
            }

            with torch.no_grad():

                features = model.get_text_features(**inputs)

                if not isinstance(features, torch.Tensor):
                    features = features.pooler_output

                features = torch.nn.functional.normalize(
                    features,
                    dim=-1
                )

            template_embeddings.append(
                features.cpu().numpy()
            )

        template_embeddings = np.concatenate(
            template_embeddings,
            axis=0
        )

        all_embeddings.append(template_embeddings)

    # Average embeddings from all templates
    print("\nAveraging embeddings...")

    embeddings = np.mean(
        np.stack(all_embeddings, axis=0),
        axis=0
    )

    # Normalize final embeddings
    norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True
    )

    embeddings = embeddings / np.maximum(norms, 1e-12)

    embeddings = embeddings.astype(np.float32)
    
    assert embeddings.shape[0] == len(labels)
    assert embeddings.shape[1] == 768
    assert np.isfinite(embeddings).all()

    # Save index
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    np.save(
        OUTPUT_DIR / "embeddings.npy",
        embeddings
    )

    with open(
        OUTPUT_DIR / "labels.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            labels,
            f,
            ensure_ascii=False,
            indent=2
        )

    metadata = {
        "model": "google/siglip-base-patch16-224",
        "embedding_dimension": int(embeddings.shape[1]),
        "number_of_words": len(labels),
        "number_of_templates": len(TEMPLATES),
        "dtype": "float32"
    }
    
    

    with open(
        OUTPUT_DIR / "metadata.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2
        )

    print("\nIndex built successfully!")
    print("Embedding shape:", embeddings.shape)
    print("Output directory:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
