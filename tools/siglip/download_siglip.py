
from transformers import AutoProcessor, SiglipModel
import torch

MODEL_ID = "google/siglip-base-patch16-224"

MODEL_DIR = "tools/siglip/model"

print("Downloading SigLIP model...")

processor = AutoProcessor.from_pretrained(MODEL_ID)

model = SiglipModel.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32
)

processor.save_pretrained(MODEL_DIR)
model.save_pretrained(MODEL_DIR)

print("SigLIP downloaded successfully!")
print("Model saved at:", MODEL_DIR)
