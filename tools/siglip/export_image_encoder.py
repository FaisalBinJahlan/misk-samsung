
import torch
import onnx
import onnxruntime as ort

from pathlib import Path
from transformers import SiglipModel

# Project paths
ROOT = Path(__file__).absolute().parents[2]
MODEL_DIR = ROOT / "tools" / "siglip" / "model"
OUTPUT_DIR = ROOT / "vendor" / "siglip" / "onnx"
OUTPUT_PATH = OUTPUT_DIR / "image_encoder.onnx"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading SigLIP model...")

model = SiglipModel.from_pretrained(str(MODEL_DIR))
model.eval()


class SigLIPImageEncoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        features = self.model.get_image_features(
            pixel_values=pixel_values
        )

        if not isinstance(features, torch.Tensor):
            features = features.pooler_output

        features = torch.nn.functional.normalize(
            features, dim=-1
        )

        return features


encoder = SigLIPImageEncoder(model)
encoder.eval()

# Dummy input: batch, RGB channels, height, width
dummy_input = torch.randn(1, 3, 224, 224)

print("Exporting image encoder to ONNX...")

with torch.no_grad():
    torch.onnx.export(
        encoder,
        (dummy_input,),
        str(OUTPUT_PATH),
        input_names=["pixel_values"],
        output_names=["image_embeddings"],
        dynamic_axes={
            "pixel_values": {0: "batch_size"},
            "image_embeddings": {0: "batch_size"},
        },
        opset_version=17,
        do_constant_folding=True,
        export_params=True
    )

print("Checking ONNX model...")

onnx_model = onnx.load(str(OUTPUT_PATH))
onnx.checker.check_model(onnx_model)

print("Testing ONNX Runtime...")

session = ort.InferenceSession(
    str(OUTPUT_PATH),
    providers=["CPUExecutionProvider"]
)

with torch.no_grad():
    pytorch_output = encoder(dummy_input).numpy()

onnx_output = session.run(
    ["image_embeddings"],
    {"pixel_values": dummy_input.numpy()}
)[0]

difference = abs(pytorch_output - onnx_output).max()

print("\nExport completed successfully!")
print("PyTorch output shape:", pytorch_output.shape)
print("ONNX output shape:", onnx_output.shape)
print("Maximum difference:", difference)
print("ONNX file:", OUTPUT_PATH)
print("ONNX file size:", round(OUTPUT_PATH.stat().st_size / (1024**2), 2), "MB")
