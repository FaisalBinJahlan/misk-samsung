from pathlib import Path
import onnx

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "vendor/siglip/onnx/image_encoder.onnx"

model = onnx.load(
    str(MODEL_PATH),
    load_external_data=False
)

locations = set()

for initializer in model.graph.initializer:
    for item in initializer.external_data:
        if item.key == "location":
            locations.add(item.value)

print("External data locations:")

for location in sorted(locations):
    print("-", location)

print("\nNumber of initializers:", len(model.graph.initializer))
print("ONNX file size:", MODEL_PATH.stat().st_size)