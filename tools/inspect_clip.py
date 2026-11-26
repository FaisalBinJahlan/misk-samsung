
import onnxruntime as ort

model_path = "tools/clip-vit-base-patch32/onnx/vision_model_int8.onnx"

session = ort.InferenceSession(
    model_path,
    providers=["CPUExecutionProvider"]
)

print("Inputs:")
for item in session.get_inputs():
    print(item.name, item.shape, item.type)

print("\nOutputs:")
for item in session.get_outputs():
    print(item.name, item.shape, item.type)
