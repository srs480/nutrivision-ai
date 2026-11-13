"""
Verifies the exported ONNX model graph input/output tensor shapes using ONNX Runtime.
"""

from pathlib import Path
import numpy as np
import onnxruntime as ort


def verify_model() -> None:
    model_path = Path(__file__).resolve().parent.parent / "public" / "models" / "yolov8n.onnx"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run `python ml/export_model.py` first."
        )

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]

    print(f"Input Name : {input_meta.name}, Shape: {input_meta.shape}, Type: {input_meta.type}")
    print(f"Output Name: {output_meta.name}, Shape: {output_meta.shape}, Type: {output_meta.type}")

    dummy_input = np.zeros((1, 3, 640, 640), dtype=np.float32)
    outputs = session.run([output_meta.name], {input_meta.name: dummy_input})
    print(f"Verification Forward Pass Succeeded! Output Tensor Shape: {outputs[0].shape}")


if __name__ == "__main__":
    verify_model()